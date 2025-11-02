"""SmolVLA Policy Implementation

Complete implementation of SmolVLA policy for vision-language-action learning,
adapted from HF SmolVLA pAPER. Uses flow matching for action generation.

Architecture:
- VLM: SmolVLM for vision-language understanding
- Action Expert: Smaller language model for action prediction
- Training: Flow matching objective
- Inference: Euler denoising

Based on: https://huggingface.co/papers/2506.01844
"""

import math
from collections import deque
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .SmolVLAConfig import SmolVLAConfig
from .SmolVLMWithExpert import SmolVLMWithExpertModel
from .helpers import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_tensor,
    pad_vector,
    resize_with_pad,
    aloha_gripper_to_angular,
    aloha_gripper_from_angular,
    aloha_gripper_from_angular_inv,
)

# Constants for observation/action keys
OBS_STATE = "observation.state"
OBS_LANGUAGE_TOKENS = "observation.language_tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language_attention_mask"
ACTION = "action"


class VLAFlowMatching(nn.Module):
    """SmolVLA Flow Matching Model.
    
    This is the core model that implements flow matching for action generation.
    It combines SmolVLM (vision-language model) with an action expert.
    """
    
    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config
        
        # Dual-model architecture: VLM + Action Expert
        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device,
        )
        
        # Projection layers
        self.state_proj = nn.Linear(
            self.config.max_state_dim, 
            self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(
            self.config.max_action_dim, 
            self.vlm_with_expert.expert_hidden_size
        )
        self.action_out_proj = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, 
            self.config.max_action_dim
        )
        
        # Action-time fusion MLP
        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2,
            self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size,
            self.vlm_with_expert.expert_hidden_size
        )
        
        # Set requires_grad flags
        self.set_requires_grad()
        
        # Special tokens
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        
        # Configuration
        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.prefix_length = self.config.prefix_length
    
    def set_requires_grad(self):
        """Set requires_grad flags based on configuration."""
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj
    
    def sample_noise(self, shape: tuple, device: torch.device) -> Tensor:
        """Sample noise from standard normal distribution.
        
        Args:
            shape: Shape of noise tensor
            device: Device to create tensor on
        
        Returns:
            Noise tensor
        """
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise
    
    def sample_time(self, bsize: int, device: torch.device) -> Tensor:
        """Sample timesteps from Beta distribution.
        
        Args:
            bsize: Batch size
            device: Device to create tensor on
        
        Returns:
            Time tensor of shape (bsize,)
        """
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        # Scale to [0.001, 1.0]
        time = time_beta * 0.999 + 0.001
        return time
    
    def embed_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed prefix (images, language, state) for VLM processing.
        
        Args:
            images: List of image tensors
            img_masks: List of image padding masks
            lang_tokens: Language token IDs
            lang_masks: Language attention masks
            state: State vector (optional)
        
        Returns:
            Tuple of (embeddings, padding_masks, attention_masks)
        """
        embs = []
        pad_masks = []
        att_masks = []
        
        # Embed images
        for img, img_mask in zip(images, img_masks, strict=False):
            # Add image special tokens if configured
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)
            
            # Embed image
            img_emb = self.vlm_with_expert.embed_image(img)
            
            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim ** 0.5, dtype=img_emb.dtype, device=img_emb.device)
            
            # Expand image mask
            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)
            
            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * num_img_embs
            
            # Add image end token if configured
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        
        # Embed language tokens
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs
        
        # Embed state
        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        
        bsize = state_emb.shape[0]
        device = state_emb.device
        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        
        # State tokens cannot be attended to by images/language (causal mask)
        att_masks += [1] * states_seq_len
        
        # Concatenate all embeddings
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]
        
        # Pad to fixed prefix length if configured
        seq_len = pad_masks.shape[1]
        if self.prefix_length > 0 and seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)
        
        att_masks = att_masks.expand(bsize, -1)
        
        return embs, pad_masks, att_masks
    
    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Embed suffix (actions, timestep) for Expert processing.
        
        Args:
            noisy_actions: Noisy action tensor
            timestep: Time tensor
        
        Returns:
            Tuple of (embeddings, padding_masks, attention_masks)
        """
        embs = []
        pad_masks = []
        att_masks = []
        
        # Embed actions
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        
        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)
        
        # Fuse action and time embeddings
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        
        # Apply MLP
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        
        embs.append(action_time_emb)
        
        # Create padding mask
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)
        
        # Action tokens cannot be attended to by prefix (causal mask)
        att_masks += [1] * self.config.chunk_size
        
        # Concatenate
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        
        return embs, pad_masks, att_masks
    
    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None
    ) -> Tensor:
        """Training forward pass with flow matching objective.
        
        Args:
            images: List of image tensors
            img_masks: List of image padding masks
            lang_tokens: Language token IDs
            lang_masks: Language attention masks
            state: State vector
            actions: Ground truth actions
            noise: Optional noise (sampled if None)
            time: Optional time (sampled if None)
        
        Returns:
            Loss tensor of shape (batch_size, chunk_size, action_dim)
        """

        # Sample noise and time if not provided
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        
        # Flow matching interpolation: x_t = t * noise + (1-t) * actions
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        
        # Velocity field (ground truth): u_t = noise - actions
        u_t = noise - actions
        
        # Embed prefix (images, language, state)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        
        # Embed suffix (noisy actions, timestep)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)
        
        # Concatenate prefix and suffix
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        
        # Create 2D attention mask
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        
        # Forward through dual-model
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        
        # Extract action predictions
        suffix_out = suffix_out[:, -self.config.chunk_size:]
        
        # Upcast to float32 for stability
        suffix_out = suffix_out.to(dtype=torch.float32)
        
        # Project to action space
        v_t = self.action_out_proj(suffix_out)
        
        # Compute MSE loss: ||u_t - v_t||^2
        losses = F.mse_loss(u_t, v_t, reduction="none")
        
        return losses
    
    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        noise: Tensor | None = None
    ) -> Tensor:
        """Inference forward pass to sample actions via denoising.
        
        Args:
            images: List of image tensors
            img_masks: List of image padding masks
            lang_tokens: Language token IDs
            lang_masks: Language attention masks
            state: State vector
            noise: Optional noise (sampled if None)
        
        Returns:
            Sampled actions of shape (batch_size, chunk_size, action_dim)
        """
        bsize = state.shape[0]
        device = state.device
        
        # Sample noise if not provided
        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)
        
        # Embed prefix and compute KV cache
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        
        # Fill KV cache with prefix
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        
        # Euler denoising
        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            
            # Euler step: x_{t+dt} = x_t + dt * v_t
            x_t += dt * v_t
            time += dt
        
        return x_t
    
    def denoise_step(
        self,
        prefix_pad_masks: Tensor,
        past_key_values: dict,
        x_t: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        """Apply one denoising step.
        
        Args:
            prefix_pad_masks: Prefix padding masks
            past_key_values: Cached key-values from prefix
            x_t: Current noisy actions
            timestep: Current timestep
        
        Returns:
            Velocity field v_t
        """
        # Embed suffix
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)
        
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        
        # Create prefix padding mask for suffix
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        
        # Create suffix attention mask
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        
        # Concatenate masks
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        
        # Compute position IDs
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        
        # Forward through expert (using cached prefix)
        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        
        # Extract suffix output
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size:]
        
        # Upcast to float32
        suffix_out = suffix_out.to(dtype=torch.float32)
        
        # Project to action space
        v_t = self.action_out_proj(suffix_out)
        
        return v_t


class SmolVLAPolicy(nn.Module):
    """SmolVLA Policy wrapper for training and inference.
    
    This class wraps the VLAFlowMatching model and provides:
    - Image preprocessing
    - State/action padding
    - Action queue management for multi-step execution
    - Aloha-specific adaptations (optional)
    """
    
    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        config.validate_features()
        self.config = config
        
        # Create model
        self.model = VLAFlowMatching(config)
        
        # Reset internal state
        self.reset()
    
    def reset(self):
        """Reset internal state (called when environment resets)."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
    
    def get_optim_params(self) -> Any:
        """Get parameters for optimizer."""
        return self.parameters()
    
    def prepare_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Prepare images for the model.
        
        Args:
            batch: Batch dictionary containing image observations
        
        Returns:
            Tuple of (images, image_masks)
        """
        images = []
        img_masks = []
        
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]
        
        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )
        
        # Process present images
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            
            # Resize with padding if configured
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
            
            # Normalize from [0, 1] to [-1, 1] for SigLIP
            img = img * 2.0 - 1.0
            
            # Create mask
            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            
            images.append(img)
            img_masks.append(mask)
        
        # Add empty cameras if needed
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        
        return images, img_masks
    
    def prepare_state(self, batch: dict[str, Tensor]) -> Tensor:
        """Prepare state vector.
        
        Args:
            batch: Batch dictionary containing state observation
        
        Returns:
            Padded state tensor
        """
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state
    
    def prepare_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Prepare action vector.
        
        Args:
            batch: Batch dictionary containing actions
        
        Returns:
            Padded action tensor
        """
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions
    
    def _pi_aloha_decode_state(self, state: Tensor) -> Tensor:
        """Decode state from Aloha format to PI format.
        
        Args:
            state: State in Aloha format
        
        Returns:
            State in PI format
        """
        # Flip specific joints
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        
        # Reverse gripper transformation
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        
        return state
    
    def _pi_aloha_encode_actions(self, actions: Tensor) -> Tensor:
        """Encode actions from PI format to Aloha format.
        
        Args:
            actions: Actions in PI format
        
        Returns:
            Actions in Aloha format
        """
        # Flip specific joints
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        
        # Apply gripper transformation
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        
        return actions
    
    def _pi_aloha_encode_actions_inv(self, actions: Tensor) -> Tensor:
        """Inverse of _pi_aloha_encode_actions.
        
        Args:
            actions: Actions in Aloha format
        
        Returns:
            Actions in PI format
        """
        # Flip specific joints
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        
        # Inverse gripper transformation
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        
        return actions
    
    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Prepare batch with Aloha adaptations if needed.
        
        Args:
            batch: Input batch
        
        Returns:
            Prepared batch
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
        
        return batch
    
    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Predict a chunk of actions.
        
        Args:
            batch: Batch dictionary
            noise: Optional noise for reproducibility
        
        Returns:
            Action chunk of shape (batch_size, chunk_size, action_dim)
        """
        self.eval()
        batch = self._prepare_batch(batch)
        
        # Prepare inputs
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        
        # Sample actions
        actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)
        
        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        
        # Apply Aloha encoding if needed
        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)
        
        return actions
    
    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action for environment execution.
        
        Args:
            batch: Batch dictionary
            noise: Optional noise for reproducibility
        
        Returns:
            Single action of shape (batch_size, action_dim)
        """
        self.eval()
        batch = self._prepare_batch(batch)
        
        # Populate action queue if empty
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise)
            
            # Transpose and add to queue
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        
        return self._queues[ACTION].popleft()
    
    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict]:
        """Training forward pass.
        
        Args:
            batch: Batch dictionary
            noise: Optional noise
            time: Optional time
        
        Returns:
            Tuple of (loss, loss_dict)
        """
        # Apply Aloha adaptations if needed
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])
        
        # Prepare inputs
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("actions_is_pad")
        
        # Forward pass
        losses = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, state, actions, noise, time
        )
        
        # Mask padding if provided
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
        
        # Remove padding dimensions
        losses = losses[:, :, :self.config.max_action_dim]
        
        # Compute mean loss
        loss = losses.mean()
        
        # Create loss dictionary
        loss_dict = {
            "loss": loss.item(),
            "mse": loss.item(),
        }
        
        return loss, loss_dict


