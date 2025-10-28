import torch
import builtins
import logging
import math, json
from torch import nn
from torch import Tensor
from pathlib import Path
from safetensors.torch import load_file
from typing import TypedDict, TypeVar, Self
from abc import ABC, abstractmethod
import torch.nn.functional as F

from vla_module.models.pi0.PaliGemmaWithActionExpert import PaliGemmaWithActionExpert
from vla_module.models.pi0.PI0Config import (
    OPENPI_ATTENTION_MASK_VALUE,
    PI0Config,
    action_expert_config,
    USE_ADARMS,
    vlm_config,
)
from vla_module.models.pi0.helpers import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    pad_vector,
    resize_with_pad_torch,
)

OBS_STR = "observation"
OBS_PREFIX = OBS_STR + "."
OBS_ENV_STATE = OBS_STR + ".environment_state"
OBS_STATE = OBS_STR + ".state"
OBS_IMAGE = OBS_STR + ".image"
OBS_IMAGES = OBS_IMAGE + "s"
OBS_LANGUAGE = OBS_STR + ".language"
OBS_LANGUAGE_TOKENS = OBS_LANGUAGE + ".tokens"
OBS_LANGUAGE_ATTENTION_MASK = OBS_LANGUAGE + ".attention_mask"


class PI0Pytorch(nn.Module):
    """Core PI0 PyTorch model - Inference Only."""

    def __init__(self, config: PI0Config):
        super().__init__()
        self.config = config


        self.paligemma_with_expert = PaliGemmaWithActionExpert(
            vlm_config,
            action_expert_config,
            use_adarms=list(USE_ADARMS),
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)

        self.state_proj = nn.Linear(config.max_state_dim, action_expert_config.width)
        self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
        self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        """Sample Gaussian noise for denoising process."""
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def embed_prefix(
            self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.paligemma_with_expert.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.int32, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        state_emb = self.state_proj(state)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        x = self.action_time_mlp_in(action_time_emb)
        x = F.silu(x)
        action_time_emb = self.action_time_mlp_out(x)
        adarms_cond = None

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.int32, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing on underlying transformer modules."""
        for module in (
            self.paligemma_with_expert.paligemma,
            self.paligemma_with_expert.gemma_expert,
        ):
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()

    @torch.no_grad()
    def sample_actions(
            self, images, img_masks, lang_tokens, lang_masks, state, noise=None, num_steps=None
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = state.shape[0]
        device = state.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_kv=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time += dt

        return x_t

    def denoise_step(
            self,
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_kv=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class PI0Policy(nn.Module):
        """PI0 Policy - standalone implementation without PreTrainedPolicy."""

        config_class = PI0Config
        name = "pi0"

        def __init__(self, config: PI0Config, **kwargs):
            super().__init__()
            self.config = config

            # Ensure config-derived feature specs are populated for downstream lookups
            self.config.validate_features()

            # Initialize the core PI0 model
            self.model = PI0Pytorch(config)

            # Enable gradient checkpointing if requested
            if config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()

            self.model.to(config.device)
            self.reset()

        def save_pretrained(self, save_directory: str | Path) -> None:
            """Save model weights and config."""
            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)

            # Save config
            config_path = save_directory / "config.json"
            with open(config_path, "w") as f:
                # Convert dataclass to dict and save
                import dataclasses
                config_dict = dataclasses.asdict(self.config)
                # Handle device serialization
                if isinstance(config_dict.get('device'), torch.device):
                    config_dict['device'] = str(config_dict['device'])
                json.dump(config_dict, f, indent=4)

            # Save model weights
            model_to_save = self.module if hasattr(self, "module") else self
            from safetensors.torch import save_file
            state_dict = model_to_save.state_dict()
            save_file(state_dict, str(save_directory / "model.safetensors"))

            print(f"Model saved to {save_directory}")

        @classmethod
        def load_stored_weights(
                cls,
                weights_file: str | Path = "model.safetensors",
                config_file: str | Path | None = None,
                config: PI0Config | None = None,
                strict: bool = False,
        ) -> Self:
            """
            Load model weights from a local safetensors file and return a model instance.

            Args:
                weights_file: Path to the safetensors file (e.g., "model.safetensors" or "/path/to/model.safetensors")
                config_file: Optional path to config.json file. If None, looks for config.json in same directory as weights_file
                config: Optional PI0Config instance. If provided, uses this instead of loading from file
                strict: Whether to strictly enforce state dict key matching

            Returns:
                PI0Policy instance with loaded weights

            Example:
                # Load from same directory
                policy = PI0Policy.load_stored_weights("model.safetensors")

                # Load from specific path
                policy = PI0Policy.load_stored_weights("/path/to/model.safetensors")

                # Load with custom config
                config = PI0Config(device="cuda", chunk_size=50)
                policy = PI0Policy.load_stored_weights("model.safetensors", config=config)
            """
            weights_path = Path(weights_file)
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")

            # Load or use provided config
            if config is None:
                # Determine config file path
                if config_file is None:
                    # Look for config.json in same directory as weights file
                    config_path = weights_path.parent / "config.json"
                else:
                    config_path = Path(config_file)

                if not config_path.exists():
                    raise FileNotFoundError(
                        f"Config file not found: {config_path}. "
                        f"Please provide a config file or pass a PI0Config instance."
                    )

                print(f"Loading config from: {config_path}")
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                    # Handle device deserialization
                    if 'device' in config_dict:
                        config_dict['device'] = torch.device(config_dict['device'])
                    config = PI0Config(**config_dict)

            # Create instance
            print(f"Creating PI0Policy instance with config...")
            instance = cls(config)

            # Load weights
            print(f"Loading weights from: {weights_path}")
            from safetensors.torch import load_file
            original_state_dict = load_file(str(weights_path))

            # Fix any key differences
            fixed_state_dict = instance._fix_pytorch_state_dict_keys(original_state_dict, None)

            # Remap keys: add "model." prefix if not present
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"✓ Remapped {remap_count} keys with 'model.' prefix")

            # Load state dict into the inner model
            missing_keys, unexpected_keys = instance.model.load_state_dict(
                remapped_state_dict,
                strict=strict
            )

            # Report loading status
            if missing_keys:
                print(f"⚠ Missing {len(missing_keys)} keys")
                for key in missing_keys[:3]:
                    print(f"  - {key}")
                if len(missing_keys) > 3:
                    print(f"  ... and {len(missing_keys) - 3} more")

            if unexpected_keys:
                print(f"⚠ Unexpected {len(unexpected_keys)} keys")
                for key in unexpected_keys[:3]:
                    print(f"  - {key}")
                if len(unexpected_keys) > 3:
                    print(f"  ... and {len(unexpected_keys) - 3} more")

            if not missing_keys and not unexpected_keys:
                print("✓ All keys loaded successfully!")

            # Move to device and set to eval mode
            instance.model.to(config.device)
            instance.eval()

            print(f"✓ Model loaded and ready on device: {config.device}")
            return instance

        def _fix_pytorch_state_dict_keys(self, state_dict, model_config):
            """Fix state dict keys to match current model architecture."""
            import re

            fixed_state_dict = {}

            for key, value in state_dict.items():
                new_key = key

                # Handle layer norm structure changes
                if re.match(
                        r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                        key,
                ):
                    expert_uses_adarms = getattr(
                        self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                    )
                    if expert_uses_adarms:
                        logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                        continue

                if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                    expert_uses_adarms = getattr(
                        self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                    )
                    if expert_uses_adarms:
                        logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                        continue

                # Handle MLP naming changes
                if key.startswith("time_mlp_in."):
                    new_key = key.replace("time_mlp_in.", "action_time_mlp_in.")
                elif key.startswith("time_mlp_out."):
                    new_key = key.replace("time_mlp_out.", "action_time_mlp_out.")

                if "patch_embedding" in key:
                    logging.warning(f"Vision embedding key might need handling: {key}")

                fixed_state_dict[new_key] = value

            return fixed_state_dict

        def get_optim_params(self) -> dict:
            """Returns parameters for optimizer."""
            return self.model.parameters()

        def reset(self):
            """Reset internal state - called when environment resets."""
            from collections import deque
            self._action_queue = deque(maxlen=self.config.chunk_size)
            self._queues = {
                "action": deque(maxlen=self.config.chunk_size),
            }


        def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
            """Preprocess images for the model.

            Images from LeRobotDataset are typically in [B, C, H, W] format and normalized to [0, 1].
            PaliGemma VLM expects images in [B, C, H, W] format and normalized to [-1, 1].
            """
            images = []
            img_masks = []

            # Get device from model parameters
            device = next(self.parameters()).device

            present_img_keys = [key for key in self.config.image_features if key in batch]
            missing_img_keys = [key for key in self.config.image_features if key not in batch]

            if len(present_img_keys) == 0:
                raise ValueError(
                    f"All image features are missing from the batch. At least one expected. "
                    f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
                )

            for key in present_img_keys:
                img = batch[key]

                # Ensure tensor is on the same device as the model
                if img.device != device:
                    img = img.to(device)

                # Ensure float32 dtype for consistency
                if img.dtype != torch.float32:
                    img = img.to(torch.float32)

                # from openpi preprocess_observation_pytorch: Handle both [B, C, H, W] and [B, H, W, C] formats
                is_channels_first = img.shape[1] == 3  # Check if channels are in dimension 1

                if is_channels_first:
                    # Convert [B, C, H, W] to [B, H, W, C] for processing
                    img = img.permute(0, 2, 3, 1)

                # from openpi preprocess_observation_pytorch: Resize with padding if needed
                if img.shape[1:3] != self.config.image_resolution:
                    img = resize_with_pad_torch(img, *self.config.image_resolution)

                # Normalize from [0,1] to [-1,1] as expected by siglip
                img = img * 2.0 - 1.0

                # from openpi preprocess_observation_pytorch: Convert back to [B, C, H, W] format if it was originally channels-first
                if is_channels_first:
                    img = img.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

                images.append(img)
                # Create mask (all ones for real images)
                bsize = img.shape[0]
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
                img_masks.append(mask)

            # Create image features not present in the batch as fully 0 padded images
            for _num_empty_cameras in range(len(missing_img_keys)):
                img = torch.ones_like(img) * -1  # padded with -1 for SigLIP
                mask = torch.zeros_like(mask)  # mask is zero for empty cameras
                images.append(img)
                img_masks.append(mask)

            return images, img_masks

        def prepare_state(self, batch):
            """Pad state"""
            state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
            return state

        def prepare_action(self, batch):
            """Pad action"""
            action_key = "action"
            if action_key not in batch:
                legacy_key = action_key.upper()
                if legacy_key in batch:
                    action_key = legacy_key
                else:
                    raise KeyError(
                        f"Action key '{action_key}' not found in batch (tried '{legacy_key}' as well)."
                    )

            actions = pad_vector(batch[action_key], self.config.max_action_dim)
            return actions

        @torch.no_grad()
        def select_action(self, batch: dict[str, Tensor]) -> Tensor:
            """Select a single action given environment observations."""
            self.eval()

            # Action queue logic for n_action_steps > 1
            if len(self._action_queue) == 0:
                actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
                # Transpose to get shape (n_action_steps, batch_size, action_dim)
                self._action_queue.extend(actions.transpose(0, 1))

            return self._action_queue.popleft()

        @torch.no_grad()
        def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
            """Predict a chunk of actions given environment observations."""
            self.eval()

            # Prepare inputs
            images, img_masks = self._preprocess_images(batch)
            lang_tokens, lang_masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
            state = self.prepare_state(batch)

            # Sample actions using the model
            actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)

            # Unpad actions to actual action dimension
            original_action_dim = self.config.output_features["action"].shape[0]
            actions = actions[:, :, :original_action_dim]

            return actions

        def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
            """Run the batch through the model and compute the loss for training."""

            # Prepare inputs
            images, img_masks = self._preprocess_images(batch)
            lang_tokens, lang_masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
            state = self.prepare_state(batch)
            actions = self.prepare_action(batch)

            # Compute loss
            losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions)

            # Truncate losses to actual action dimensions
            original_action_dim = self.config.output_features["action"].shape[0]
            losses = losses[:, :, :original_action_dim]

            loss = losses.mean()

            loss_dict = {
                "loss": loss.item(),
                "loss_per_dim": losses.mean(dim=[0, 1]).detach().cpu().numpy().tolist(),
            }

            return loss, loss_dict
        

if __name__ == "__main__":
    # Example: load weights when running this file directly.
    policy = PI0Policy.load_stored_weights("model.safetensors")
