"""SmolVLM with Action Expert Model

Dual-model architecture combining SmolVLM (vision-language model) with a smaller
action expert for robotic control. Adapted from LeRobot's SmolVLA implementation.
"""

import copy
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLMForConditionalGeneration,
)

from .helpers import apply_rope, get_intermediate_size


class SmolVLMWithExpertModel(nn.Module):
    """SmolVLM with Action Expert dual-model architecture.
    
    This model combines:
    1. SmolVLM: Vision-language model for visual and language understanding
    2. Action Expert: Smaller language model specialized for action prediction
    
    The two models can interact via:
    - Self-attention: Both models process independently
    - Cross-attention: Action expert attends to VLM's key-value cache
    """
    
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = False,
        attention_mode: str = "self_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = -1,
        self_attn_every_n_layers: int = -1,
        expert_width_multiplier: float = 0.5,
        device: str = "auto",
    ):
        """Initialize SmolVLM with Action Expert.
        
        Args:
            model_id: HuggingFace model ID for SmolVLM
            load_vlm_weights: Whether to load pre-trained VLM weights
            train_expert_only: Whether to freeze VLM and train only action expert
            freeze_vision_encoder: Whether to freeze vision encoder
            attention_mode: 'self_attn' or 'cross_attn'
            num_expert_layers: Number of layers in action expert (-1 = same as VLM)
            num_vlm_layers: Number of VLM layers to use (-1 = all)
            self_attn_every_n_layers: Interleave self-attention every N layers in cross-attn mode
            expert_width_multiplier: Action expert hidden size relative to VLM
            device: Device to load model on
        """
        super().__init__()
        
        # Load VLM model
        if load_vlm_weights:
            print(f"Loading {model_id} weights...")
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
            )
            config = self.vlm.config
        else:
            config = AutoConfig.from_pretrained(model_id)
            self.vlm = SmolVLMForConditionalGeneration(config=config)
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Optionally reduce VLM layers
        if num_vlm_layers > 0:
            print(f"Reducing VLM layers to {num_vlm_layers}...")
            self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
        
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.config = config
        
        # Create action expert configuration
        lm_expert_config = copy.deepcopy(config.text_config)
        hidden_size = lm_expert_config.hidden_size
        
        # Scale down expert hidden size
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)
        lm_expert_config.intermediate_size = get_intermediate_size(int(hidden_size * expert_width_multiplier))
        lm_expert_config.num_hidden_layers = self.num_vlm_layers
        
        # Optionally reduce expert layers
        if num_expert_layers > 0:
            assert self.num_vlm_layers % num_expert_layers == 0, (
                f"VLM layers ({self.num_vlm_layers}) must be multiple of expert layers ({num_expert_layers})"
            )
            lm_expert_config.num_hidden_layers = num_expert_layers
        
        # Create expert model
        self.lm_expert = AutoModel.from_config(lm_expert_config)
        self.num_expert_layers = len(self.lm_expert.layers)
        self.self_attn_every_n_layers = self_attn_every_n_layers
        
        # Reshape expert projections for cross-attention
        if "cross" in attention_mode:
            for layer_idx in range(len(self.lm_expert.layers)):
                # Skip self-attention layers
                if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
                    continue
                
                # Reshape k_proj and v_proj to match VLM key-value dimensions
                self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
                self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
        
        # Remove unused embed_tokens from expert
        self.lm_expert.embed_tokens = None
        
        # Store attention configuration
        self.num_attention_heads = self.config.text_config.num_attention_heads
        self.num_key_value_heads = self.config.text_config.num_key_value_heads
        
        # Store training configuration
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_mode = attention_mode
        self.expert_hidden_size = lm_expert_config.hidden_size
        
        # Set requires_grad flags
        self.set_requires_grad()
    
    def get_vlm_model(self):
        """Get the underlying VLM text model."""
        return self.vlm.model
    
    def set_requires_grad(self):
        """Set requires_grad flags based on training configuration."""
        # Freeze vision encoder if requested
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False
        
        # Freeze VLM if training expert only
        if self.train_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False
        else:
            # Freeze specific layers to avoid unused params in distributed training
            last_layers = [self.num_vlm_layers - 1]
            if (
                self.num_vlm_layers != self.num_expert_layers
                and self.num_vlm_layers % self.num_expert_layers == 0
            ):
                last_layers.append(self.num_vlm_layers - 2)
            
            frozen_layers = [
                "lm_head",
                "text_model.model.norm.weight",
            ]
            for layer in last_layers:
                frozen_layers.append(f"text_model.model.layers.{layer}.")
            
            for name, params in self.vlm.named_parameters():
                if any(k in name for k in frozen_layers):
                    params.requires_grad = False
        
        # Freeze lm_head in expert to avoid unused params
        for name, params in self.lm_expert.named_parameters():
            if "lm_head" in name:
                params.requires_grad = False
    
    def train(self, mode: bool = True):
        """Set training mode, respecting freeze flags."""
        super().train(mode)
        
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
        
        if self.train_expert_only:
            self.vlm.eval()
    
    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Embed image using vision encoder and connector.
        
        Args:
            image: Image tensor of shape (B, C, H, W)
        
        Returns:
            Image embeddings of shape (B, num_patches, hidden_size)
        """
        patch_attention_mask = None
        
        # Get features from vision encoder
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                patch_attention_mask=patch_attention_mask,
            )
            .last_hidden_state
        )
        
        # Project to text model dimension
        image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
        
        return image_hidden_states
    
    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed language tokens using text model embeddings.
        
        Args:
            tokens: Token IDs of shape (B, L)
        
        Returns:
            Token embeddings of shape (B, L, hidden_size)
        """
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)
    
    def get_model_layers(self, models: list) -> list:
        """Get aligned layers from VLM and expert models.
        
        Args:
            models: List of [vlm_model, expert_model]
        
        Returns:
            List of [vlm_layers, expert_layers] where layers are aligned
        """
        vlm_layers = []
        expert_layers = []
        multiple_of = self.num_vlm_layers // self.num_expert_layers if self.num_expert_layers > 0 else 1
        
        for i in range(self.num_vlm_layers):
            # Add VLM layer
            vlm_layers.append(models[0].layers[i])
            
            # Add expert layer (or None if skipping)
            if multiple_of > 1 and i > 0 and i % multiple_of != 0:
                expert_layers.append(None)
            else:
                expert_layer_index = i // multiple_of if multiple_of > 0 else i
                expert_layers.append(models[1].layers[expert_layer_index])
        
        return [vlm_layers, expert_layers]
    
    def forward_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ) -> tuple[list[torch.Tensor], dict]:
        """Forward pass through attention layer (self-attention mode).
        
        Args:
            model_layers: List of [vlm_layers, expert_layers]
            inputs_embeds: List of input embeddings
            layer_idx: Current layer index
            position_ids: Position IDs for RoPE
            attention_mask: 2D attention mask
            batch_size: Batch size
            head_dim: Head dimension
            use_cache: Whether to use KV cache
            fill_kv_cache: Whether to fill KV cache
            past_key_values: Past key-value cache
        
        Returns:
            Tuple of (attention outputs, updated past_key_values)
        """
        query_states = []
        key_states = []
        value_states = []
        
        # Compute Q, K, V for all input embeddings
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue
            
            # Layer norm
            hidden_states = layer.input_layernorm(hidden_states)
            
            # Reshape for multi-head attention
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            
            # Project to Q, K, V
            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)
            
            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)
        
        # Concatenate along sequence dimension
        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)
        
        seq_len = query_states.shape[1]
        
        # Adjust position IDs and attention mask if needed
        if seq_len < position_ids.shape[1]:
            _position_ids = position_ids[:, :seq_len]
            _attention_mask = attention_mask[:, :seq_len, :seq_len]
        else:
            _position_ids = position_ids
            _attention_mask = attention_mask
        
        # Apply RoPE
        query_states = apply_rope(query_states, _position_ids)
        key_states = apply_rope(key_states, _position_ids)
        
        # Initialize cache if needed
        if use_cache and past_key_values is None:
            past_key_values = {}
        
        # Update cache
        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                # Concatenate with past
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat([past_key_values[layer_idx]["value_states"], value_states], dim=1)
        
        # Compute attention
        attention_interface = self.get_attention_interface()
        att_output = attention_interface(
            _attention_mask, batch_size, head_dim, query_states, key_states, value_states
        )
        
        return [att_output], past_key_values
    
    def forward_cross_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ) -> tuple[list[torch.Tensor], dict]:
        """Forward pass through attention layer (cross-attention mode).
        
        In cross-attention mode, the action expert attends to VLM's key-value cache.
        
        Args:
            model_layers: List of [vlm_layers, expert_layers]
            inputs_embeds: List of [prefix_embeds, suffix_embeds]
            layer_idx: Current layer index
            position_ids: Position IDs for RoPE
            attention_mask: 2D attention mask
            batch_size: Batch size
            head_dim: Head dimension
            use_cache: Whether to use KV cache
            fill_kv_cache: Whether to fill KV cache
            past_key_values: Past key-value cache
        
        Returns:
            Tuple of (attention outputs, updated past_key_values)
        """
        attention_interface = self.get_attention_interface()
        att_outputs = []
        
        # Process prefix (VLM) if both embeddings provided and no cache
        if len(inputs_embeds) == 2 and not past_key_values:
            seq_len = inputs_embeds[0].shape[1]
            position_id = position_ids[:, :seq_len]
            expert_position_id = position_ids[:, seq_len:]
            prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]
            
            layer = model_layers[0][layer_idx]
            
            # Compute prefix attention
            hidden_states = layer.input_layernorm(inputs_embeds[0])
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            
            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)
            
            # Apply RoPE
            query_states = apply_rope(query_state, position_id)
            key_states = apply_rope(key_state, position_id)
            
            # Compute attention
            att_output = attention_interface(
                prefix_attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )
            att_outputs.append(att_output)
        else:
            expert_position_id = position_ids
        
        # Initialize cache if needed
        if use_cache and past_key_values is None:
            past_key_values = {}
        
        # Update cache
        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                # Use cached values
                key_states = past_key_values[layer_idx]["key_states"]
                value_states = past_key_values[layer_idx]["value_states"]
        
        # Process expert attention
        expert_layer = model_layers[1][layer_idx]
        if expert_layer is not None:
            expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])
            
            expert_input_shape = expert_hidden_states.shape[:-1]
            expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim)
            
            # Compute expert queries
            expert_hidden_states = expert_hidden_states.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape)
            
            # Project VLM key-value to expert dimension
            _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).view(
                *key_states.shape[:2], -1
            )
            expert_key_states = expert_layer.self_attn.k_proj(_key_states).view(
                *_key_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )
            
            _value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).view(
                *value_states.shape[:2], -1
            )
            expert_value_states = expert_layer.self_attn.v_proj(_value_states).view(
                *_value_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )
            
            # Adjust expert position IDs to start from 0
            expert_position_id = (
                expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values
            )
            
            # Get expert attention mask
            expert_attention_mask = attention_mask[
                :, -inputs_embeds[1].shape[1]:, :expert_key_states.shape[1]
            ]
            
            # Apply RoPE to expert queries
            expert_query_states = apply_rope(expert_query_state, expert_position_id)
            
            # Compute expert attention
            att_output = attention_interface(
                expert_attention_mask,
                batch_size,
                head_dim,
                expert_query_states,
                expert_key_states,
                expert_value_states,
            )
            att_outputs.append(att_output)
        else:
            att_outputs.append(None)
        
        return att_outputs, past_key_values
    
    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ) -> tuple[list[torch.Tensor], dict]:
        """Forward pass through dual-model architecture.
        
        Args:
            attention_mask: 2D attention mask
            position_ids: Position IDs for RoPE
            past_key_values: Past key-value cache
            inputs_embeds: List of input embeddings
            use_cache: Whether to use KV cache
            fill_kv_cache: Whether to fill KV cache
        
        Returns:
            Tuple of (output embeddings, past_key_values)
        """
        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = self.get_model_layers(models)
        
        # Get batch size from inputs
        for hidden_states in inputs_embeds:
            if hidden_states is not None:
                batch_size = hidden_states.shape[0]
                break
        
        # Process all layers
        num_layers = self.num_vlm_layers
        head_dim = self.vlm.config.text_config.head_dim
        
        for layer_idx in range(num_layers):
            # Choose attention type
            use_self_attn = (
                fill_kv_cache
                or "cross" not in self.attention_mode
                or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
            )
            
            if use_self_attn:
                att_outputs, past_key_values = self.forward_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            else:
                att_outputs, past_key_values = self.forward_cross_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            
            # Process attention outputs through o_proj, residual, and FFN
            outputs_embeds = []
            start = 0
            
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                
                if hidden_states is not None:
                    if layer is None:
                        outputs_embeds.append(hidden_states)
                        continue
                    
                    end = start + hidden_states.shape[1]
                    
                    # Output projection
                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    att_out = att_output[:, start:end]
                    out_emb = layer.self_attn.o_proj(att_out)
                    
                    # First residual connection
                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()
                    
                    # FFN
                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)
                    
                    # Second residual connection
                    out_emb += after_first_residual
                    
                    outputs_embeds.append(out_emb)
                    
                    start = end if len(att_outputs) == 1 else 0
                else:
                    outputs_embeds.append(None)
            
            inputs_embeds = outputs_embeds
        
        # Final layer normalization
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        
        return outputs_embeds, past_key_values
    
    def get_attention_interface(self):
        """Get attention implementation (eager by default)."""
        return self.eager_attention_forward
    
    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        """Eager (non-flash) attention implementation.
        
        Args:
            attention_mask: 2D attention mask
            batch_size: Batch size
            head_dim: Head dimension
            query_states: Query states of shape (B, L, H, D)
            key_states: Key states of shape (B, L_kv, H_kv, D)
            value_states: Value states of shape (B, L_kv, H_kv, D)
        
        Returns:
            Attention output of shape (B, L, H * D)
        """
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads
        
        sequence_length = key_states.shape[1]
        
        # Expand key-value heads to match attention heads (GQA)
        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )
        
        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )
        
        # Upcast to float32 for attention computation
        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)
        
        # Transpose for attention: (B, L, H, D) -> (B, H, L, D)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        
        # Compute attention scores
        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim ** -0.5
        
        # Apply attention mask
        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        
        # Softmax
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)
        
        # Apply attention to values
        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
        
        # Reshape: (B, H, L, D) -> (B, L, H * D)
        att_output = att_output.permute(0, 2, 1, 3)
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)
        
        return att_output

