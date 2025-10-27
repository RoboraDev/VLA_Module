# Author : Shiven Saini
# Email : shiven.career@proton.me

"""
PI0Config Data
This file act as a Single source of truth for any config, hyperparameters etc PI0Policy may require.
"""

import torch
from torch import nn
from transformers.models.auto import CONFIG_MAPPING
from dataclasses import dataclass
from typing import Literal
from transformers.models.gemma import modeling_gemma
from transformers import (
    PaliGemmaForConditionalGeneration,
    GemmaForCausalLM
)




# GemmaConfig taken from OpenPI Repository
class GemmaConfig:
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


gemma_language = GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )

gemma_expert = GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )

# TODO: SHIVEN -> Need to take care of this, added just to see how much similarity I can decouple from each VLA Policy

# VLM Language part
_text_config = CONFIG_MAPPING["gemma"](
    hidden_size = gemma_language.width,
    intermediate_size = gemma_language.mlp_dim,
    num_attention_heads = gemma_language.num_heads,
    head_dim = gemma_language.head_dim,
    num_hidden_layers = gemma_language.depth,
    num_key_value_heads = gemma_language.num_kv_heads,
    hidden_activation = "gelu_pytorch_tanh",
    torch_dtype = "float32",
    vocab_size = 257152,
    use_adarms = use_adarms[0],
    adarms_cond_dim = gemma_language.width if use_adarms[0] else None,
)

# VLM vision tower aka siglip vision encoder
_vision_config = CONFIG_MAPPING["siglip_vision_model"](
    intermediate_size = 4304,
    projection_dim = 2048,
    projector_hidden_act = "gelu_fast",
    torch_dtype = "float32"
)

vlm_config = CONFIG_MAPPING["paligemma"](
    _vocab_size = 257152,  # noqa: SLF001
    image_token_index = 257152,
    text_config = _text_config,
    vision_config = _vision_config,
)


action_expert_config = CONFIG_MAPPING["gemma"](
    head_dim=gemma_expert.head_dim,
    hidden_size=gemma_expert.width,
    intermediate_size=gemma_expert.mlp_dim,
    num_attention_heads=gemma_expert.num_heads,
    num_hidden_layers=gemma_expert.depth,
    num_key_value_heads=gemma_expert.num_kv_heads,
    vocab_size=257152,
    hidden_activation="gelu_pytorch_tanh",
    torch_dtype="float32",
    use_adarms=use_adarms[1],
    adarms_cond_dim=gemma_expert.width if use_adarms[1] else None,
)

@dataclass
class PI0Config:
    """Simplified PI0 Configuration containing only properties used in PI0Pytorch."""

    # Model precision
    dtype: Literal["bfloat16", "float32"] = "float32"

    # Dimension configuration
    max_action_dim: int = 32  # Maximum action vector dimension (with padding)
    max_state_dim: int = 32  # Maximum state vector dimension (with padding)
    chunk_size: int = 50  # Number of action steps to predict (action_horizon)

    # Flow matching parameters for denoising
    num_inference_steps: int = 10  # Number of denoising steps during inference
    min_period: float = 4e-3  # Minimum period for sinusoidal timestep encoding
    max_period: float = 4.0  # Maximum period for sinusoidal timestep encoding

    # Model optimization
    compile_model: bool = False  # Whether to use torch.compile
    compile_mode: str = "max-autotune"  # Torch compile mode

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}. Must be 'bfloat16' or 'float32'")

        if self.max_action_dim <= 0:
            raise ValueError(f"max_action_dim must be positive, got {self.max_action_dim}")

        if self.max_state_dim <= 0:
            raise ValueError(f"max_state_dim must be positive, got {self.max_state_dim}")

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

        if self.num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {self.num_inference_steps}")

OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38