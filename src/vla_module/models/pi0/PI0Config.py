# Author : Shiven Saini
# Email : shiven.career@proton.me

"""
PI0Config Data
This file act as a Single source of truth for any config, hyperparameters etc PI0Policy may require.
"""

import torch
from torch import nn
from transformers.models.auto import CONFIG_MAPPING
from dataclasses import dataclass, field
from vla_module.training.optimizers.OptimizerConfigs import AdamWConfig
from vla_module.training.lr_schedulers.LRSchedulerConfigs import CosineDecayWithWarmupSchedulerConfig
from enum import Enum
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

# PI0 uses AdaRMS gating in some research variants, but defaults to disabled for both branches.
USE_ADARMS: tuple[bool, bool] = (False, False)
OBS_IMAGES = "observation.images"

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
    use_adarms = USE_ADARMS[0],
    adarms_cond_dim = gemma_language.width if USE_ADARMS[0] else None,
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
    use_adarms=USE_ADARMS[1],
    adarms_cond_dim=gemma_expert.width if USE_ADARMS[1] else None,
)

# @dataclass
# class PI0Config:
#     """Simplified PI0 Configuration containing only properties used in PI0Pytorch."""
#
#     # Model precision
#     dtype: Literal["bfloat16", "float32"] = "float32"
#     gradient_checkpointing: bool = False
#     device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     # Dimension configuration
#     max_action_dim: int = 32  # Maximum action vector dimension (with padding)
#     max_state_dim: int = 32  # Maximum state vector dimension (with padding)
#     chunk_size: int = 50  # Number of action steps to predict (action_horizon)
#
#     # Flow matching parameters for denoising
#     num_inference_steps: int = 10  # Number of denoising steps during inference
#     min_period: float = 4e-3  # Minimum period for sinusoidal timestep encoding
#     max_period: float = 4.0  # Maximum period for sinusoidal timestep encoding
#
#     image_resolution: tuple[int, int] = (224, 224)  # From openpi `preprocessing_pytorch.py`
#     n_action_steps: int = 50  # Number of action steps to execute
#
#     # Model optimization
#     compile_model: bool = False  # Whether to use torch.compile
#     compile_mode: str = "max-autotune"  # Torch compile mode
#
#     def __post_init__(self):
#         """Validate configuration after initialization."""
#         if self.dtype not in ["bfloat16", "float32"]:
#             raise ValueError(f"Invalid dtype: {self.dtype}. Must be 'bfloat16' or 'float32'")
#
#         if self.max_action_dim <= 0:
#             raise ValueError(f"max_action_dim must be positive, got {self.max_action_dim}")
#
#         if self.max_state_dim <= 0:
#             raise ValueError(f"max_state_dim must be positive, got {self.max_state_dim}")
#
#         if self.chunk_size <= 0:
#             raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
#
#         if self.num_inference_steps <= 0:
#             raise ValueError(f"num_inference_steps must be positive, got {self.num_inference_steps}")
#
#     def validate_features(self) -> None:
#         """Validate and set up input/output features."""
#         for i in range(self.empty_cameras):
#             key = f"{OBS_IMAGES}.empty_camera_{i}"
#             empty_camera = PolicyFeature(
#                 type=FeatureType.VISUAL,
#                 shape=(3, *self.image_resolution),  # Use configured image resolution
#             )
#             self.input_features[key] = empty_camera
#
#         if "observation.state" not in self.input_features:
#             state_feature = PolicyFeature(
#                 type=FeatureType.STATE,
#                 shape=(self.max_state_dim,),  # Padded to max_state_dim
#             )
#             self.input_features["observation.state"] = state_feature
#
#         if "action" not in self.output_features:
#             action_feature = PolicyFeature(
#                 type=FeatureType.ACTION,
#                 shape=(self.max_action_dim,),  # Padded to max_action_dim
#             )
#             self.output_features["action"] = action_feature

class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"
    LANGUAGE = "LANGUAGE"

class PipelineFeatureType(str, Enum):
    ACTION = "ACTION"
    OBSERVATION = "OBSERVATION"


class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"
    QUANTILES = "QUANTILES"
    QUANTILE10 = "QUANTILE10"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple

@dataclass
class PI0Config:

    paligemma_variant: str = "gemma_language"
    action_expert_variant: str = "gemma_action"
    dtype: str = "float32"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, or action horizon as per PI
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10  # Number of denoising steps during inference
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    # Add missing attributes that were inherited from PreTrainedConfig
    image_resolution: tuple[int, int] = (224, 224)  # Default image resolution (height, width)
    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = "cpu"  # Device to use for the model (None = auto-detect)

    # Optimizer settings: see openpi `AdamW``
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 48  # see openpi `__post_init__`

    def __post_init__(self):

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.paligemma_variant not in ["gemma_language", "gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_action", "gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        # Ensure feature specs are populated for downstream modules
        self.validate_features()

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features["action"] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38