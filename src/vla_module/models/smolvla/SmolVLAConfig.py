"""SmolVLA Configuration

Configuration class for SmolVLA policy, adapted from LeRobot's SmolVLA implementation.
Designed by Hugging Face, SmolVLA uses flow matching for action generation with a dual-model
architecture: SmolVLM (vision-language model) + Action Expert (smaller language model).
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SmolVLAConfig:
    """Configuration for SmolVLA policy.
    
    SmolVLA Architecture:
    - VLM Backbone: SmolVLM (Vision-Language Model from HuggingFace)
    - Action Expert: Smaller language model for action prediction
    - Training: Flow matching objective for action generation
    - Attention: Cross-attention or self-attention between VLM and expert
    
    Key Features:
    - Flexible image preprocessing with padding
    - State and action padding to fixed dimensions
    - Multi-camera support with empty camera handling
    - Aloha-specific adaptations (optional)
    - Gradient checkpointing for memory efficiency
    """
    
    # Model Architecture for SmolVLA
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    """Pre-trained SmolVLM model from HuggingFace Hub"""
    
    load_vlm_weights: bool = False
    """Whether to load pre-trained VLM weights. Set to True if training expert from scratch."""
    
    attention_mode: str = "cross_attn"
    """Attention mode: 'cross_attn' for cross-attention between VLM and expert, 'self_attn' for self-attention"""
    
    num_expert_layers: int = -1
    """Number of layers in action expert. <= 0 means same as VLM. Otherwise expert has fewer layers."""
    
    num_vlm_layers: int = 16
    """Number of layers to use from the VLM (first num_vlm_layers layers)"""
    
    self_attn_every_n_layers: int = 2
    """Interleave self-attention layers every N layers in cross-attention mode"""
    
    expert_width_multiplier: float = 0.75
    """Action expert hidden size multiplier relative to VLM hidden size"""
    
    
    n_obs_steps: int = 1
    """Number of observation steps to use as input"""
    
    chunk_size: int = 50
    """Number of actions to predict in a single forward pass/action horizon"""
    
    n_action_steps: int = 50
    """Number of action steps to execute per model invocation"""
    
    max_state_dim: int = 32
    """Maximum state dimension (shorter vectors will be padded)"""
    
    max_action_dim: int = 32
    """Maximum action dimension (shorter vectors will be padded)"""
    
    # Image Processing
    resize_imgs_with_padding: tuple[int, int] | None = (512, 512)
    """Target image size with padding to preserve aspect ratio. None to disable resizing."""
    
    image_resolution: tuple[int, int] = (512, 512)
    """Input image resolution expected by the model"""
    
    add_image_special_tokens: bool = False
    """Whether to add special tokens around image features"""
    
    empty_cameras: int = 0
    """Number of empty cameras to add (useful for datasets with missing cameras)"""
    
    # Language Processing
    tokenizer_max_length: int = 48
    """Maximum length for language token sequences"""
    
    pad_language_to: str = "longest"
    """Padding strategy: 'longest' or 'max_length'"""
    
    prefix_length: int = -1
    """Fixed prefix length for padding. -1 means no fixed length."""
    
    # Flow Matching Configuration
    num_steps: int = 10
    """Number of denoising steps during inference"""
    
    min_period: float = 4e-3
    """Minimum period for sine-cosine positional encoding of timesteps"""
    
    max_period: float = 4.0
    """Maximum period for sine-cosine positional encoding of timesteps"""
    
    # Training Configuration
    freeze_vision_encoder: bool = True
    """Whether to freeze the vision encoder during training"""
    
    train_expert_only: bool = True
    """Whether to train only the action expert (freeze VLM)"""
    
    train_state_proj: bool = True
    """Whether to train the state projection layer"""
    
    # Optimizer settings
    optimizer_lr: float = 1e-4
    """Learning rate for AdamW optimizer"""
    
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    """Beta coefficients for AdamW optimizer"""
    
    optimizer_eps: float = 1e-8
    """Epsilon for AdamW optimizer"""
    
    optimizer_weight_decay: float = 1e-10
    """Weight decay for AdamW optimizer"""
    
    optimizer_grad_clip_norm: float = 10.0
    """Gradient clipping norm"""
    
    # Scheduler Settings
    scheduler_warmup_steps: int = 1_000
    """Number of warmup steps for learning rate scheduler"""
    
    scheduler_decay_steps: int = 30_000
    """Number of decay steps for learning rate scheduler"""
    
    scheduler_decay_lr: float = 2.5e-6
    """Final learning rate after decay"""
    # Inference settings
    use_cache: bool = True
    """Whether to use KV cache during inference"""
    
    # Adapted from HF Lerobot, maybe useful later.
    adapt_to_pi_aloha: bool = False
    """Convert joint/gripper values to PI internal runtime space (for Aloha compatibility)"""
    
    use_delta_joint_actions_aloha: bool = False
    """Convert joint dimensions to deltas relative to current state (for Aloha)"""
    
    # FeaturesConfiguration 
    input_features: dict[str, Any] = field(default_factory=dict)
    """Dictionary of input features (images, state, etc.)"""
    
    output_features: dict[str, Any] = field(default_factory=dict)
    """Dictionary of output features (actions, etc.)"""
    
    # Device Configuration
    device: str = "auto"        
    """Device to use for model: 'auto', 'cuda', 'cpu', etc."""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Perform configuration validation."""
        # Check n_action_steps vs chunk_size
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"chunk_size must be >= n_action_steps. "
                f"Got n_action_steps={self.n_action_steps}, chunk_size={self.chunk_size}"
            )
        
        # Check attention mode
        valid_attention_modes = ["cross_attn", "self_attn"]
        if self.attention_mode not in valid_attention_modes:
            raise ValueError(
                f"attention_mode must be one of {valid_attention_modes}. "
                f"Got: {self.attention_mode}"
            )
        
        # Check delta joint actions
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "use_delta_joint_actions_aloha is not yet implemented. "
                "This feature is used in SmolVLA for Aloha real robot models."
            )
        
        # Check image resolution
        if self.resize_imgs_with_padding is not None:
            if len(self.resize_imgs_with_padding) != 2:
                raise ValueError(
                    f"resize_imgs_with_padding must be a tuple of (width, height). "
                    f"Got: {self.resize_imgs_with_padding}"
                )
        
        # Check optimizer betas
        if len(self.optimizer_betas) != 2:
            raise ValueError(
                f"optimizer_betas must be a tuple of (beta1, beta2). "
                f"Got: {self.optimizer_betas}"
            )
    
    def validate_features(self):
        """Validate and setup input/output features.
        
        This method:
        1. Validates that required features are present
        2. Adds empty camera features if needed
        3. Ensures feature shapes are consistent
        """
        # Add empty camera features if needed
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            if key not in self.input_features:
                self.input_features[key] = {
                    "shape": [3, 480, 640],
                    "dtype": "float32"
                }
    
    @property
    def image_features(self) -> list[str]:
        """Get list of image feature keys from input_features."""
        return [
            key for key in self.input_features.keys()
            if "image" in key.lower() or "camera" in key.lower()
        ]
    
    @property
    def state_feature(self):
        """Get state feature configuration."""
        for key, value in self.input_features.items():
            if "state" in key.lower():
                return value
        return None
    
    @property
    def action_feature(self):
        """Get action feature configuration."""
        for key, value in self.output_features.items():
            if "action" in key.lower():
                return value
        return None
    
    @classmethod
    def create(
        cls,
        vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        input_features: dict[str, Any] | None = None,
        output_features: dict[str, Any] | None = None,
        **kwargs
    ) -> "SmolVLAConfig":
        """Factory method to create SmolVLAConfig with sensible defaults.
        
        Args:
            vlm_model_name: HuggingFace model ID for SmolVLM
            input_features: Dictionary of input features
            output_features: Dictionary of output features
            **kwargs: Additional configuration parameters
        
        Returns:
            SmolVLAConfig instance
        
        Example:
            >>> config = SmolVLAConfig.create(
            ...     vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            ...     input_features={
            ...         "observation.images.top": {"shape": [3, 224, 224], "dtype": "float32"},
            ...         "observation.state": {"shape": [7], "dtype": "float32"},
            ...     },
            ...     output_features={
            ...         "action": {"shape": [7], "dtype": "float32"},
            ...     },
            ... )
        """
        config = cls(
            vlm_model_name=vlm_model_name,
            input_features=input_features or {},
            output_features=output_features or {},
            **kwargs
        )
        config.validate_features()
        return config
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "vlm_model_name": self.vlm_model_name,
            "load_vlm_weights": self.load_vlm_weights,
            "attention_mode": self.attention_mode,
            "num_expert_layers": self.num_expert_layers,
            "num_vlm_layers": self.num_vlm_layers,
            "self_attn_every_n_layers": self.self_attn_every_n_layers,
            "expert_width_multiplier": self.expert_width_multiplier,
            "n_obs_steps": self.n_obs_steps,
            "chunk_size": self.chunk_size,
            "n_action_steps": self.n_action_steps,
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "resize_imgs_with_padding": self.resize_imgs_with_padding,
            "image_resolution": self.image_resolution,
            "add_image_special_tokens": self.add_image_special_tokens,
            "empty_cameras": self.empty_cameras,
            "tokenizer_max_length": self.tokenizer_max_length,
            "pad_language_to": self.pad_language_to,
            "prefix_length": self.prefix_length,
            "num_steps": self.num_steps,
            "min_period": self.min_period,
            "max_period": self.max_period,
            "freeze_vision_encoder": self.freeze_vision_encoder,
            "train_expert_only": self.train_expert_only,
            "train_state_proj": self.train_state_proj,
            "optimizer_lr": self.optimizer_lr,
            "optimizer_betas": self.optimizer_betas,
            "optimizer_eps": self.optimizer_eps,
            "optimizer_weight_decay": self.optimizer_weight_decay,
            "optimizer_grad_clip_norm": self.optimizer_grad_clip_norm,
            "scheduler_warmup_steps": self.scheduler_warmup_steps,
            "scheduler_decay_steps": self.scheduler_decay_steps,
            "scheduler_decay_lr": self.scheduler_decay_lr,
            "use_cache": self.use_cache,
            "adapt_to_pi_aloha": self.adapt_to_pi_aloha,
            "use_delta_joint_actions_aloha": self.use_delta_joint_actions_aloha,
            "input_features": self.input_features,
            "output_features": self.output_features,
            "device": self.device,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "SmolVLAConfig":
        """Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        
        Returns:
            SmolVLAConfig instance
        """
        return cls(**config_dict)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"SmolVLAConfig(\n"
            f"  vlm_model_name={self.vlm_model_name},\n"
            f"  chunk_size={self.chunk_size},\n"
            f"  n_action_steps={self.n_action_steps},\n"
            f"  num_vlm_layers={self.num_vlm_layers},\n"
            f"  num_expert_layers={self.num_expert_layers},\n"
            f"  expert_width_multiplier={self.expert_width_multiplier},\n"
            f"  train_expert_only={self.train_expert_only},\n"
            f"  freeze_vision_encoder={self.freeze_vision_encoder}\n"
            f")"
        )
    

    
