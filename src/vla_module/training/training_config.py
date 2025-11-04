from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    log_every_n_steps: int = 10
    """Log metrics every N steps"""
    
    use_wandb: bool = False
    """Whether to use Weights & Biases"""
    
    wandb_project: Optional[str] = None
    """W&B project name"""
    
    wandb_entity: Optional[str] = None
    """W&B entity/team name"""
    
    wandb_run_name: Optional[str] = None
    """W&B run name"""
    
    log_gradients: bool = False
    """Whether to log gradient statistics"""
    
    log_weights: bool = False
    """Whether to log weight statistics"""



@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    
    name: str = "adamw"
    """Optimizer name: 'adamw', 'adam', 'sgd'"""
    
    lr: float = 1e-4
    """Learning rate"""
    
    weight_decay: float = 1e-4
    """Weight decay (L2 regularization)"""
    
    betas: Tuple[float, float] = (0.9, 0.999)
    """Adam/AdamW betas"""
    
    eps: float = 1e-8
    """Epsilon for numerical stability"""
    
    momentum: float = 0.9
    """Momentum for SGD"""
    
    gradient_clip_norm: float = 1.0
    """Gradient clipping norm (0 to disable)"""


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""
    
    name: str = "cosine_with_warmup"
    """Scheduler name: 'cosine_with_warmup', 'linear_warmup', 'constant', 'step'"""
    
    warmup_steps: int = 1000
    """Number of warmup steps"""
    
    total_steps: Optional[int] = None
    """Total training steps (auto-computed if None)"""
    
    min_lr: float = 1e-6
    """Minimum learning rate for cosine schedule"""
    
    step_size: int = 10000
    """Step size for step scheduler"""
    
    gamma: float = 0.1
    """Multiplicative factor for step scheduler"""

@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""
    
    save_dir: Path = Path("checkpoints")
    """Directory to save checkpoints"""
    
    save_every_n_steps: int = 1000
    """Save checkpoint every N steps"""
    
    keep_last_n: int = 3
    """Keep only last N checkpoints"""
    
    save_optimizer_state: bool = True
    """Whether to save optimizer state"""
    
    save_scheduler_state: bool = True
    """Whether to save scheduler state"""


@dataclass
class DataConfig:
    """Configuration for data loading."""
    
    repo_id: str = "lerobot/pusht"
    """HuggingFace dataset repository ID"""
    
    root: Optional[Path] = None
    """Local dataset cache directory"""
    
    split: str = "train"
    """Dataset split"""
    
    batch_size: int = 8
    """Batch size per device"""
    
    num_workers: int = 4
    """Number of data loading workers"""
    
    shuffle: bool = True
    """Whether to shuffle training data"""
    
    pin_memory: bool = True
    """Whether to pin memory for faster GPU transfer"""
    
    # Image configuration
    image_keys: List[str] = field(default_factory=lambda: ["observation.images.top"])
    """Image observation keys"""
    
    resize_images: Optional[Tuple[int, int]] = (512, 512)
    """Image resize dimensions"""
    
    # Temporal configuration  
    n_obs_steps: int = 1
    """Number of observation steps"""
    
    chunk_size: int = 50
    """Action chunk size"""


@dataclass
class ActionHeadTrainingConfig:
    """Configuration for action-head-only fine-tuning.
    
    This mode freezes the VLM backbone and only trains the action prediction head.
    """
    
    # Model configuration
    model_type: str = "smolvla"
    """Model type: 'smolvla' or 'pi0'"""
    
    model_path: Optional[str] = None
    """Path to pretrained model checkpoint"""
    
    freeze_vision: bool = True
    """Freeze vision encoder"""
    
    freeze_language: bool = True
    """Freeze language model"""
    
    train_projections: bool = True
    """Train projection layers (state_proj, etc.)"""
    
    # Action/observation space
    max_action_dim: int = 32
    """Maximum action dimension"""
    
    max_state_dim: int = 32
    """Maximum state dimension"""
    
    use_action_projection: bool = False
    """Use learned action projection (vs simple padding)"""
    
    use_state_projection: bool = False
    """Use learned state projection (vs simple padding)"""
    
    # Training hyperparameters
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    """Optimizer configuration"""
    
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    """Scheduler configuration"""
    
    num_epochs: int = 10
    """Number of training epochs"""
    
    max_steps: Optional[int] = None
    """Maximum training steps (overrides num_epochs if set)"""
    
    gradient_accumulation_steps: int = 1
    """Number of gradient accumulation steps"""
    
    mixed_precision: str = "no"
    """Mixed precision training: 'no', 'fp16', 'bf16'"""
    
    # Data configuration
    data: DataConfig = field(default_factory=DataConfig)
    """Data loading configuration"""
    
    # Checkpointing and logging
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    """Checkpoint configuration"""
    
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    """Logging configuration"""
    
    # Evaluation, ful
    eval_every_n_steps: int = 500
    """Evaluate every N steps"""
    
    eval_split: str = "validation"
    """Evaluation dataset split"""
    
    # Misc
    seed: int = 42
    """Random seed"""
    
    output_dir: Path = Path("outputs/action_head_training")
    """Output directory"""
    
    resume_from_checkpoint: Optional[Path] = None
    """Path to checkpoint to resume from"""


@dataclass
class FullTrainingConfig:
    """Configuration for full model fine-tuning.
    
    This mode trains both the VLM backbone and action head.
    """
    
    # Model configuration
    model_type: str = "smolvla"
    """Model type: 'smolvla' or 'pi0'"""
    
    model_path: Optional[str] = None
    """Path to pretrained model checkpoint"""
    
    freeze_vision: bool = False
    """Freeze vision encoder"""
    
    freeze_language: bool = False
    """Freeze language model"""
    
    freeze_first_n_layers: Optional[int] = None
    """Freeze first N layers of language model"""
    
    # Action/observation space
    max_action_dim: int = 32
    """Maximum action dimension"""
    
    max_state_dim: int = 32
    """Maximum state dimension"""
    
    use_action_projection: bool = False
    """Use learned action projection"""
    
    use_state_projection: bool = False
    """Use learned state projection"""
    
    # Training hyperparameters
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(
        lr=5e-5,  # Lower LR for full fine-tuning
        weight_decay=1e-4,
    ))
    """Optimizer configuration"""
    
    scheduler: SchedulerConfig = field(default_factory=lambda: SchedulerConfig(
        warmup_steps=2000,  # More warmup for full fine-tuning
    ))
    """Scheduler configuration"""
    
    num_epochs: int = 20
    """Number of training epochs"""
    
    max_steps: Optional[int] = None
    """Maximum training steps"""
    
    gradient_accumulation_steps: int = 4
    """Number of gradient accumulation steps"""
    
    mixed_precision: str = "bf16"
    """Mixed precision training: 'no', 'fp16', 'bf16'"""
    
    # Data configuration
    data: DataConfig = field(default_factory=DataConfig)
    """Data loading configuration"""
    
    # Checkpointing and logging
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    """Checkpoint configuration"""
    
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    """Logging configuration"""
    
    # Evaluation
    eval_every_n_steps: int = 1000
    """Evaluate every N steps"""
    
    eval_split: str = "validation"
    """Evaluation dataset split"""
    
    # Misc
    seed: int = 42
    """Random seed"""
    
    output_dir: Path = Path("outputs/full_training")
    """Output directory"""
    
    resume_from_checkpoint: Optional[Path] = None
    """Path to checkpoint to resume from"""
    
    # Progressive unfreezing
    progressive_unfreezing: bool = False
    """Enable progressive unfreezing of layers"""
    
    unfreeze_schedule: Optional[List[Tuple[int, int]]] = None
    """List of (step, num_layers_to_unfreeze) tuples"""
