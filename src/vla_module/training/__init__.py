"""VLA Module Training Components

This package provides comprehensive training functionality for VLA models including:
- Action-head-only fine-tuning with frozen VLM
- Full model fine-tuning
- LeRobot dataset integration
- Configurable action/observation spaces
- Checkpoint management
- Mixed precision training
"""

from vla_module.training.training_config import (
    ActionHeadTrainingConfig,
    ActionSpaceConfig,
    CheckpointConfig,
    DataConfig,
    FullTrainingConfig,
    LoggingConfig,
    ObservationSpaceConfig,
    OptimizerConfig,
    SchedulerConfig,
    create_default_action_head_config,
    create_default_full_training_config,
)

# Trainers
from vla_module.training.action_head_trainer import ActionHeadTrainer

# Freezing utilities
from vla_module.training.freeze_utils import (
    freeze_module,
    freeze_vision_encoder,
    freeze_language_model,
    freeze_vlm_backbone,
    setup_action_head_only_training,
    unfreeze_action_head,
    print_trainable_parameters,
    count_parameters,
)

# Checkpoint utilities
from vla_module.training.checkpoint_utils import (
    save_checkpoint,
    load_checkpoint,
    load_pretrained_model,
    cleanup_old_checkpoints,
    get_latest_checkpoint,
)

# Action/Observation space configuration
from vla_module.training.action_space_config import (
    ActionProjector,
    ActionSpaceAdapter,
    ObservationSpaceAdapter,
    StateProjector,
    create_action_space_config_from_dataset,
    create_observation_space_config_from_dataset,
)

__all__ = [
    # Training configurations
    "ActionHeadTrainingConfig",
    "FullTrainingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "DataConfig",
    "ActionSpaceConfig",
    "ObservationSpaceConfig",
    "create_default_action_head_config",
    "create_default_full_training_config",

    # Trainers
    "ActionHeadTrainer",

    # Freezing utilities
    "freeze_module",
    "freeze_vision_encoder",
    "freeze_language_model",
    "freeze_vlm_backbone",
    "setup_action_head_only_training",
    "unfreeze_action_head",
    "print_trainable_parameters",
    "count_parameters",

    # Checkpoint utilities
    "save_checkpoint",
    "load_checkpoint",
    "load_pretrained_model",
    "cleanup_old_checkpoints",
    "get_latest_checkpoint",

    # Action/Observation space
    "ActionProjector",
    "ActionSpaceAdapter",
    "ObservationSpaceAdapter",
    "StateProjector",
    "create_action_space_config_from_dataset",
    "create_observation_space_config_from_dataset",
]
