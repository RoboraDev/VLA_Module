"""# VLA Training Module

Comprehensive training implementation for Vision-Language-Action (VLA) models with LeRobot dataset integration.

## Overview

This module provides complete training functionality for fine-tuning VLA models (SmolVLA, PI0) on robotic manipulation tasks using LeRobot datasets. It supports two main training modes:

1. **Action Head Only**: Fast fine-tuning with frozen VLM backbone
2. **Full Model**: Complete fine-tuning of all model components

## Features

- LeRobot dataset integration with automatic preprocessing
-  Configurable action/observation space (default 32 dims, easily modified)
-  Action head only training with VLM freezing
-  Full model fine-tuning with progressive unfreezing
-  Mixed precision training (FP16/BF16)
-  Gradient accumulation for larger effective batch sizes
-  Checkpoint management with automatic cleanup
-  State projection layers for dimension adaptation
-  Multi-camera image support
-  Comprehensive logging and metrics

## Quick Start

### Installation

Ensure you have the required dependencies:

```bash
pip install torch transformers accelerate datasets huggingface-hub
pip install lerobot  # For dataset pipeline support
```

### Basic Usage

```python
from vla_module.training import (
    ActionHeadTrainer,
    create_default_action_head_config,
)

# Create configuration
config = create_default_action_head_config(
    repo_id="lerobot/aloha_sim_insertion_human",
    model_type="smolvla",
    model_path="HuggingFaceTB/SmolVLA-Base",
)

# Customize as needed
config.data.batch_size = 16
config.optimizer.lr = 1e-4
config.num_epochs = 10
config.mixed_precision = "bf16"

# Load your model
model = load_your_smolvla_model(config.model_path)

# Create trainer and train
trainer = ActionHeadTrainer(config, model, device="cuda")
trainer.train()
```

### Command Line Interface

```bash
python -m vla_module.training.train_vla \\
    --model-type smolvla \\
    --model-path HuggingFaceTB/SmolVLA-Base \\
    --dataset lerobot/aloha_sim_insertion_human \\
    --training-mode action_head_only \\
    --batch-size 16 \\
    --num-epochs 10 \\
    --lr 1e-4 \\
    --mixed-precision bf16 \\
    --output-dir outputs/smolvla_aloha \\
    --checkpoint-dir checkpoints/smolvla_aloha
```

## Architecture

### Components

1. **Dataset Adapter** (`data/lerobot_adapter.py`)
   - Bridges LeRobot datasets with VLA models
   - Handles multi-camera images
   - Pads/projects actions and states to model dimensions

2. **Action Space Configuration** (`training/action_space_config.py`)
   - Configurable action/observation dimensions
   - Learned projection layers for dimension adaptation
   - Normalization with dataset statistics

3. **Freezing Utilities** (`training/freeze_utils.py`)
   - Freeze vision encoder
   - Freeze language model
   - Selective layer freezing
   - Parameter counting

4. **Training Configurations** (`training/training_config.py`)
   - ActionHeadTrainingConfig
   - FullTrainingConfig
   - Optimizer, scheduler, checkpoint configs

5. **Checkpoint Utilities** (`training/checkpoint_utils.py`)
   - Save/load checkpoints with full training state
   - Best model tracking
   - Automatic cleanup

6. **Trainers** (`training/action_head_trainer.py`)
   - ActionHeadTrainer: Frozen VLM, train action head
   - FullModelTrainer: Train entire model (coming soon)

## Training Modes

### Mode 1: Action Head Only

**Best for**: Quick task adaptation with limited data

**Characteristics**:
- Freezes VLM backbone (99% of parameters)
- Trains only action head (1% of parameters)
- Fast training: 1-2 hours on single GPU
- Requires less data

**Example**:
```python
config = create_default_action_head_config(
    repo_id="lerobot/pusht",
    model_type="smolvla",
    model_path="HuggingFaceTB/SmolVLA-Base",
)
config.freeze_vision = True
config.freeze_language = True
config.train_projections = True  # Train projection layers

trainer = ActionHeadTrainer(config, model)
trainer.train()
```

### Mode 2: Full Model Fine-Tuning

**Best for**: New robot morphology or significantly different data

**Characteristics**:
- Trains all parameters
- Slower: 10-20 hours on single GPU
- More flexible adaptation
- Requires larger dataset

**Example** (coming soon):
```python
config = create_default_full_training_config(
    repo_id="your/custom_dataset",
    model_type="smolvla",
)
config.freeze_vision = False
config.freeze_language = False
config.progressive_unfreezing = True

trainer = FullModelTrainer(config, model)
trainer.train()
```

## Modifying Action/Observation Space

### Changing Dimensions

Both SmolVLA and PI0 default to 32 dimensions. To change:

```python
# In training config
config.max_action_dim = 64  # Change from 32 to 64
config.max_state_dim = 64

# Or in dataset adapter
from vla_module.data.lerobot_adapter import DatasetAdapterConfig

adapter_config = DatasetAdapterConfig(
    repo_id="your/dataset",
    max_action_dim=64,
    max_state_dim=64,
)
```

### Using Learned Projections

Instead of simple padding:

```python
config.use_action_projection = True
config.use_state_projection = True
```

This creates small neural networks that learn optimal mappings between robot and model dimensions.

### Custom Action Components

Extract specific action components (joints, gripper):

```python
from vla_module.training.action_space_config import ActionSpaceConfig

action_config = ActionSpaceConfig(
    original_dim=7,
    max_dim=32,
    joint_dims=[0, 1, 2, 3, 4, 5],  # First 6 dims are joints
    gripper_dims=[6],  # Last dim is gripper
)
```

## Dataset Integration

### Supported Datasets

Any LeRobot dataset on HuggingFace Hub:
- `lerobot/aloha_sim_insertion_human`
- `lerobot/pusht`
- `lerobot/xarm_lift_medium`
- Your custom datasets following LeRobot format

### Multi-Camera Support

```python
config.data.image_keys = [
    "observation.images.top",
    "observation.images.wrist",
    "observation.images.left",
    "observation.images.right",
]
```

The adapter:
- Automatically handles missing cameras
- Resizes with aspect ratio preservation
- Normalizes to [-1, 1] range

### Image Preprocessing

```python
config.data.resize_images = (512, 512)  # Width, height
config.data.n_obs_steps = 1  # Temporal window
```

## Advanced Configuration

### Gradient Accumulation

For larger effective batch sizes:

```python
config.data.batch_size = 8
config.gradient_accumulation_steps = 4
# Effective batch size = 8 * 4 = 32
```

### Mixed Precision Training

Enable for 2-3x speedup on modern GPUs:

```python
config.mixed_precision = "bf16"  # or "fp16"
```

### Learning Rate Scheduling

```python
config.scheduler.name = "cosine_with_warmup"
config.scheduler.warmup_steps = 1000
config.scheduler.min_lr = 1e-6
```

### Checkpoint Management

```python
config.checkpoint.save_every_n_steps = 1000
config.checkpoint.keep_last_n = 3  # Keep only last 3 checkpoints
config.checkpoint.save_optimizer_state = True
config.checkpoint.save_scheduler_state = True
```

### Evaluation

```python
config.eval_every_n_steps = 500
config.eval_split = "validation"
```

## Examples

### Example 1: Fine-tune SmolVLA on Aloha

```python
from vla_module.training import (
    ActionHeadTrainer,
    create_default_action_head_config,
)

config = create_default_action_head_config(
    repo_id="lerobot/aloha_sim_insertion_human",
    model_type="smolvla",
    model_path="HuggingFaceTB/SmolVLA-Base",
)

# Configure for Aloha (14 dims: 2x 6-DOF arms + 2x grippers)
config.max_action_dim = 32  # Pad to 32
config.max_state_dim = 32

# Training settings
config.data.batch_size = 16
config.optimizer.lr = 1e-4
config.num_epochs = 10
config.mixed_precision = "bf16"
config.gradient_accumulation_steps = 2

# Multi-camera setup
config.data.image_keys = [
    "observation.images.top",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
]

# Train
model = load_smolvla_model(config.model_path)
trainer = ActionHeadTrainer(config, model, device="cuda")
trainer.train()
```

### Example 2: Fine-tune PI0 on Custom Dataset

```python
config = create_default_action_head_config(
    repo_id="your/custom_robot_dataset",
    model_type="pi0",
    model_path="physical-intelligence/pi0-base",
)

# Custom action space (e.g., 7-DOF arm)
config.max_action_dim = 32
config.use_action_projection = True  # Learn projection

# Training settings
config.optimizer.lr = 5e-5
config.num_epochs = 20
config.data.batch_size = 8
config.gradient_accumulation_steps = 4

# Train
model = load_pi0_model(config.model_path)
trainer = ActionHeadTrainer(config, model)
trainer.train()
```

### Example 3: Resume from Checkpoint

```python
config = create_default_action_head_config(...)

# Specify checkpoint to resume from
config.resume_from_checkpoint = Path("checkpoints/step_5000")

trainer = ActionHeadTrainer(config, model)
trainer.train()  # Continues from step 5000
```

## Troubleshooting

### Out of Memory

1. Reduce batch size:
   ```python
   config.data.batch_size = 4
   ```

2. Use gradient accumulation:
   ```python
   config.gradient_accumulation_steps = 8
   ```

3. Enable mixed precision:
   ```python
   config.mixed_precision = "bf16"
   ```

### Slow Training

1. Increase batch size (if GPU memory allows)
2. Use multiple data loading workers:
   ```python
   config.data.num_workers = 8
   ```

3. Enable pin_memory:
   ```python
   config.data.pin_memory = True
   ```

### NaN Loss
sfd

1. Reduce learning rate:
   ```python
   config.optimizer.lr = 1e-5
   ```

2. Enable gradient clipping:
   ```python
   config.optimizer.gradient_clip_norm = 1.0
   ```

3. Check for data issues (normalize actions/states)

## API Reference

See individual module documentation:
- [Dataset Adapter](data/lerobot_adapter.py)
- [Action Space Config](training/action_space_config.py)
- [Freezing Utilities](training/freeze_utils.py)
- [Training Config](training/training_config.py)
- [Checkpoint Utils](training/checkpoint_utils.py)
- [Action Head Trainer](training/action_head_trainer.py)