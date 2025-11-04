"""Main Training Script for VLA Models

Entry point for training VLA models (SmolVLA, PI0) with LeRobot datasets.
Supports both action-head-only and full model fine-tuning.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

from vla_module.training.action_head_trainer import ActionHeadTrainer
from vla_module.training.training_config import (
    ActionHeadTrainingConfig,
    create_default_action_head_config,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train VLA models on LeRobot datasets"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default="smolvla",
        choices=["smolvla", "pi0"],
        help="Type of VLA model",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint",
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset repository ID (e.g., lerobot/aloha_sim_insertion_human)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Local dataset cache directory",
    )
    
    # Training mode
    parser.add_argument(
        "--training-mode",
        type=str,
        default="action_head_only",
        choices=["action_head_only", "full"],
        help="Training mode: action head only or full model",
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per device",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides num_epochs)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    
    # Action/Observation space
    parser.add_argument(
        "--max-action-dim",
        type=int,
        default=32,
        help="Maximum action dimension",
    )
    parser.add_argument(
        "--max-state-dim",
        type=int,
        default=32,
        help="Maximum state dimension",
    )
    parser.add_argument(
        "--use-action-projection",
        action="store_true",
        help="Use learned projection for actions",
    )
    parser.add_argument(
        "--use-state-projection",
        action="store_true",
        help="Use learned projection for states",
    )
    
    # Checkpointing and logging
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vla_training"),
        help="Output directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--save-every-n-steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--eval-every-n-steps",
        type=int,
        default=500,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=10,
        help="Log metrics every N steps",
    )
    
    # Resumption
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume training from checkpoint",
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    
    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> ActionHeadTrainingConfig:
    """Create training configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Training configuration
    """
    if args.training_mode == "action_head_only":
        config = create_default_action_head_config(
            repo_id=args.dataset,
            model_type=args.model_type,
            model_path=args.model_path,
        )
    else:
        # TODO: Implement full training config
        raise NotImplementedError("Full training mode not yet implemented")
    
    # Update config with arguments
    config.data.repo_id = args.dataset
    config.data.root = args.dataset_root
    config.data.batch_size = args.batch_size
    config.data.num_workers = args.num_workers
    
    config.num_epochs = args.num_epochs
    config.max_steps = args.max_steps
    config.optimizer.lr = args.lr
    config.optimizer.weight_decay = args.weight_decay
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.mixed_precision = args.mixed_precision
    
    config.max_action_dim = args.max_action_dim
    config.max_state_dim = args.max_state_dim
    config.use_action_projection = args.use_action_projection
    config.use_state_projection = args.use_state_projection
    
    config.output_dir = args.output_dir
    config.checkpoint.save_dir = args.checkpoint_dir
    config.checkpoint.save_every_n_steps = args.save_every_n_steps
    config.eval_every_n_steps = args.eval_every_n_steps
    config.logging.log_every_n_steps = args.log_every_n_steps
    
    config.resume_from_checkpoint = args.resume_from
    config.seed = args.seed
    
    return config


def load_model(model_type: str, model_path: Optional[str] = None):
    """Load VLA model.
    
    Args:
        model_type: Type of model ('smolvla' or 'pi0')
        model_path: Optional path to pretrained weights
        
    Returns:
        Loaded model
    """
    from vla_module.models.model_loader import load_vla_model
    
    try:
        model = load_vla_model(
            model_type=model_type,
            model_path=model_path,
        )
        return model
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nTo use this training script, make sure lerobot is installed:")
        print("  pip install lerobot")
        raise
    except Exception as e:
        print(f"\nError loading model: {e}")
        raise


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    print("="*80)
    print("VLA Model Training")
    print("="*80)
    print(f"Model type: {args.model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Training mode: {args.training_mode}")
    print(f"Device: {args.device}")
    print("="*80)
    print()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create configuration
    print("Creating training configuration...")
    config = create_config_from_args(args)
    
    # Load model
    print(f"Loading {args.model_type} model...")
    try:
        model = load_model(args.model_type, args.model_path)
    except Exception as e:
        print(f"\nFailed to load model: {e}")
        print("\nPlease ensure:")
        print("1. lerobot is installed: pip install lerobot")
        print("2. Model path is correct (if specified)")
        print("3. You have internet connection (for HuggingFace downloads)")
        sys.exit(1)
    
    # Create trainer
    print("Creating trainer...")
    if args.training_mode == "action_head_only":
        trainer = ActionHeadTrainer(
            config=config,
            model=model,
            device=args.device,
        )
    else:
        # TODO: Implement full trainer
        print("Error: Full training mode not yet implemented")
        sys.exit(1)
    
    # Start training
    print("\nStarting training...\n")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer._save_checkpoint()
        print("Done!")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print(f"Checkpoints saved to: {config.checkpoint.save_dir}")
    print(f"Outputs saved to: {config.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
