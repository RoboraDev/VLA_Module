"""Checkpoint Utilities

Functions for saving and loading model checkpoints, including support for
partial loading, optimizer state, and training resumption.
"""

# TODO: Shiven, wandb integration is pending.

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[Optimizer],
    scheduler: Optional[_LRScheduler],
    step: int,
    epoch: int,
    save_dir: Path,
    checkpoint_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    save_optimizer: bool = True,
    save_scheduler: bool = True,
) -> Path:
    """Save a training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save (optional)
        scheduler: LR scheduler to save (optional)
        step: Current training step
        epoch: Current epoch
        save_dir: Directory to save checkpoint
        checkpoint_name: Custom checkpoint name (default: step_{step})
        metadata: Additional metadata to save
        save_optimizer: Whether to save optimizer state
        save_scheduler: Whether to save scheduler state
        
    Returns:
        Path to saved checkpoint
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if checkpoint_name is None:
        checkpoint_name = f"step_{step}"
    
    checkpoint_dir = save_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint dictionary
    checkpoint = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    
    # Add optimizer state if requested
    if save_optimizer and optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    # Add scheduler state if requested
    if save_scheduler and scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    # Add metadata
    if metadata is not None:
        checkpoint["metadata"] = metadata
    
    # Save main checkpoint
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save metadata as JSON for easy inspection
    metadata_dict = {
        "step": step,
        "epoch": epoch,
        "has_optimizer": save_optimizer and optimizer is not None,
        "has_scheduler": save_scheduler and scheduler is not None,
    }
    if metadata is not None:
        metadata_dict.update(metadata)
    
    metadata_path = checkpoint_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata_dict, f, indent=2)
    
    print(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: str = "cpu",
    strict: bool = True,
    load_optimizer: bool = True,
    load_scheduler: bool = True,
) -> Tuple[int, int, Dict[str, Any]]:
    """Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file or directory
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load checkpoint on
        strict: Whether to strictly enforce state dict keys match
        load_optimizer: Whether to load optimizer state
        load_scheduler: Whether to load scheduler state
        
    Returns:
        Tuple of (step, epoch, metadata)
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Handle both file and directory paths
    if checkpoint_path.is_dir():
        checkpoint_file = checkpoint_path / "checkpoint.pt"
    else:
        checkpoint_file = checkpoint_path
    
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    
    print(f"Loading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    else:
        # Assume entire checkpoint is model state
        model.load_state_dict(checkpoint, strict=strict)
    
    # Load optimizer state if available and requested
    if load_optimizer and optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Load scheduler state if available and requested
    if load_scheduler and scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    # Extract step and epoch
    step = checkpoint.get("step", 0)
    epoch = checkpoint.get("epoch", 0)
    metadata = checkpoint.get("metadata", {})
    
    print(f"Loaded checkpoint from step {step}, epoch {epoch}")
    return step, epoch, metadata


def load_pretrained_model(
    model_path: str,
    model: nn.Module,
    device: str = "cpu",
    strict: bool = False,
) -> None:
    """Load pretrained model weights.
    
    Args:
        model_path: Path to pretrained model
        model: Model to load weights into
        device: Device to load on
        strict: Whether to strictly match keys
    """
    print(f"Loading pretrained model from {model_path}")
    
    # Handle HuggingFace model loading
    if "/" in model_path and not Path(model_path).exists():
        # Assume it's a HuggingFace model ID
        from transformers import AutoModel
        try:
            pretrained = AutoModel.from_pretrained(model_path)
            model.load_state_dict(pretrained.state_dict(), strict=strict)
        except Exception as e:
            print(f"Warning: Could not load from HuggingFace: {e}")
            print("Trying direct checkpoint load...")
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=strict)
    else:
        # Local checkpoint
        state_dict = torch.load(model_path, map_location=device)
        
        # Handle nested state dicts
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        
        model.load_state_dict(state_dict, strict=strict)
    
    print("Successfully loaded pretrained weights")


def cleanup_old_checkpoints(
    save_dir: Path,
    keep_last_n: int = 3,
    pattern: str = "step_*",
) -> None:
    """Remove old checkpoints, keeping only the most recent ones.
    
    Args:
        save_dir: Directory containing checkpoints
        keep_last_n: Number of most recent checkpoints to keep
        pattern: Glob pattern for checkpoint directories
    """
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return
    
    # Find all checkpoint directories
    checkpoints = sorted(
        save_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,  # Sort by modification time
    )
    
    # Remove old checkpoints
    for checkpoint_dir in checkpoints[:-keep_last_n]:
        if checkpoint_dir.is_dir():
            print(f"Removing old checkpoint: {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)


def get_latest_checkpoint(save_dir: Path, pattern: str = "step_*") -> Optional[Path]:
    """Get the path to the most recent checkpoint.
    
    Args:
        save_dir: Directory containing checkpoints
        pattern: Glob pattern for checkpoint directories
        
    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return None
    
    checkpoints = sorted(
        save_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
    )
    
    if not checkpoints:
        return None
    
    return checkpoints[-1]


def save_best_checkpoint(
    model: nn.Module,
    optimizer: Optional[Optimizer],
    scheduler: Optional[_LRScheduler],
    step: int,
    epoch: int,
    metric_value: float,
    save_dir: Path,
    metric_name: str = "loss",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save checkpoint as best model based on metric.
    
    Args:
        model: Model to save
        optimizer: Optimizer
        scheduler: Scheduler
        step: Current step
        epoch: Current epoch
        metric_value: Metric value
        save_dir: Save directory
        metric_name: Name of metric
        metadata: Additional metadata
        
    Returns:
        Path to saved checkpoint
    """
    # Add metric to metadata
    if metadata is None:
        metadata = {}
    metadata[f"best_{metric_name}"] = metric_value
    
    return save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        epoch=epoch,
        save_dir=save_dir,
        checkpoint_name="best",
        metadata=metadata,
    )


def load_partial_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    prefix: Optional[str] = None,
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """Load partial state dict with optional prefix matching.
    
    Useful for loading only specific components (e.g., action head).
    
    Args:
        model: Model to load weights into
        state_dict: State dictionary to load
        prefix: Optional prefix to filter keys
        strict: Whether to require all keys to match
        
    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    if prefix is not None:
        # Filter state dict by prefix
        filtered_state_dict = {
            k[len(prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
    else:
        filtered_state_dict = state_dict
    
    # Load state dict
    result = model.load_state_dict(filtered_state_dict, strict=strict)
    
    missing_keys = result.missing_keys if hasattr(result, "missing_keys") else []
    unexpected_keys = result.unexpected_keys if hasattr(result, "unexpected_keys") else []
    
    return missing_keys, unexpected_keys


def export_model_for_inference(
    model: nn.Module,
    save_path: Path,
    include_config: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Export model for inference (without training state).
    
    Args:
        model: Model to export
        save_path: Path to save exported model
        include_config: Whether to save config
        config: Optional config dictionary
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), save_path)
    
    # Save config if requested
    if include_config and config is not None:
        config_path = save_path.parent / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    print(f"Exported model to {save_path}")

