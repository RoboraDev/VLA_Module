from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


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

