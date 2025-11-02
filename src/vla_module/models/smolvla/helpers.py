"""SmolVLA Helper Functions

Utility functions for SmolVLA policy implementation, adapted from LeRobot.
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor


def create_sinusoidal_pos_embedding(
    time: Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Compute sine-cosine positional embedding vectors for scalar positions.
    
    Args:
        time: Time tensor of shape (batch_size,)
        dimension: Embedding dimension (must be divisible by 2)
        min_period: Minimum period for frequency range
        max_period: Maximum period for frequency range
        device: Device to create tensors on
    
    Returns:
        Positional embeddings of shape (batch_size, dimension)
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    
    if time.ndim != 1:
        raise ValueError(f"time tensor must be 1D, got shape {time.shape}")
    
    # Use float64 for precision if available
    dtype = torch.float64 if device != "mps" else torch.float32
    
    # Create frequency range
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    
    # Compute sine and cosine components
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    
    # Concatenate sin and cos
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    
    return pos_emb
