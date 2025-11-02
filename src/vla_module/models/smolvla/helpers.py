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

def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """Create 2D attention masks from padding and attention masks.
    
    Tokens can attend to valid input tokens which have a cumulative mask_ar
    smaller or equal to theirs. This enables different attention patterns:
    
    Examples:
      [[1 1 1 1 1 1]]: pure causal attention
      [[0 0 0 1 1 1]]: prefix-lm attention (first 3 tokens attend to each other,
                       last 3 tokens have causal attention)
      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks
    
    Args:
        pad_masks: Boolean tensor of shape (B, N). True if part of input, False if padding.
        att_masks: Integer tensor of shape (B, N). 1 where previous tokens cannot attend to it,
                   0 where it shares attention mask with previous token.
    
    Returns:
        2D attention mask of shape (B, N, N)
    """
    if att_masks.ndim != 2:
        raise ValueError(f"att_masks must be 2D, got shape {att_masks.shape}")
    if pad_masks.ndim != 2:
        raise ValueError(f"pad_masks must be 2D, got shape {pad_masks.shape}")
    
    # Compute cumulative sum of attention masks
    cumsum = torch.cumsum(att_masks, dim=1)
    
    # Create 2D attention mask: position i can attend to position j if cumsum[j] <= cumsum[i]
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    
    # Apply padding mask
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    
    return att_2d_masks


def resize_with_pad(img: Tensor, width: int, height: int, pad_value: float = -1) -> Tensor:
    """Resize image with padding to preserve aspect ratio.
    
    Args:
        img: Image tensor of shape (B, C, H, W)
        width: Target width
        height: Target height
        pad_value: Value to use for padding
    
    Returns:
        Resized and padded image of shape (B, C, height, width)
    """
    if img.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W), got shape {img.shape}")
    
    cur_height, cur_width = img.shape[2:]
    
    # Compute resize ratio to fit within target dimensions
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    
    # Resize image
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )
    
    # Compute padding (pad on left and top)
    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    
    # Apply padding
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    
    return padded_img


def pad_vector(vector: Tensor, new_dim: int) -> Tensor:
    """Pad vector to new dimension.
    
    Can handle:
    - (batch_size, features_dimension)
    - (batch_size, sequence_length, features_dimension)
    
    Args:
        vector: Input tensor
        new_dim: Target dimension for last axis
    
    Returns:
        Padded tensor with last dimension = new_dim
    """
    if vector.shape[-1] == new_dim:
        return vector
    
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    
    # Create new tensor filled with zeros
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    
    return new_vector
