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

def pad_tensor(tensor: Tensor, max_len: int, pad_value: float = 0) -> Tensor:
    """Efficiently pad tensor along sequence dimension.
    
    Args:
        tensor: Input tensor of shape (B, L, ...) or (B, L)
        max_len: Target sequence length
        pad_value: Value to use for padding
    
    Returns:
        Padded tensor of shape (B, max_len, ...) or (B, max_len)
    """
    b, d = tensor.shape[:2]
    
    # Create padded tensor and copy existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor
    
    return padded_tensor


def normalize(x: Tensor, min_val: float, max_val: float) -> Tensor:
    """Normalize values to [0, 1] range.
    
    Args:
        x: Input tensor
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Normalized tensor
    """
    return (x - min_val) / (max_val - min_val)


def unnormalize(x: Tensor, min_val: float, max_val: float) -> Tensor:
    """Unnormalize values from [0, 1] range.
    
    Args:
        x: Input tensor (assumed to be in [0, 1])
        min_val: Minimum value of target range
        max_val: Maximum value of target range
    
    Returns:
        Unnormalized tensor
    """
    return x * (max_val - min_val) + min_val


def safe_arcsin(value: Tensor) -> Tensor:
    """Safe arcsin that clamps input to [-1, 1].
    
    Args:
        value: Input tensor
    
    Returns:
        Arcsin of clamped value
    """
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


## Aloha Gripper conversion, might be useful later
def aloha_gripper_to_angular(value: Tensor) -> Tensor:
    """Convert Aloha gripper position to angular space.
    
    Aloha transforms gripper positions into a linear space. This reverses the
    transformation to be consistent with SmolVLA pretraining in angular space.
    
    Args:
        value: Gripper position in Aloha linear space
    
    Returns:
        Gripper position in angular space [0, 1]
    """
    # Unnormalize from Aloha's linear range
    # Values from Aloha code: PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)
    
    def linear_to_radian(linear_position: Tensor, arm_length: float, horn_radius: float) -> Tensor:
        """Convert linear position to radians (inverse of Interbotix transformation)."""
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)
    
    # Constants from Interbotix code
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    
    # Normalize to [0, 1]
    # Values measured on actual Trossen robot
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value: Tensor) -> Tensor:
    """Convert gripper position from SmolVLA angular space to Aloha space.
    
    Args:
        value: Gripper position in angular space [0, 1]
    
    Returns:
        Gripper position in Aloha space
    """
    # Unnormalize from [0, 1] (values measured on Trossen robot)
    value = unnormalize(value, min_val=0.4, max_val=1.5)
    
    # Normalize to Aloha's gripper joint range
    # Values from Aloha code: PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value: Tensor) -> Tensor:
    """Directly invert aloha_gripper_from_angular function.
    
    Args:
        value: Gripper position in Aloha space
    
    Returns:
        Gripper position in angular space [0, 1]
    """
    # Unnormalize from Aloha's gripper joint range
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    
    # Normalize back to [0, 1]
    return normalize(value, min_val=0.4, max_val=1.5)


def apply_rope(x: Tensor, positions: Tensor, max_wavelength: float = 10_000) -> Tensor:
    """Apply Rotary Position Embedding (RoPE) to input tensor.
    
    Args:
        x: Input tensor of shape (B, L, H, D) where:
           B = batch size, L = sequence length, H = num heads, D = head dim
        positions: Position indices of shape (B, L)
        max_wavelength: Maximum wavelength for frequency computation
    
    Returns:
        Tensor with RoPE applied, same shape as input
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    
    # Convert to float32 for computation
    x = x.to(torch.float32)
    
    # Compute frequency exponents
    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength ** freq_exponents
    
    # Compute radians
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)
    radians = radians[..., None, :]
    
    # Compute sin and cos
    sin = torch.sin(radians)
    cos = torch.cos(radians)
    
    # Split input and apply rotation
    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin
    
    return res.to(dtype)


def get_intermediate_size(hidden_dim: int, ffn_dim_multiplier: float = 4, multiple_of: int = 256) -> int:
    """Compute intermediate size for feed-forward network.
    
    Args:
        hidden_dim: Hidden dimension
        ffn_dim_multiplier: Multiplier for FFN dimension
        multiple_of: Round to multiple of this value
    
    Returns:
        Intermediate size
    """
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim

