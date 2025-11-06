"""LeRobot Dataset Adapter for VLA Models

This module provides adapters to bridge LeRobot datasets with VLA models (SmolVLA, PI0),
handling observation/action space mapping, camera feeds, and state features with
configurable dimensions.

This part of the dataloader pipeline directly integrates or link with the lerobot dataset from the framework.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)



@dataclass
class DatasetAdapterConfig:
    """Configuration for dataset adapter."""
    
    # Dataset specification
    repo_id: str
    """HuggingFace dataset repository ID"""
    
    root: Optional[Path] = None
    """Local root directory for dataset cache"""
    
    # Action/Observation space configuration
    max_action_dim: int = 32
    """Maximum action dimension (will pad shorter vectors)"""
    
    max_state_dim: int = 32
    """Maximum state/observation dimension (will pad shorter vectors)"""
    
    # Image configuration
    image_keys: List[str] = field(default_factory=lambda: ["observation.images.top"])
    """List of image observation keys to use"""
    
    resize_images: Optional[Tuple[int, int]] = (512, 512)
    """Target image size (width, height). None to disable resizing"""
    
    # Language configuration
    use_language: bool = True
    """Whether to include language instructions"""
    
    tokenizer_max_length: int = 48
    """Maximum token sequence length for language"""
    
    # Temporal configuration
    n_obs_steps: int = 1
    """Number of observation steps"""
    
    chunk_size: int = 50
    """Action chunk size / prediction horizon"""
    
    # Data loading
    split: str = "train"
    """Dataset split to use"""
    
    streaming: bool = False
    """Whether to use streaming dataset"""


def pad_vector(vector: torch.Tensor, new_dim: int) -> torch.Tensor:
    """Pad the last dimension of a vector to new_dim with zeros.
    
    Args:
        vector: Tensor of shape (..., features_dim)
        new_dim: Target dimension
        
    Returns:
        Padded tensor of shape (..., new_dim)
    """
    if vector.shape[-1] >= new_dim:
        return vector[..., :new_dim]  # Truncate if larger
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def resize_with_pad(
    img: torch.Tensor,
    width: int,
    height: int,
    pad_value: float = -1.0
) -> torch.Tensor:
    """Resize image with padding to preserve aspect ratio.
    
    Args:
        img: Image tensor of shape (B, C, H, W) or (C, H, W)
        width: Target width
        height: Target height
        pad_value: Value to use for padding
        
    Returns:
        Resized and padded image
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    if img.ndim != 4:
        raise ValueError(f"Expected (B,C,H,W) or (C,H,W), got {img.shape}")
    
    cur_height, cur_width = img.shape[2:]
    
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    
    resized_img = F.interpolate(
        img,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False
    )
    
    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    
    # Pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    
    if squeeze_output:
        padded_img = padded_img.squeeze(0)
    
    return padded_img


class LeRobotDatasetAdapter(Dataset):
    """Adapter for LeRobot datasets to work with VLA models.
    
    This adapter handles:
    - Action/observation space mapping and padding
    - Multi-camera image preprocessing
    - Language instruction encoding
    - State feature extraction and padding
    - Temporal windowing for observations and actions
    """
    
    def __init__(self, config: DatasetAdapterConfig):
        """Initialize the dataset adapter.
        
        Args:
            config: Configuration for the adapter
        """
        self.config = config
        
        # Create the LeRobot dataset
        self.dataset = self._load_dataset()
        
        # Extract feature dimensions from dataset metadata
        self._extract_feature_dims()
        
        # Validate configuration
        self._validate_config()
    
    def _load_dataset(self) -> LeRobotDataset:
        """Load the LeRobot dataset."""
        # TODO: SHIVEN Check the import statement once
        from lerobot.configs.dataset import DatasetConfig
        
        ds_config = DatasetConfig(
            repo_id=self.config.repo_id,
            root=str(self.config.root) if self.config.root else None,
            split=self.config.split,
            streaming=self.config.streaming,
        )
        
        return make_dataset(ds_config)
    
    def _extract_feature_dims(self):
        """Extract original action and state dimensions from dataset."""
        # Get feature info from dataset metadata
        features = self.dataset.meta.info.get("features", {})
        
        # Extract action dimension
        if "action" in features:
            action_shape = features["action"].get("shape", [])
            self.original_action_dim = action_shape[0] if action_shape else 0
        else:
            self.original_action_dim = 0
        
        # Extract state dimension
        state_keys = [k for k in features.keys() if "state" in k.lower()]
        if state_keys:
            state_shape = features[state_keys[0]].get("shape", [])
            self.original_state_dim = state_shape[0] if state_shape else 0
        else:
            self.original_state_dim = 0
        
        # Store available image keys
        self.available_image_keys = [
            k for k in features.keys()
            if "image" in k.lower() or k.startswith(OBS_IMAGES)
        ]
    
    def _validate_config(self):
        """Validate adapter configuration."""
        # Check if requested image keys are available
        for img_key in self.config.image_keys:
            # Handle both full keys and partial matches
            found = any(
                img_key in avail_key or avail_key in img_key
                for avail_key in self.available_image_keys
            )
            if not found:
                print(f"Warning: Requested image key '{img_key}' not found in dataset. "
                      f"Available: {self.available_image_keys}")
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
            - images: List of image tensors (one per camera)
            - image_masks: Boolean mask indicating valid images
            - state: Padded state vector
            - action: Padded action vector
            - language_tokens: Tokenized language instruction (if available)
            - language_attention_mask: Attention mask for language tokens
        """
        # Get raw sample from dataset
        raw_sample = self.dataset[idx]
        
        # Process sample
        processed = {}
        
        # Process images
        images, image_masks = self._process_images(raw_sample)
        processed["images"] = images
        processed["image_masks"] = image_masks
        
        # Process state
        if OBS_STATE in raw_sample:
            state = raw_sample[OBS_STATE]
            if isinstance(state, torch.Tensor):
                # Pad to max dimension
                processed["state"] = pad_vector(state, self.config.max_state_dim)
            else:
                # Convert numpy to tensor if needed
                state = torch.from_numpy(state).float()
                processed["state"] = pad_vector(state, self.config.max_state_dim)
        else:
            # Create zero state if not available
            processed["state"] = torch.zeros(self.config.max_state_dim)
        
        # Process actions
        if ACTION in raw_sample:
            action = raw_sample[ACTION]
            if isinstance(action, torch.Tensor):
                processed["action"] = pad_vector(action, self.config.max_action_dim)
            else:
                action = torch.from_numpy(action).float()
                processed["action"] = pad_vector(action, self.config.max_action_dim)
        else:
            processed["action"] = torch.zeros(self.config.max_action_dim)
        
        # Process language (if available and enabled)
        if self.config.use_language:
            if OBS_LANGUAGE_TOKENS in raw_sample:
                processed["language_tokens"] = raw_sample[OBS_LANGUAGE_TOKENS]
                processed["language_attention_mask"] = raw_sample.get(
                    OBS_LANGUAGE_ATTENTION_MASK,
                    torch.ones_like(raw_sample[OBS_LANGUAGE_TOKENS])
                )
            else:
                # Create dummy language tokens if not available
                processed["language_tokens"] = torch.zeros(
                    self.config.tokenizer_max_length, dtype=torch.long
                )
                processed["language_attention_mask"] = torch.zeros(
                    self.config.tokenizer_max_length, dtype=torch.bool
                )
        
        # Add metadata
        processed["original_action_dim"] = self.original_action_dim
        processed["original_state_dim"] = self.original_state_dim
        
        return processed
    
    def _process_images(
        self,
        sample: Dict[str, Any]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Process images from sample.
        
        Args:
            sample: Raw sample from dataset
            
        Returns:
            Tuple of (list of image tensors, boolean mask of valid images)
        """
        images = []
        masks = []
        
        for img_key in self.config.image_keys:
            # Try to find the image in the sample
            img = None
            
            # Try exact match first
            if img_key in sample:
                img = sample[img_key]
            else:
                # Try partial match
                for key in sample.keys():
                    if img_key in key or key in img_key:
                        if "image" in key.lower():
                            img = sample[key]
                            break
            
            if img is not None:
                # Convert to tensor if needed
                if not isinstance(img, torch.Tensor):
                    img = torch.from_numpy(img).float()
                
                # Ensure correct shape (C, H, W) or (B, C, H, W)
                if img.ndim == 3 and img.shape[-1] in [1, 3, 4]:
                    # (H, W, C) -> (C, H, W)
                    img = img.permute(2, 0, 1)
                elif img.ndim == 4 and img.shape[-1] in [1, 3, 4]:
                    # (B, H, W, C) -> (B, C, H, W)
                    img = img.permute(0, 3, 1, 2)
                
                # Take last frame if temporal dimension exists
                if img.ndim == 4:
                    img = img[-1]  # Take last timestep
                
                # Normalize to [-1, 1] if in [0, 255]
                if img.max() > 1.0:
                    img = img / 127.5 - 1.0
                elif img.max() <= 1.0 and img.min() >= 0.0:
                    img = img * 2.0 - 1.0
                
                # Resize if configured
                if self.config.resize_images is not None:
                    img = resize_with_pad(
                        img,
                        width=self.config.resize_images[0],
                        height=self.config.resize_images[1],
                        pad_value=-1.0
                    )
                
                images.append(img)
                masks.append(True)
            else:
                # Create dummy image if not found
                if self.config.resize_images is not None:
                    dummy_img = torch.ones(
                        3,
                        self.config.resize_images[1],
                        self.config.resize_images[0]
                    ) * -1.0
                else:
                    dummy_img = torch.ones(3, 224, 224) * -1.0
                
                images.append(dummy_img)
                masks.append(False)
        
        # Convert masks to tensor
        image_masks = torch.tensor(masks, dtype=torch.bool)
        
        return images, image_masks
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics for normalization.
        
        Returns:
            Dictionary containing mean, std, min, max for actions and states
        """
        if hasattr(self.dataset, "meta") and hasattr(self.dataset.meta, "stats"):
            return self.dataset.meta.stats
        return {}
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about dataset features.
        
        Returns:
            Dictionary with feature specifications
        """
        return {
            "original_action_dim": self.original_action_dim,
            "original_state_dim": self.original_state_dim,
            "max_action_dim": self.config.max_action_dim,
            "max_state_dim": self.config.max_state_dim,
            "image_keys": self.config.image_keys,
            "available_image_keys": self.available_image_keys,
        }


def create_lerobot_dataloader(
    config: DatasetAdapterConfig,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader for LeRobot dataset.
    
    Args:
        config: Dataset adapter configuration
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the dataset
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    dataset = LeRobotDatasetAdapter(config)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop last incomplete batch
    )

