"""Model Loader for VLA Training

Provides utilities to load VLA models (SmolVLA, PI0) for training.
Wraps LeRobot policies to work with the training infrastructure.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn


class VLAModelWrapper(nn.Module):
    """Wrapper to adapt LeRobot policies to training interface.
    
    This wrapper:
    - Provides a consistent interface for training
    - Handles forward pass for both training and inference
    - Delegates to the underlying policy for actual computation
    - Ensures compatibility with freezing utilities
    """
    
    def __init__(self, policy: nn.Module):
        """Initialize wrapper.
        
        Args:
            policy: LeRobot policy instance (SmolVLAPolicy or PI0Policy)
        """
        super().__init__()
        self.policy = policy
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Union[torch.Tensor, Dict]:
        """Forward pass.
        
        Args:
            batch: Dictionary with keys:
                - images: List of image tensors
                - image_masks: Boolean masks for valid images
                - state: State tensor
                - action: Action tensor (for training)
                - language_tokens: Language token tensor
                - language_attention_mask: Attention mask for language
        
        Returns:
            Loss tensor (training) or action predictions (inference)
        """
        # Check if we're in training mode (have ground truth actions)
        if 'action' in batch:
            # Training: compute loss
            result = self.policy.forward(batch)
            
            # Handle different return formats
            if isinstance(result, tuple):
                loss = result[0]
            elif isinstance(result, dict):
                loss = result.get('loss', result.get('total_loss'))
            else:
                loss = result
            
            return loss
        else:
            # Inference: predict actions
            actions = self.policy.select_action(batch)
            return actions
    
    def to(self, device: Union[str, torch.device]) -> 'VLAModelWrapper':
        """Move model to device.
        
        Args:
            device: Target device
            
        Returns:
            Self
        """
        self.policy = self.policy.to(device)
        return super().to(device)
    
    def state_dict(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Get state dict from wrapped policy.
        
        Returns:
            State dictionary
        """
        return self.policy.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """Load state dict into wrapped policy.
        
        Args:
            state_dict: State dictionary to load
            strict: Whether to strictly enforce key matching
        """
        return self.policy.load_state_dict(state_dict, strict=strict)
    
    def parameters(self, *args, **kwargs):
        """Get parameters from wrapped policy."""
        return self.policy.parameters(*args, **kwargs)
    
    def named_parameters(self, *args, **kwargs):
        """Get named parameters from wrapped policy."""
        return self.policy.named_parameters(*args, **kwargs)
    
    def train(self, mode: bool = True) -> 'VLAModelWrapper':
        """Set training mode."""
        self.policy.train(mode)
        return super().train(mode)
    
    def eval(self) -> 'VLAModelWrapper':
        """Set evaluation mode."""
        self.policy.eval()
        return super().eval()


def load_vla_model(
    model_type: str,
    model_path: Optional[str] = None,
    config: Optional[Dict] = None,
) -> nn.Module:
    """Load a VLA model (SmolVLA or PI0).
    
    Args:
        model_type: Type of model ('smolvla' or 'pi0')
        model_path: Path to pretrained model or HuggingFace model ID
        config: Optional config dict to override defaults
    
    Returns:
        Loaded VLA model wrapped for training
    
    Example:
        >>> # Load pretrained SmolVLA
        >>> model = load_vla_model('smolvla', 'HuggingFaceTB/SmolVLA-Base')
        
        >>> # Load PI0
        >>> model = load_vla_model('pi0', 'physical-intelligence/pi0-base')
        
        >>> # Create from scratch with custom config
        >>> model = load_vla_model('smolvla', config={'max_action_dim': 64})
    """
    if model_type.lower() == 'smolvla':
        return load_smolvla_model(model_path, config)
    elif model_type.lower() == 'pi0':
        return load_pi0_model(model_path, config)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'smolvla' or 'pi0'")


def load_smolvla_model(
    model_path: Optional[str] = None,
    config: Optional[Dict] = None,
) -> nn.Module:
    """Load SmolVLA model.
    
    Args:
        model_path: Path to pretrained model or HuggingFace model ID
        config: Optional config overrides
    
    Returns:
        SmolVLA model wrapped for training
    
    Example:
        >>> model = load_smolvla_model('HuggingFaceTB/SmolVLA-Base')
    """
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    except ImportError as e:
        raise ImportError(
            "Could not import SmolVLA from lerobot. "
            "Make sure lerobot is installed: pip install lerobot"
        ) from e
    
    # Create config
    if config is not None:
        policy_config = SmolVLAConfig(**config)
    else:
        policy_config = SmolVLAConfig()
    
    # Load or create policy
    if model_path:
        print(f"Loading SmolVLA from {model_path}")
        try:
            # Try loading from HuggingFace or local path
            policy = SmolVLAPolicy.from_pretrained(model_path)
            print("✓ Successfully loaded pretrained SmolVLA")
        except Exception as e:
            print(f"Warning: Could not load from {model_path}: {e}")
            print("Creating new SmolVLA model with config...")
            policy = SmolVLAPolicy(policy_config)
    else:
        print("Creating new SmolVLA model")
        policy = SmolVLAPolicy(policy_config)
    
    # Wrap the policy
    model = VLAModelWrapper(policy)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"SmolVLA model created: {num_params:,} parameters")
    
    return model


def load_pi0_model(
    model_path: Optional[str] = None,
    config: Optional[Dict] = None,
) -> nn.Module:
    """Load PI0 model.
    
    Args:
        model_path: Path to pretrained model or HuggingFace model ID
        config: Optional config overrides
    
    Returns:
        PI0 model wrapped for training
    
    Example:
        >>> model = load_pi0_model('physical-intelligence/pi0-base')
    """
    try:
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        from lerobot.policies.pi0.configuration_pi0 import PI0Config
    except ImportError as e:
        raise ImportError(
            "Could not import PI0 from lerobot. "
            "Make sure lerobot is installed: pip install lerobot"
        ) from e
    
    # Create config
    if config is not None:
        policy_config = PI0Config(**config)
    else:
        policy_config = PI0Config()
    
    # Load or create policy
    if model_path:
        print(f"Loading PI0 from {model_path}")
        try:
            policy = PI0Policy.from_pretrained(model_path)
            print("✓ Successfully loaded pretrained PI0")
        except Exception as e:
            print(f"Warning: Could not load from {model_path}: {e}")
            print("Creating new PI0 model with config...")
            policy = PI0Policy(policy_config)
    else:
        print("Creating new PI0 model")
        policy = PI0Policy(policy_config)
    
    # Wrap the policy
    model = VLAModelWrapper(policy)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"PI0 model created: {num_params:,} parameters")
    
    return model


def create_model_for_training(
    model_type: str,
    model_path: Optional[str] = None,
    dataset_stats: Optional[Dict] = None,
    max_action_dim: int = 32,
    max_state_dim: int = 32,
) -> nn.Module:
    """Create a model configured for training.
    
    This is a convenience function that sets up a model with the right
    configuration for training on a specific dataset.
    
    Args:
        model_type: 'smolvla' or 'pi0'
        model_path: Path to pretrained weights
        dataset_stats: Dataset statistics for normalization
        max_action_dim: Maximum action dimension
        max_state_dim: Maximum state dimension
    
    Returns:
        Model ready for training
    
    Example:
        >>> model = create_model_for_training(
        ...     model_type='smolvla',
        ...     model_path='HuggingFaceTB/SmolVLA-Base',
        ...     max_action_dim=32,
        ...     max_state_dim=32,
        ... )
    """
    config = {
        'max_action_dim': max_action_dim,
        'max_state_dim': max_state_dim,
    }
    
    model = load_vla_model(model_type, model_path, config)
    
    # If we have dataset stats, we could set them on the model
    # This depends on your model implementation
    if dataset_stats and hasattr(model.policy, 'set_normalization_stats'):
        model.policy.set_normalization_stats(dataset_stats)
    
    return model


# Export main functions
__all__ = [
    'VLAModelWrapper',
    'load_vla_model',
    'load_smolvla_model',
    'load_pi0_model',
    'create_model_for_training',
]

