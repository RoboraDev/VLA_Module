"""Action and Observation Space Configuration

This module provides configurable action and observation space systems that allow
modification from default dimensions (32 for SmolVLA/PI0 EAch). It includes state projection
layers that adapt to different robot configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActionSpaceConfig:
    """Configuration for action space."""
    
    # Original robot action dimensions
    original_dim: int = 7
    """Original action dimension from the robot (e.g., 7 for 6-DOF + gripper)"""
    
    # Padded/projected dimensions
    max_dim: int = 32
    """Maximum action dimension used by the model (will pad or project to this)"""
    
    # Action components
    joint_dims: Optional[List[int]] = None
    """Indices of joint position actions (e.g., [0, 1, 2, 3, 4, 5] for 6-DOF arm)"""
    
    gripper_dims: Optional[List[int]] = None
    """Indices of gripper actions (e.g., [6] for single gripper value)"""
    
    # Projection settings
    use_projection: bool = False
    """Whether to use learned projection instead of simple padding"""
    
    projection_hidden_dim: int = 128
    """Hidden dimension for projection network if used"""
    
    # Normalization
    normalize: bool = True
    """Whether to normalize actions"""
    
    action_mean: Optional[torch.Tensor] = None
    """Mean for action normalization (computed from dataset stats)"""
    
    action_std: Optional[torch.Tensor] = None
    """Std for action normalization (computed from dataset stats)"""


@dataclass
class ObservationSpaceConfig:
    """Configuration for observation space (state)."""
    
    # Original robot state dimensions
    original_dim: int = 7
    """Original state dimension from the robot"""
    
    # Padded/projected dimensions
    max_dim: int = 32
    """Maximum state dimension used by the model"""
    
    # State components
    joint_state_dims: Optional[List[int]] = None
    """Indices of joint state observations"""
    
    gripper_state_dims: Optional[List[int]] = None
    """Indices of gripper state observations"""
    
    # Projection settings
    use_projection: bool = False
    """Whether to use learned projection instead of simple padding"""
    
    projection_hidden_dim: int = 128
    """Hidden dimension for projection network if used"""
    
    # Normalization
    normalize: bool = True
    """Whether to normalize states"""
    
    state_mean: Optional[torch.Tensor] = None
    """Mean for state normalization"""
    
    state_std: Optional[torch.Tensor] = None
    """Std for state normalization"""


class ActionProjector(nn.Module):
    """Learnable projection layer for actions.
    
    Projects between original robot action space and model action space.
    Useful when the model expects a different dimension than the robot provides.
    """
    
    def __init__(
        self,
        original_dim: int,
        target_dim: int,
        hidden_dim: int = 128,
        use_residual: bool = True,
    ):
        """Initialize action projector.
        
        Args:
            original_dim: Original action dimension
            target_dim: Target action dimension
            hidden_dim: Hidden layer dimension
            use_residual: Whether to use residual connection (requires same dims)
        """
        super().__init__()
        self.original_dim = original_dim
        self.target_dim = target_dim
        self.use_residual = use_residual and (original_dim == target_dim)
        
        # Encoder: original_dim -> target_dim
        self.encoder = nn.Sequential(
            nn.Linear(original_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, target_dim),
        )
        
        # Decoder: target_dim -> original_dim
        self.decoder = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, original_dim),
        )
    
    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Project actions from original space to model space.
        
        Args:
            actions: Actions in original space (..., original_dim)
            
        Returns:
            Projected actions (..., target_dim)
        """
        projected = self.encoder(actions)
        
        # Add residual if dimensions match
        if self.use_residual:
            # Pad or truncate to match
            if self.original_dim < self.target_dim:
                actions_padded = F.pad(actions, (0, self.target_dim - self.original_dim))
                projected = projected + actions_padded
            elif self.original_dim == self.target_dim:
                projected = projected + actions
        
        return projected
    
    def decode(self, actions: torch.Tensor) -> torch.Tensor:
        """Project actions from model space to original space.
        
        Args:
            actions: Actions in model space (..., target_dim)
            
        Returns:
            Projected actions (..., original_dim)
        """
        projected = self.decoder(actions)
        
        # Add residual if dimensions match
        if self.use_residual:
            # Truncate or use subset
            if self.target_dim >= self.original_dim:
                actions_subset = actions[..., :self.original_dim]
                projected = projected + actions_subset
        
        return projected


class StateProjector(nn.Module):
    """Learnable projection layer for states/observations.
    
    Projects between original robot state space and model state space.
    """
    
    def __init__(
        self,
        original_dim: int,
        target_dim: int,
        hidden_dim: int = 128,
        use_residual: bool = True,
    ):
        """Initialize state projector.
        
        Args:
            original_dim: Original state dimension
            target_dim: Target state dimension
            hidden_dim: Hidden layer dimension
            use_residual: Whether to use residual connection
        """
        super().__init__()
        self.original_dim = original_dim
        self.target_dim = target_dim
        self.use_residual = use_residual and (original_dim == target_dim)
        
        # Projection network
        self.projection = nn.Sequential(
            nn.Linear(original_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, target_dim),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Project state from original space to model space.
        
        Args:
            state: State in original space (..., original_dim)
            
        Returns:
            Projected state (..., target_dim)
        """
        projected = self.projection(state)
        
        # Add residual if dimensions match
        if self.use_residual:
            if self.original_dim < self.target_dim:
                state_padded = F.pad(state, (0, self.target_dim - self.original_dim))
                projected = projected + state_padded
            elif self.original_dim == self.target_dim:
                projected = projected + state
        
        return projected


class ActionSpaceAdapter(nn.Module):
    """Adapter for action space transformations.
    
    Handles:
    - Padding/projection to model dimension
    - Normalization/denormalization
    - Component extraction (joints, gripper)
    """
    
    def __init__(self, config: ActionSpaceConfig):
        """Initialize action space adapter.
        
        Args:
            config: Action space configuration
        """
        super().__init__()
        self.config = config
        
        # Create projector if needed
        if config.use_projection:
            self.projector = ActionProjector(
                original_dim=config.original_dim,
                target_dim=config.max_dim,
                hidden_dim=config.projection_hidden_dim,
            )
        else:
            self.projector = None
        
        # Register normalization parameters
        if config.normalize and config.action_mean is not None:
            self.register_buffer("action_mean", config.action_mean)
            self.register_buffer("action_std", config.action_std)
    
    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode actions from robot space to model space.
        
        Args:
            actions: Actions in robot space (..., original_dim)
            
        Returns:
            Actions in model space (..., max_dim)
        """
        # Normalize if configured
        if self.config.normalize and hasattr(self, "action_mean"):
            actions = (actions - self.action_mean) / (self.action_std + 1e-8)
        
        # Project or pad
        if self.config.use_projection and self.projector is not None:
            actions = self.projector.encode(actions)
        else:
            # Simple padding
            if actions.shape[-1] < self.config.max_dim:
                actions = F.pad(actions, (0, self.config.max_dim - actions.shape[-1]))
            elif actions.shape[-1] > self.config.max_dim:
                actions = actions[..., :self.config.max_dim]
        
        return actions
    
    def decode(self, actions: torch.Tensor) -> torch.Tensor:
        """Decode actions from model space to robot space.
        
        Args:
            actions: Actions in model space (..., max_dim)
            
        Returns:
            Actions in robot space (..., original_dim)
        """
        # Project or truncate
        if self.config.use_projection and self.projector is not None:
            actions = self.projector.decode(actions)
        else:
            # Simple truncation
            actions = actions[..., :self.config.original_dim]
        
        # Denormalize if configured
        if self.config.normalize and hasattr(self, "action_mean"):
            actions = actions * (self.action_std + 1e-8) + self.action_mean
        
        return actions
    
    def extract_components(
        self,
        actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract action components (joints, gripper, etc.).
        
        Args:
            actions: Actions in robot space (..., original_dim)
            
        Returns:
            Dictionary with component names and values
        """
        components = {}
        
        if self.config.joint_dims is not None:
            components["joints"] = actions[..., self.config.joint_dims]
        
        if self.config.gripper_dims is not None:
            components["gripper"] = actions[..., self.config.gripper_dims]
        
        return components


class ObservationSpaceAdapter(nn.Module):
    """Adapter for observation/state space transformations."""
    
    def __init__(self, config: ObservationSpaceConfig):
        """Initialize observation space adapter.
        
        Args:
            config: Observation space configuration
        """
        super().__init__()
        self.config = config
        # Create projesdfctor if needed
        if config.use_projection:
            self.projector = StateProjector(
                original_dim=config.original_dim,
                target_dim=config.max_dim,
                hidden_dim=config.projection_hidden_dim,
            )
        else:
            self.projector = None
        
        # Register normalization parameters
        if config.normalize and config.state_mean is not None:
            self.register_buffer("state_mean", config.state_mean)
            self.register_buffer("state_std", config.state_std)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Process state from robot space to model space.
        
        Args:
            state: State in robot space (..., original_dim)
            
        Returns:
            State in model space (..., max_dim)
        """
        # Normalize if configured
        if self.config.normalize and hasattr(self, "state_mean"):
            state = (state - self.state_mean) / (self.state_std + 1e-8)
        
        # Project or pad
        if self.config.use_projection and self.projector is not None:
            state = self.projector(state)
        else:
            # Simple padding
            if state.shape[-1] < self.config.max_dim:
                state = F.pad(state, (0, self.config.max_dim - state.shape[-1]))
            elif state.shape[-1] > self.config.max_dim:
                state = state[..., :self.config.max_dim]
        
        return state


def create_action_space_config_from_dataset(
    dataset_stats: Dict[str, torch.Tensor],
    original_action_dim: int,
    max_action_dim: int = 32,
    use_projection: bool = False,
    joint_dims: Optional[List[int]] = None,
    gripper_dims: Optional[List[int]] = None,
) -> ActionSpaceConfig:
    """Create action space configuration from dataset statistics.
    
    Args:
        dataset_stats: Dataset statistics containing action mean/std
        original_action_dim: Original action dimension
        max_action_dim: Maximum action dimension for model
        use_projection: Whether to use learned projection
        joint_dims: Indices of joint actions
        gripper_dims: Indices of gripper actions
        
    Returns:
        ActionSpaceConfig instance
    """
    # Extract normalization stats
    action_mean = dataset_stats.get("action_mean", None)
    action_std = dataset_stats.get("action_std", None)
    
    # Ensure correct shape
    if action_mean is not None and action_mean.shape[0] != original_action_dim:
        action_mean = action_mean[:original_action_dim]
    if action_std is not None and action_std.shape[0] != original_action_dim:
        action_std = action_std[:original_action_dim]
    
    return ActionSpaceConfig(
        original_dim=original_action_dim,
        max_dim=max_action_dim,
        joint_dims=joint_dims,
        gripper_dims=gripper_dims,
        use_projection=use_projection,
        normalize=True,
        action_mean=action_mean,
        action_std=action_std,
    )


def create_observation_space_config_from_dataset(
    dataset_stats: Dict[str, torch.Tensor],
    original_state_dim: int,
    max_state_dim: int = 32,
    use_projection: bool = False,
    joint_state_dims: Optional[List[int]] = None,
    gripper_state_dims: Optional[List[int]] = None,
) -> ObservationSpaceConfig:
    """Create observation space configuration from dataset statistics.
    
    Args:
        dataset_stats: Dataset statistics containing state mean/std
        original_state_dim: Original state dimension
        max_state_dim: Maximum state dimension for model
        use_projection: Whether to use learned projection
        joint_state_dims: Indices of joint states
        gripper_state_dims: Indices of gripper states
        
    Returns:
        ObservationSpaceConfig instance
    """
    # Extract normalization stats
    state_mean = dataset_stats.get("state_mean", None)
    state_std = dataset_stats.get("state_std", None)
    
    # Ensure correct shape
    if state_mean is not None and state_mean.shape[0] != original_state_dim:
        state_mean = state_mean[:original_state_dim]
    if state_std is not None and state_std.shape[0] != original_state_dim:
        state_std = state_std[:original_state_dim]
    
    return ObservationSpaceConfig(
        original_dim=original_state_dim,
        max_dim=max_state_dim,
        joint_state_dims=joint_state_dims,
        gripper_state_dims=gripper_state_dims,
        use_projection=use_projection,
        normalize=True,
        state_mean=state_mean,
        state_std=state_std,
    )
