"""Freezing Utilities for VLM Backbone

This module provides utilities to freeze/unfreeze vision-language model components
for action-head-only fine-tuning or selective fine-tuning strategies.
"""

from typing import List, Optional, Set

import torch
import torch.nn as nn


def freeze_module(module: nn.Module) -> None:
    """Freeze all parameters in a module.
    
    Args:
        module: PyTorch module to freeze
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """Unfreeze all parameters in a module.
    
    Args:
        module: PyTorch module to unfreeze
    """
    for param in module.parameters():
        param.requires_grad = True


def freeze_parameters_by_name(
    model: nn.Module,
    parameter_names: List[str],
    freeze: bool = True,
) -> None:
    """Freeze or unfreeze specific parameters by name.
    
    Args:
        model: Model containing the parameters
        parameter_names: List of parameter names to freeze/unfreeze
        freeze: If True, freeze the parameters; if False, unfreeze them
    """
    for name, param in model.named_parameters():
        if any(pname in name for pname in parameter_names):
            param.requires_grad = not freeze


def get_frozen_parameters(model: nn.Module) -> Set[str]:
    """Get names of all frozen parameters in a model.
    
    Args:
        model: Model to inspect
        
    Returns:
        Set of parameter names that are frozen (requires_grad=False)
    """
    frozen = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen.add(name)
    return frozen


def get_trainable_parameters(model: nn.Module) -> Set[str]:
    """Get names of all trainable parameters in a model.
    
    Args:
        model: Model to inspect
        
    Returns:
        Set of parameter names that are trainable (requires_grad=True)
    """
    trainable = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable.add(name)
    return trainable


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count parameters in a model.
    
    Args:
        model: Model to count parameters for
        trainable_only: If True, only count trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def freeze_vision_encoder(
    model: nn.Module,
    model_type: str = "smolvla",
) -> None:
    """Freeze vision encoder in VLA model.
    
    Args:
        model: VLA model (SmolVLA or PI0)
        model_type: Type of model ("smolvla" or "pi0")
    """
    if model_type.lower() == "smolvla":
        # SmolVLA structure: model.vlm.vision_model or model.vlm.vision_tower
        if hasattr(model, "vlm"):
            if hasattr(model.vlm, "vision_model"):
                freeze_module(model.vlm.vision_model)
            elif hasattr(model.vlm, "vision_tower"):
                freeze_module(model.vlm.vision_tower)
            elif hasattr(model.vlm, "vision_encoder"):
                freeze_module(model.vlm.vision_encoder)
        # Also check if it's wrapped in 'model' attribute
        elif hasattr(model, "model"):
            if hasattr(model.model, "vlm"):
                if hasattr(model.model.vlm, "vision_model"):
                    freeze_module(model.model.vlm.vision_model)
    
    elif model_type.lower() == "pi0":
        # PI0 structure: model.vision_model or model.multi_modal_projector
        if hasattr(model, "vision_model"):
            freeze_module(model.vision_model)
        elif hasattr(model, "vision_tower"):
            freeze_module(model.vision_tower)
        # Check wrapped structure
        elif hasattr(model, "model"):
            if hasattr(model.model, "vision_model"):
                freeze_module(model.model.vision_model)


def freeze_language_model(
    model: nn.Module,
    model_type: str = "smolvla",
    freeze_embeddings: bool = True,
) -> None:
    """Freeze language model in VLA model.
    
    Args:
        model: VLA model (SmolVLA or PI0)
        model_type: Type of model ("smolvla" or "pi0")
        freeze_embeddings: Whether to also freeze embedding layers
    """
    if model_type.lower() == "smolvla":
        # SmolVLA structure
        if hasattr(model, "vlm"):
            vlm = model.vlm
            # Freeze language model layers
            if hasattr(vlm, "language_model"):
                freeze_module(vlm.language_model)
            elif hasattr(vlm, "text_model"):
                freeze_module(vlm.text_model)
            
            # Optionally freeze embeddings
            if freeze_embeddings:
                if hasattr(vlm, "embed_tokens"):
                    freeze_module(vlm.embed_tokens)
                if hasattr(vlm, "embeddings"):
                    freeze_module(vlm.embeddings)
    
    elif model_type.lower() == "pi0":
        # PI0 structure
        if hasattr(model, "language_model"):
            freeze_module(model.language_model)
        elif hasattr(model, "text_model"):
            freeze_module(model.text_model)
        
        # Check wrapped structure
        if hasattr(model, "model"):
            if hasattr(model.model, "language_model"):
                freeze_module(model.model.language_model)


def freeze_vlm_backbone(
    model: nn.Module,
    model_type: str = "smolvla",
    freeze_vision: bool = True,
    freeze_language: bool = True,
) -> None:
    """Freeze the entire VLM backbone (vision + language).
    
    This is the main function to use for action-head-only fine-tuning.
    
    Args:
        model: VLA model
        model_type: Type of model ("smolvla" or "pi0")
        freeze_vision: Whether to freeze vision encoder
        freeze_language: Whether to freeze language model
    """
    if freeze_vision:
        freeze_vision_encoder(model, model_type)
    
    if freeze_language:
        freeze_language_model(model, model_type)


def freeze_vlm_layers_selective(
    model: nn.Module,
    model_type: str = "smolvla",
    num_layers_to_freeze: Optional[int] = None,
    freeze_from_bottom: bool = True,
) -> None:
    """Selectively freeze layers in the VLM.
    
    Useful for progressive unfreezing or layer-wise fine-tuning.
    
    Args:
        model: VLA model
        model_type: Type of model
        num_layers_to_freeze: Number of layers to freeze (None = all)
        freeze_from_bottom: If True, freeze from bottom layers; else from top
    """
    # Get the language model
    language_model = None
    if model_type.lower() == "smolvla":
        if hasattr(model, "vlm"):
            if hasattr(model.vlm, "language_model"):
                language_model = model.vlm.language_model
            elif hasattr(model.vlm, "text_model"):
                language_model = model.vlm.text_model
    elif model_type.lower() == "pi0":
        if hasattr(model, "language_model"):
            language_model = model.language_model
    
    if language_model is None:
        print("Warning: Could not find language model to freeze selectively")
        return
    
    # Get layers
    if hasattr(language_model, "layers"):
        layers = language_model.layers
    elif hasattr(language_model, "encoder") and hasattr(language_model.encoder, "layers"):
        layers = language_model.encoder.layers
    else:
        print("Warning: Could not find layers in language model")
        return
    
    # Determine which layers to freeze
    if num_layers_to_freeze is None:
        num_layers_to_freeze = len(layers)
    
    if freeze_from_bottom:
        layers_to_freeze = list(range(num_layers_to_freeze))
    else:
        total_layers = len(layers)
        layers_to_freeze = list(range(total_layers - num_layers_to_freeze, total_layers))
    
    # Freeze selected layers
    for idx in layers_to_freeze:
        if idx < len(layers):
            freeze_module(layers[idx])


def unfreeze_action_head(
    model: nn.Module,
    model_type: str = "smolvla",
) -> None:
    """Unfreeze the action prediction head.
    
    Args:
        model: VLA model
        model_type: Type of model
    """
    if model_type.lower() == "smolvla":
        # SmolVLA has action expert
        if hasattr(model, "action_expert"):
            unfreeze_module(model.action_expert)
        elif hasattr(model, "expert"):
            unfreeze_module(model.expert)
        # Also unfreeze action-related projections
        if hasattr(model, "action_head"):
            unfreeze_module(model.action_head)
        if hasattr(model, "state_projection"):
            unfreeze_module(model.state_projection)
    
    elif model_type.lower() == "pi0":
        # PI0 action head
        if hasattr(model, "action_head"):
            unfreeze_module(model.action_head)
        if hasattr(model, "action_decoder"):
            unfreeze_module(model.action_decoder)


def setup_action_head_only_training(
    model: nn.Module,
    model_type: str = "smolvla",
    also_train_projections: bool = True,
) -> None:
    """Setup model for action-head-only training.
    
    This freezes the VLM backbone and unfreezes only the action prediction components.
    
    Args:
        model: VLA model
        model_type: Type of model ("smolvla" or "pi0")
        also_train_projections: Whether to also train projection layers (recommended)
    """
    # First, freeze everything
    freeze_module(model)
    
    # Then unfreeze action head
    unfreeze_action_head(model, model_type)
    
    # Optionally unfreeze projection layers
    if also_train_projections:
        # Unfreeze any projection layers that connect VLM to action space
        for name, param in model.named_parameters():
            if any(keyword in name.lower() for keyword in [
                "projection", "adapter", "state_proj", "action_proj"
            ]):
                param.requires_grad = True


def print_trainable_parameters(model: nn.Module, verbose: bool = False) -> None:
    """Print summary of trainable vs frozen parameters.
    
    Args:
        model: Model to analyze
        verbose: If True, print all parameter names
    """
    trainable_params = 0
    frozen_params = 0
    
    trainable_names = []
    frozen_names = []
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        if param.requires_grad:
            trainable_params += num_params
            trainable_names.append(name)
        else:
            frozen_params += num_params
            frozen_names.append(name)
    
    total_params = trainable_params + frozen_params
    trainable_percentage = 100 * trainable_params / total_params if total_params > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"Parameter Summary:")
    print(f"{'='*80}")
    print(f"Total parameters:      {total_params:,} ({format_number(total_params)})")
    print(f"Trainable parameters:  {trainable_params:,} ({format_number(trainable_params)}) - {trainable_percentage:.2f}%")
    print(f"Frozen parameters:     {frozen_params:,} ({format_number(frozen_params)}) - {100-trainable_percentage:.2f}%")
    print(f"{'='*80}\n")
    
    if verbose:
        print("\nTrainable parameters:")
        for name in trainable_names:
            print(f"  ✓ {name}")
        
        print("\nFrozen parameters:")
        for name in frozen_names:
            print(f"  ✗ {name}")


def format_number(num: int) -> str:
    """Format large numbers with K, M, B suffixes.
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)


def validate_frozen_state(
    model: nn.Module,
    expected_trainable_patterns: Optional[List[str]] = None,
    expected_frozen_patterns: Optional[List[str]] = None,
) -> bool:
    """Validate that the model's frozen state matches expectations.
    
    Args:
        model: Model to validate
        expected_trainable_patterns: List of patterns that should be trainable
        expected_frozen_patterns: List of patterns that should be frozen
        
    Returns:
        True if validation passes, False otherwise
    """
    all_valid = True
    
    if expected_trainable_patterns:
        for pattern in expected_trainable_patterns:
            found_trainable = False
            for name, param in model.named_parameters():
                if pattern in name and param.requires_grad:
                    found_trainable = True
                    break
            
            if not found_trainable:
                print(f"Warning: Expected trainable pattern '{pattern}' not found or frozen")
                all_valid = False
    
    if expected_frozen_patterns:
        for pattern in expected_frozen_patterns:
            found_frozen = False
            for name, param in model.named_parameters():
                if pattern in name and not param.requires_grad:
                    found_frozen = True
                    break
            
            if not found_frozen:
                print(f"Warning: Expected frozen pattern '{pattern}' not found or trainable")
                all_valid = False
    
    return all_valid
