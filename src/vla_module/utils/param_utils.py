import torch
import logging
from torch import nn

"""
These utils will be responsible later on, might be to know the device or dtype of parameters
Maybe useful when not sure where the model is loaded because the observation tensors must be mapped properly for that particular device.
"""

def get_device_from_param(module: nn.Module) -> torch.device:
    """Get device info from reading one of the parameters.

    Note: assumes that all parameters have the same device, which most likely is!
    """

    return next(iter(module.parameters())).device


def get_dtype_from_param(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note: assumes that all parameters have the same dtype, I am not sure about it; only after due testing I can know.
    Models often use mixed precision for vision towers, language part etc.
    """

    return next(iter(module.parameters())).dtype


def get_output_shape(module: nn.Module, input_shape: tuple) -> tuple:
    """
    Calculates the output shape of a PyTorch module given an input shape,
    will be useful to know when dealing with action_encoder, state_projector, action_decoder architecture details.

    Args:
        module (nn.Module): a PyTorch module
        input_shape (tuple): A tuple representing the input shape, e.g., (batch_size, channels, height, width)

    Returns:
        tuple: The output shape of the module.
    """

    # Generate random input data to perform inference on that specific module and to know the output dimension features.
    dummy_input = torch.rand(size=input_shape)

    with torch.inference_mode():
        output = module(dummy_input)

    return tuple(output.shape)


def model_loading_keys_debugging(missing_keys: list[str], unexpected_keys: list[str]) -> None:
    """Log missing and unexpected keys when loading a model;
    Useful to debug our custom VLA Action head architecture when we change it's input output features of specific modules for
    more than 32 DOFs support.

    Will be used in conjuction with others.

    Args:
        missing_keys (list[str]): Keys that were expected but not found.
        unexpected_keys (list[str]): Keys that were found but not expected.
    """
    if missing_keys:
        logging.warning(f"Missing key(s) when loading model: {missing_keys}")

    if unexpected_keys:
        logging.warning(f"Unexpected key(s) when loading model: {unexpected_keys}")

