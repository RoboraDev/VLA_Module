import torch, logging

def get_device():
    """
    Returns the available device accelerator for PyTorch.
    
    Checks for available accelerators in the following order:
    1. CUDA (NVIDIA GPUs and AMD GPUs via ROCm on HIP)
    2. MPS (Apple Silicon)
    3. CPU (fallback)
    
    Returns:
        str: Device string that can be directly used with .to(device) in pytorch
             Examples: "cuda:0", "mps", "cpu"
    """
    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
        return device
    
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    
    else:
        # Fallback to CPU
        return "cpu"

def get_device_torch() -> torch.device:
    """Tries to select automatically a torch device."""
    if torch.cuda.is_available():
        logging.info("Cuda backend detected, using cuda.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logging.info("Metal backend detected, using mps.")
        return torch.device("mps")
    else:
        logging.warning("No accelerated backend detected. Using default cpu, this will be slow.")
        return torch.device("cpu")

def is_torch_device_available(try_device: str) -> bool:
    try_device = str(try_device)  # Ensure try_device is a string
    if try_device == "cuda":
        return torch.cuda.is_available()
    elif try_device == "mps":
        return torch.backends.mps.is_available()
    elif try_device == "cpu":
        return True
    else:
        raise ValueError(f"Unknown device {try_device}. Supported devices are: cuda, mps or cpu.")

def get_device_info():
    """
    Returns detailed information about the available device.
    
    Returns:
        dict: Dictionary containing device information
            - device: Device string (e.g., "cuda:0", "mps", "cpu")
            - type: Device type (e.g., "cuda", "mps", "cpu")
            - name: Device name (GPU name or CPU)
            - count: Number of available devices (for CUDA)
    """
    device_str = get_device()
    device_type = device_str.split(':')[0]
    
    info = {
        "device": device_str,
        "type": device_type,
        "name": None,
        "count": 1
    }
    
    if device_type == "cuda":
        info["name"] = torch.cuda.get_device_name(torch.cuda.current_device())
        info["count"] = torch.cuda.device_count()
    elif device_type == "mps":
        info["name"] = "Apple MPS"
    else:
        info["name"] = "CPU"
    
    return info

if __name__ == "__main__":
    
    # Added, just so I can test it out single as well.
    print(f"Device: {get_device()}")
    print(f"\nDevice Info:")
    for key, value in get_device_info().items():
        print(f"  {key}: {value}")
