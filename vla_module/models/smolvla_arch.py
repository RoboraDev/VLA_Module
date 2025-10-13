import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Check if accelerators(CUDA, MPS) are available and use them.
device=""
if torch.accelerator.is_available:
    print("Accelerator is available")
    device = torch.accelerator.current_accelerator()
else:
    print("No Accelerator is found on the system.")
    print("Fallback to CPU")
    device = "cpu"

print(f"Using {device} accelerated device ....")

# Using the SmolVLAPolicy wrapper for SmolVLA Model
model_id="lerobot/smolvla_base"

# Moving the policy/VLA to the accelerator device
policy = SmolVLAPolicy.from_pretrained(model_id).to(device=device)
policy = policy.to(device)

# Tensor details of the policy
tensor_details = policy.named_parameters()
tensor_details_count = len(list(tensor_details))
buffer_details = policy.named_buffers()
buffer_details_count = len(list(buffer_details))

# Total parameters(scalars) count, what we typically refer when we talk about parameters
params_count = 0
for p in policy.parameters():
    params_count += p.numel()

# Fetching the tensors count & any relevant stats
print("Total tensors unit in the SmolVLA Model =>", tensor_details_count)
print("Total count of trainable tensors (excluding buffers, batchnorm RM etc) =>", tensor_details_count - buffer_details_count)
print("Total parameters count in the SmolVLA Model =>", params_count)

# Modules inside the SmolVLA VLA, useful to know which one belongs to action-head only.
modules_count = len(list(policy.modules()))
for name, module in policy.named_modules():
    print(name)

