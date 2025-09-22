# Robora VLA Modules: Modular Vision-Language-Action Fine-tuning for Robotics Infrastructure

A self-contained, modular Python framework for integrating Vision-Language-Action (VLA) models into robotics systems. This project provides a plug-and-play solution for adapting pre-trained VLA models to specific robot platforms and tasks, enabling rapid development of robotic applications that understand natural language commands and perform complex physical actions.

## Overview

This framework offers a unified interface for state-of-the-art open source VLA models including SmolVLA, π₀ (Pi-Zero), OpenVLA, and GR00T N1.5. It streamlines the complete VLA lifecycle from model downloading and optimized inference to fine-tuning and deployment. The framework features hardware-agnostic interfaces, simulation integration with PyBullet/Gymnasium, and a powerful CLI ecosystem for managing VLA models.

## Features

- **Modular Architecture**: Self-contained system designed for easy integration into existing robotics infrastructure
- **Multi-Model Support**: Unified interface for SmolVLA, OpenVLA, Pi0, Pi0-FAST, and GrootN1.5 models
- **Hardware Flexibility**: Optimized for both consumer-grade and enterprise GPUs
- **Simulation Integration**: Native support for PyBullet and Gymnasium environments
- **Fine-tuning Strategies**: Support for both action head-only and full model fine-tuning
- **RESTful API**: FastAPI-based server for model serving and inference
- **CLI Management**: Comprehensive command-line interface for model lifecycle management
- **Cross-Platform**: Compatible with various robot morphologies and form factors

## Prerequisites

- Python 3.11
- UV package manager : [UV Docs](https://docs.astral.sh/uv/)
- NVIDIA (CUDA-compatible) GPU (recommended for optimal performance)

## Installation

This project uses the UV package manager for dependency management. Ensure you have UV installed on your system.

```bash
# Clone the repository
git clone https://github.com/RoboraDev/VLA_Module.git
cd VLA_Module

# Install dependencies using UV (Recommended)
uv sync

# Or if you want to use python 3.11 virtual environment
python311 -m venv .venv

# Activate the virtual environment (If needed)
source .venv/bin/activate
```

## Project Structure (Subject to change)

```
VLA_Module/
├── vla_module/                   # Core module package
│   ├── actions/
│   │   ├── __init__.py
│   │   └── tokenizers.py         # Logic for discretizing action bins
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── server.py             # FastAPI application for serving models
│   │   └── endpoints.py          # API endpoints for inference
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py               # Main CLI application
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataloaders.py        # Unified dataset loaders from LeRobot
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hub.py                # Functions to download models from HF
│   │   ├── architectures.py      # Model wrapper classes for VLA models
│   │   └── inference.py          # Inference pipeline logic
│   │
│   ├── simulation/
│   │   ├── __init__.py
│   │   └── env_wrappers.py       # Wrappers for PyBullet and Gymnasium API
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainers.py           # Core training loops (Imitation, RL)
│   │   ├── action_head_tuner.py  # Logic for fine-tuning action heads
│   │   └── vlm_backbone_tuner.py # Logic for full VLM backbone fine-tuning
│   │
│   └── utils/
│       ├── __init__.py
│       └── config_loader.py      # Utilities for loading YAML configs
│
├── configs/
│   ├── models/
│   │   ├── pi0.yaml
│   │   └── openvla.yaml
│   └── training/
│       └── default_imitation.yaml
│
├── demos/
│   └── run_inference_demo.py     # Demo scripts for inference
│
├── tests/
│   ├── test_cli.py
│   └── test_inference.py
│
├── .gitignore
├── LICENSE
├── pyproject.toml                # Project metadata and dependencies
├── uv.lock                       # UV environment lock file
└── README.md                     # Project documentation
```

## Supported Models

### Model Comparison

| Model | Parameters | Key Features | Efficiency |
|-------|------------|--------------|------------|
| SmolVLA | 450M | SmolVLM2 + Flow Matching, consumer hardware compatible | Real-time control with flow matching |
| OpenVLA | 7B | Llama 2 backbone with DINOv2/SigLIP encoders | LoRA fine-tuning, quantization support |
| Pi0 (π₀) | 3B | PaliGemma VLM + flow matching, 7-platform training | 50Hz action generation, cross-embodiment |
| Pi0-FAST | 3B | Autoregressive with DCT frequency compression | Improved efficiency with FAST tokenization |
| GrootN1.5 | 2.1B+ | NVIDIA Eagle VLM, frozen architecture | 93.3% language following rate |

## Usage

### Command Line Interface (Subject to change)

```bash
# Download and setup a model
uv run vla-cli download --model smolvla --cache-dir ./models

# Run inference on an image
uv run vla-cli infer --model smolvla --image path/to/image.jpg --command "pick up the red cube"

# Start fine-tuning process
uv run vla-cli train --model pi0 --dataset custom_dataset --strategy action-head

# Launch API server
uv run vla-cli serve --model openvla --host 0.0.0.0 --port 8000
```

### API Server (Subject to change)

```python
import requests

# Start the server
response = requests.post("http://localhost:8000/inference", json={
    "image_path": "path/to/image.jpg",
    "command": "navigate to the kitchen",
    "model": "smolvla"
})

actions = response.json()["actions"]
```

### Programmatic Usage (Subject to change)

```python
from vla_module.models import load_model
from vla_module.models.inference import VLAInference

# Load a model
model = load_model("smolvla", device="cuda")
inference_engine = VLAInference(model)

# Run inference
actions = inference_engine.predict(
    image_path="observation.jpg",
    command="move forward and grasp the object"
)
```

## Fine-tuning Strategies

### Action Head Fine-tuning (Recommended)

Freeze the VLM backbone while fine-tuning only the action head for custom robot morphologies. This approach offers computational efficiency, knowledge preservation, and rapid convergence.

```bash
uv run vla-cli train \
    --model pi0 \
    --strategy action-head \
    --dataset custom_robot_data \
    --epochs 50 \
    --lr 1e-4
```

### Full Model Fine-tuning

Fine-tune both VLM backbone and action head for significant environmental or embodiment changes.

```bash
uv run vla-cli train \
    --model openvla \
    --strategy full-model \
    --dataset domain_shift_data \
    --epochs 100 \
    --lr 5e-5
```

## Simulation Integration

The framework supports PyBullet and Gymnasium environments for simulation-based fine-tuning:

```python
from vla_module.simulation import PyBulletWrapper
from vla_module.training import ImitationTrainer

# Setup simulation environment
env = PyBulletWrapper("manipulation_task.urdf")
trainer = ImitationTrainer(model="smolvla", env=env)

# Generate training episodes
trainer.collect_episodes(num_episodes=1000)
trainer.fine_tune(epochs=50)
```

## Configuration

Model and training configurations are stored in YAML files under the `configs/` directory. Customize these files for specific robot platforms and training scenarios.

## Development

### Running Tests

```bash
uv run pytest tests/ -v
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

---

### Contributors
[@shiven](https://github.com/shiven-saini/) , updating...
