#!/usr/bin/env python3
"""
VLA Modules Demo Script

Demonstrates how to use the VLA Modules framework programmatically
to load model configurations and manage downloads.
"""

from vla_module.config_loader import get_config_loader
from vla_module.models.hub import get_model_hub


def demo_config_loader():
    """Demo the configuration loader functionality."""
    print(" Configuration Loader Demo")
    print("=" * 40)
    
    # Initialize config loader
    config_loader = get_config_loader()
    
    # List available models
    available_models = config_loader.get_available_models()
    print(f"Available models: {available_models}")
    print()
    
    # Focus on SmolVLA
    model_name = "smolvla"
    print(f"Model Information for '{model_name}':")
    
    # Get basic model info
    model_info = config_loader.get_model_info(model_name)
    print(f"  • Name: {model_info['name']}")
    print(f"  • Display Name: {model_info['display_name']}")
    print(f"  • Version: {model_info['version']}")
    print(f"  • Description: {model_info['description']}")
    print()
    
    # Get architecture details
    arch = model_info.get('architecture', {})
    if arch:
        print("🏗️  Architecture:")
        print(f"  • Type: {arch.get('type')}")
        print(f"  • Backbone: {arch.get('backbone')}")
        print(f"  • Parameters: {arch.get('parameters')}")
        print(f"  • Size: {arch.get('total_size_gb')} GB")
        print()
    
    # Get capabilities
    capabilities = config_loader.get_model_capabilities(model_name)
    caps = capabilities.get('capabilities', {})
    if caps:
        print("Capabilities:")
        for category, items in caps.items():
            print(f"  • {category.title()}:")
            if isinstance(items, list):
                for item in items:
                    print(f"    - {item}")
        print()
    
    # Get download info
    download_info = config_loader.get_download_info(model_name)
    print("Download Information:")
    print(f"  • Repository: {download_info.get('repo_id')}")
    print(f"  • Library: {download_info.get('library')}")
    print(f"  • Size: {download_info.get('size_gb')} GB")
    print()


def demo_model_hub():
    """Demo the model hub functionality."""
    print("Model Hub Demo")
    print("=" * 40)
    
    # Initialize model hub
    model_hub = get_model_hub()
    
    # Check cache info
    cache_info = model_hub.get_cache_info()
    print("Cache Information:")
    print(f"  • Cache Directory: {cache_info['cache_dir']}")
    print(f"  • Total Models: {cache_info['total_models']}")
    print(f"  • Total Size: {cache_info['total_size_gb']} GB")
    print()
    
    # List downloaded models
    downloaded = model_hub.list_downloaded_models()
    if downloaded:
        print("Downloaded Models:")
        for model in downloaded:
            print(f"  • {model['name']} ({model['display_name']})")
            if model.get('actual_size_gb'):
                print(f"    Size: {model['actual_size_gb']:.2f} GB")
            print(f"    Path: {model['path']}")
        print()
    
    # Check if SmolVLA is downloaded
    model_name = "smolvla"
    is_downloaded = model_hub.is_model_downloaded(model_name)
    print(f"SmolVLA is {'already downloaded' if is_downloaded else 'not downloaded'}")
    
    if is_downloaded:
        print("You can now use the model for inference!")
    else:
        print("Run 'uv run vla-cli download --model smolvla' to download it.")
    print()


def demo_usage_examples():
    """Show usage examples from the configuration."""
    print("Usage Examples from Configuration")
    print("=" * 40)
    
    config_loader = get_config_loader()
    config = config_loader.load_model_config("smolvla")
    
    usage = config.get('model', {}).get('usage', {})
    
    # Show download example
    if 'download' in usage:
        print("📥 Download Examples:")
        download_info = usage['download']
        if 'cli' in download_info:
            print(f"  CLI: {download_info['cli']}")
        if 'python' in download_info:
            print("  Python:")
            for line in download_info['python'].strip().split('\n'):
                print(f"    {line}")
        print()
    
    # Show training examples
    if 'training' in usage:
        print("🎓 Training Examples:")
        training_info = usage['training']
        
        if 'fine_tune_action_head' in training_info:
            print("  Fine-tune Action Head:")
            for line in training_info['fine_tune_action_head'].strip().split('\n'):
                print(f"    {line}")
        print()


def main():
    """Main demo function."""
    print("VLA Modules - Programmatic Usage Demo")
    print("=" * 50)
    print()
    
    try:
        # Demo configuration loader
        demo_config_loader()
        
        # Demo model hub
        demo_model_hub()
        
        # Demo usage examples
        demo_usage_examples()
        
        print("Demo completed successfully!")
        print("For more examples, check the CLI with 'uv run vla-cli --help'")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        raise


if __name__ == "__main__":
    main()
