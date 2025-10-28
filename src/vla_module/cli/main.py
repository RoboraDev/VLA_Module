"""
Main CLI application for VLA Modules.
Provides command-line interface for managing VLA models.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, List
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

from vla_module.config_loader import get_config_loader
from vla_module.models.hub import get_model_hub


class VLACLIError(Exception):
    """Custom exception for VLA CLI errors."""
    pass


def print_success(message: str):
    """Print success message with appropriate formatting."""
    if RICH_AVAILABLE:
        console.print(f"✅ {message}", style="green")
    else:
        print(f"✅ {message}")


def print_error(message: str):
    """Print error message with appropriate formatting."""
    if RICH_AVAILABLE:
        console.print(f"❌ {message}", style="red")
    else:
        print(f"❌ {message}")


def print_warning(message: str):
    """Print warning message with appropriate formatting."""
    if RICH_AVAILABLE:
        console.print(f"⚠️ {message}", style="yellow")
    else:
        print(f"⚠️ {message}")


def print_info(message: str):
    """Print info message with appropriate formatting."""
    if RICH_AVAILABLE:
        console.print(f"ℹ️ {message}", style="blue")
    else:
        print(f"ℹ️ {message}")


def list_models(args):
    """List available models command."""
    try:
        config_loader = get_config_loader()
        available_models = config_loader.get_available_models()
        
        if not available_models:
            print_warning("No model configurations found.")
            return
        
        if RICH_AVAILABLE:
            table = Table(title="Available VLA Models")
            table.add_column("Model", style="cyan", no_wrap=True)
            table.add_column("Display Name", style="magenta")
            table.add_column("Version", style="green")
            table.add_column("Description", style="white")
            
            for model_name in available_models:
                try:
                    model_info = config_loader.get_model_info(model_name)
                    table.add_row(
                        model_name,
                        model_info.get('display_name', model_name),
                        model_info.get('version', 'unknown'),
                        model_info.get('description', '')[:60] + ("..." if len(model_info.get('description', '')) > 60 else "")
                    )
                except Exception as e:
                    table.add_row(model_name, "Error", "unknown", f"Failed to load config: {e}")
            
            console.print(table)
        else:
            print("Available VLA Models:")
            print("-" * 50)
            for model_name in available_models:
                try:
                    model_info = config_loader.get_model_info(model_name)
                    print(f"• {model_name} ({model_info.get('display_name', model_name)})")
                    print(f"  Version: {model_info.get('version', 'unknown')}")
                    print(f"  {model_info.get('description', '')}")
                    print()
                except Exception as e:
                    print(f"• {model_name} - Error loading config: {e}")
        
    except Exception as e:
        print_error(f"Failed to list models: {e}")
        sys.exit(1)


def model_info(args):
    """Get detailed information about a specific model."""
    try:
        config_loader = get_config_loader()
        model_name = args.model
        
        # Check if model exists
        if model_name not in config_loader.get_available_models():
            print_error(f"Model '{model_name}' not found. Use 'vla-cli list' to see available models.")
            sys.exit(1)
        
        model_info = config_loader.get_model_info(model_name)
        model_capabilities = config_loader.get_model_capabilities(model_name)
        download_info = config_loader.get_download_info(model_name)
        
        if RICH_AVAILABLE:
            # Basic Information Panel
            info_text = Text()
            info_text.append(f"Name: {model_info.get('name', 'N/A')}\n", style="bold cyan")
            info_text.append(f"Display Name: {model_info.get('display_name', 'N/A')}\n", style="magenta")
            info_text.append(f"Version: {model_info.get('version', 'N/A')}\n", style="green")
            info_text.append(f"Description: {model_info.get('description', 'N/A')}\n", style="white")
            
            arch = model_info.get('architecture', {})
            if arch:
                info_text.append(f"\nArchitecture:\n", style="bold yellow")
                info_text.append(f"  Type: {arch.get('type', 'N/A')}\n")
                info_text.append(f"  Backbone: {arch.get('backbone', 'N/A')}\n")
                info_text.append(f"  Parameters: {arch.get('parameters', 'N/A')}\n")
                info_text.append(f"  Size: {arch.get('total_size_gb', 'N/A')} GB\n")
            
            console.print(Panel(info_text, title="Model Information", border_style="blue"))
            
            # Hub Information
            hub_info = model_info.get('hub', {})
            if hub_info:
                hub_text = Text()
                hub_text.append(f"Repository: {hub_info.get('repo_id', 'N/A')}\n", style="cyan")
                hub_text.append(f"Library: {hub_info.get('library', 'N/A')}\n")
                hub_text.append(f"License: {hub_info.get('license', 'N/A')}\n")
                console.print(Panel(hub_text, title="Hub Information", border_style="green"))
            
            # Capabilities
            capabilities = model_capabilities.get('capabilities', {})
            if capabilities:
                cap_text = Text()
                for category, items in capabilities.items():
                    cap_text.append(f"{category.title()}:\n", style="bold")
                    if isinstance(items, list):
                        for item in items:
                            cap_text.append(f"  • {item}\n")
                    cap_text.append("\n")
                console.print(Panel(cap_text, title="Capabilities", border_style="yellow"))
            
        else:
            print(f"Model Information: {model_name}")
            print("=" * 50)
            print(f"Name: {model_info.get('name', 'N/A')}")
            print(f"Display Name: {model_info.get('display_name', 'N/A')}")
            print(f"Version: {model_info.get('version', 'N/A')}")
            print(f"Description: {model_info.get('description', 'N/A')}")
            
            arch = model_info.get('architecture', {})
            if arch:
                print("\nArchitecture:")
                print(f"  Type: {arch.get('type', 'N/A')}")
                print(f"  Backbone: {arch.get('backbone', 'N/A')}")
                print(f"  Parameters: {arch.get('parameters', 'N/A')}")
                print(f"  Size: {arch.get('total_size_gb', 'N/A')} GB")
            
            hub_info = model_info.get('hub', {})
            if hub_info:
                print("\nHub Information:")
                print(f"  Repository: {hub_info.get('repo_id', 'N/A')}")
                print(f"  Library: {hub_info.get('library', 'N/A')}")
                print(f"  License: {hub_info.get('license', 'N/A')}")
        
    except Exception as e:
        print_error(f"Failed to get model info: {e}")
        sys.exit(1)


def download_model(args):
    """Download a model command."""
    try:
        model_hub = get_model_hub(args.cache_dir)
        config_loader = get_config_loader()
        
        model_name = args.model
        
        # Check if model exists
        if model_name not in config_loader.get_available_models():
            print_error(f"Model '{model_name}' not found. Use 'vla-cli list' to see available models.")
            sys.exit(1)
        
        # Check dependencies
        deps = model_hub.check_dependencies()
        if not deps.get('huggingface_hub', False):
            print_error("huggingface_hub is required for downloading models.")
            print_info("Install with: pip install huggingface_hub")
            sys.exit(1)
        
        # Check if already downloaded
        if not args.force and model_hub.is_model_downloaded(model_name):
            print_info(f"Model '{model_name}' is already downloaded.")
            print_info("Use --force to re-download.")
            return
        
        # Get model info for display
        model_info = config_loader.get_model_info(model_name)
        download_info = config_loader.get_download_info(model_name)
        
        size_gb = download_info.get('size_gb', 'unknown')
        repo_id = download_info.get('repo_id')
        
        print_info(f"Downloading {model_info.get('display_name', model_name)} ({size_gb} GB)")
        print_info(f"Repository: {repo_id}")
        
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Downloading model...", total=None)
                model_path = model_hub.download_model(
                    model_name, 
                    force_download=args.force,
                    token=args.token
                )
        else:
            print("Downloading... This may take a while.")
            model_path = model_hub.download_model(
                model_name, 
                force_download=args.force,
                token=args.token
            )
        
        print_success(f"Model '{model_name}' downloaded successfully!")
        print_info(f"Location: {model_path}")
        
    except Exception as e:
        print_error(f"Failed to download model: {e}")
        sys.exit(1)


def list_downloaded(args):
    """List downloaded models command."""
    try:
        model_hub = get_model_hub(args.cache_dir)
        downloaded = model_hub.list_downloaded_models()
        
        if not downloaded:
            print_info("No models downloaded yet.")
            print_info("Use 'vla-cli download --model <model_name>' to download a model.")
            return
        
        if RICH_AVAILABLE:
            table = Table(title="Downloaded Models")
            table.add_column("Model", style="cyan", no_wrap=True)
            table.add_column("Display Name", style="magenta")
            table.add_column("Size (GB)", style="green", justify="right")
            table.add_column("Path", style="blue")
            
            for model in downloaded:
                size_str = f"{model.get('actual_size_gb', 0):.2f}" if model.get('actual_size_gb') else "unknown"
                table.add_row(
                    model['name'],
                    model['display_name'],
                    size_str,
                    model['path']
                )
            
            console.print(table)
            
            # Cache info
            cache_info = model_hub.get_cache_info()
            info_text = f"Total: {cache_info['total_models']} models, {cache_info['total_size_gb']:.2f} GB"
            console.print(f"\n{info_text}")
            
        else:
            print("Downloaded Models:")
            print("-" * 60)
            total_size = 0
            for model in downloaded:
                size_gb = model.get('actual_size_gb', 0) or 0
                total_size += size_gb
                print(f"• {model['name']} ({model['display_name']})")
                print(f"  Size: {size_gb:.2f} GB")
                print(f"  Path: {model['path']}")
                print()
            
            print(f"Total: {len(downloaded)} models, {total_size:.2f} GB")
        
    except Exception as e:
        print_error(f"Failed to list downloaded models: {e}")
        sys.exit(1)


def cache_info(args):
    """Show cache information command."""
    try:
        model_hub = get_model_hub(args.cache_dir)
        cache_info = model_hub.get_cache_info()
        
        if RICH_AVAILABLE:
            info_text = Text()
            info_text.append(f"Cache Directory: {cache_info['cache_dir']}\n", style="cyan")
            info_text.append(f"Total Models: {cache_info['total_models']}\n", style="magenta")
            info_text.append(f"Total Size: {cache_info['total_size_gb']} GB\n", style="green")
            
            if cache_info['available_models']:
                info_text.append("\nModels:\n", style="bold yellow")
                for model in cache_info['available_models']:
                    info_text.append(f"  • {model}\n")
            
            console.print(Panel(info_text, title="Cache Information", border_style="blue"))
        else:
            print("Cache Information:")
            print("-" * 30)
            print(f"Directory: {cache_info['cache_dir']}")
            print(f"Total Models: {cache_info['total_models']}")
            print(f"Total Size: {cache_info['total_size_gb']} GB")
            
            if cache_info['available_models']:
                print("\nModels:")
                for model in cache_info['available_models']:
                    print(f"  • {model}")
        
    except Exception as e:
        print_error(f"Failed to get cache info: {e}")
        sys.exit(1)


def validate_config(args):
    """Validate model configurations."""
    try:
        config_loader = get_config_loader()
        
        if args.model:
            models_to_check = [args.model]
            if args.model not in config_loader.get_available_models():
                print_error(f"Model '{args.model}' not found.")
                sys.exit(1)
        else:
            models_to_check = config_loader.get_available_models()
        
        all_valid = True
        
        for model_name in models_to_check:
            validation = config_loader.validate_config(model_name)
            
            if validation['valid']:
                print_success(f"✅ {model_name}: Configuration is valid")
            else:
                print_error(f"❌ {model_name}: Configuration has issues")
                all_valid = False
            
            if validation['issues']:
                for issue in validation['issues']:
                    print_error(f"    Issue: {issue}")
            
            if validation['warnings']:
                for warning in validation['warnings']:
                    print_warning(f"    Warning: {warning}")
        
        if not all_valid:
            sys.exit(1)
        
    except Exception as e:
        print_error(f"Failed to validate configurations: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="vla-cli",
        description="VLA Modules CLI - Manage Vision-Language-Action models"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List models command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.set_defaults(func=list_models)
    
    # Model info command
    info_parser = subparsers.add_parser("info", help="Get detailed model information")
    info_parser.add_argument("--model", "-m", required=True, help="Model name")
    info_parser.set_defaults(func=model_info)
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download a model")
    download_parser.add_argument("--model", "-m", required=True, help="Model name to download")
    download_parser.add_argument("--cache-dir", help="Custom cache directory")
    download_parser.add_argument("--force", "-f", action="store_true", help="Force re-download")
    download_parser.add_argument("--token", help="Hugging Face token for private models")
    download_parser.set_defaults(func=download_model)
    
    # List downloaded command
    downloaded_parser = subparsers.add_parser("downloaded", help="List downloaded models")
    downloaded_parser.add_argument("--cache-dir", help="Custom cache directory")
    downloaded_parser.set_defaults(func=list_downloaded)
    
    # Cache info command
    cache_parser = subparsers.add_parser("cache", help="Show cache information")
    cache_parser.add_argument("--cache-dir", help="Custom cache directory")
    cache_parser.set_defaults(func=cache_info)
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate model configurations")
    validate_parser.add_argument("--model", "-m", help="Specific model to validate (default: all)")
    validate_parser.set_defaults(func=validate_config)
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if command provided
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print_warning("\nOperation cancelled by user.")
        sys.exit(1)
    except VLACLIError as e:
        print_error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
