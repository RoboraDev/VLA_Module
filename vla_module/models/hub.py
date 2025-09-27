"""
Model hub utilities for downloading and managing VLA models.
Handles downloading models from Hugging Face Hub and other sources.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

try:
    from huggingface_hub import snapshot_download, hf_hub_download, login
    from huggingface_hub.utils import HfHubHTTPError
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

from ..config_loader import get_config_loader

logger = logging.getLogger(__name__)


class ModelHub:
    """Handles downloading and managing VLA models from various sources."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the model hub.
        
        Args:
            cache_dir: Directory to cache downloaded models. 
                      Defaults to ~/.cache/vla_modules
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "vla_modules"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config_loader = get_config_loader()
        
        logger.info(f"Model cache directory: {self.cache_dir}")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are available."""
        return {
            'huggingface_hub': HF_HUB_AVAILABLE,
        }
    
    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if a model is already downloaded.
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if model is cached locally
        """
        model_dir = self.cache_dir / model_name
        
        # Check if directory exists and has model files
        if not model_dir.exists():
            return False
        
        # Look for common model file patterns
        model_files = [
            "config.json",
            "pytorch_model.bin", 
            "model.safetensors",
            "adapter_config.json",
        ]
        
        for file_pattern in model_files:
            if list(model_dir.glob(f"**/{file_pattern}")):
                return True
        
        return False
    
    def get_model_size(self, model_name: str) -> Optional[float]:
        """Get the estimated size of a model in GB.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Size in GB or None if unknown
        """
        try:
            download_info = self.config_loader.get_download_info(model_name)
            size_gb = download_info.get('size_gb')
            
            if isinstance(size_gb, (int, float)):
                return float(size_gb)
            elif isinstance(size_gb, str) and size_gb.replace('.', '').isdigit():
                return float(size_gb)
                
        except Exception as e:
            logger.warning(f"Could not get model size for {model_name}: {e}")
        
        return None
    
    def download_model(self, model_name: str, force_download: bool = False, 
                      token: Optional[str] = None) -> str:
        """Download a model from Hugging Face Hub.
        
        Args:
            model_name: Name of the model to download
            force_download: Whether to re-download if already cached
            token: Hugging Face token for private models
            
        Returns:
            Path to the downloaded model directory
            
        Raises:
            ValueError: If model configuration is invalid
            RuntimeError: If download fails
        """
        if not HF_HUB_AVAILABLE:
            raise RuntimeError(
                "huggingface_hub is required for downloading models. "
                "Install with: pip install huggingface_hub"
            )
        
        # Get model download information
        try:
            download_info = self.config_loader.get_download_info(model_name)
            model_info = self.config_loader.get_model_info(model_name)
        except Exception as e:
            raise ValueError(f"Failed to load model configuration: {e}")
        
        repo_id = download_info.get('repo_id')
        if not repo_id:
            raise ValueError(f"No repository ID found for model {model_name}")
        
        model_dir = self.cache_dir / model_name
        
        # Check if already downloaded
        if not force_download and self.is_model_downloaded(model_name):
            logger.info(f"Model {model_name} already cached at {model_dir}")
            return str(model_dir)
        
        # Log download information
        size_gb = download_info.get('size_gb', 'unknown')
        logger.info(f"Downloading {model_info['display_name']} ({size_gb} GB) from {repo_id}")
        
        try:
            # Handle authentication if token provided
            if token:
                login(token=token, write_permission=False)
            
            # Download the model
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(self.cache_dir),
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,  # Use actual files instead of symlinks
                resume_download=True,
                token=token
            )
            
            logger.info(f"Successfully downloaded {model_name} to {model_dir}")
            return str(model_dir)
            
        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                raise RuntimeError(
                    f"Authentication failed. The model {repo_id} may be private. "
                    "Please provide a valid Hugging Face token."
                )
            elif e.response.status_code == 404:
                raise RuntimeError(f"Model not found: {repo_id}")
            else:
                raise RuntimeError(f"Download failed: {e}")
        
        except Exception as e:
            raise RuntimeError(f"Failed to download model {model_name}: {e}")
    
    def list_downloaded_models(self) -> List[Dict[str, Any]]:
        """List all locally downloaded models.
        
        Returns:
            List of dictionaries containing model information
        """
        downloaded = []
        
        if not self.cache_dir.exists():
            return downloaded
        
        for model_dir in self.cache_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                
                # Try to get model info from config
                try:
                    model_info = self.config_loader.get_model_info(model_name)
                    size_gb = self.get_model_size(model_name)
                except Exception:
                    model_info = {
                        'name': model_name,
                        'display_name': model_name.title(),
                        'description': 'Unknown model'
                    }
                    size_gb = None
                
                # Calculate actual disk usage
                try:
                    disk_usage = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                    disk_usage_gb = disk_usage / (1024**3)
                except Exception:
                    disk_usage_gb = None
                
                downloaded.append({
                    'name': model_name,
                    'display_name': model_info.get('display_name', model_name),
                    'description': model_info.get('description', ''),
                    'path': str(model_dir),
                    'estimated_size_gb': size_gb,
                    'actual_size_gb': disk_usage_gb,
                })
        
        return downloaded
    
    def remove_model(self, model_name: str) -> bool:
        """Remove a downloaded model from cache.
        
        Args:
            model_name: Name of the model to remove
            
        Returns:
            True if model was removed successfully
        """
        model_dir = self.cache_dir / model_name
        
        if not model_dir.exists():
            logger.warning(f"Model {model_name} not found in cache")
            return False
        
        try:
            shutil.rmtree(model_dir)
            logger.info(f"Successfully removed model {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove model {model_name}: {e}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the model cache.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_dir.exists():
            return {
                'cache_dir': str(self.cache_dir),
                'total_models': 0,
                'total_size_gb': 0.0,
                'available_models': []
            }
        
        downloaded = self.list_downloaded_models()
        total_size = sum(
            model.get('actual_size_gb', 0) or 0 
            for model in downloaded
        )
        
        return {
            'cache_dir': str(self.cache_dir),
            'total_models': len(downloaded),
            'total_size_gb': round(total_size, 2),
            'available_models': [model['name'] for model in downloaded]
        }


# Global instance for easy access
_model_hub = None

def get_model_hub(cache_dir: Optional[str] = None) -> ModelHub:
    """Get a global model hub instance."""
    global _model_hub
    if _model_hub is None:
        _model_hub = ModelHub(cache_dir)
    return _model_hub
