"""
Configuration loader utilities for VLA models.
Handles loading and parsing YAML configuration files for different models.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Utility class for loading model configurations from YAML files."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the config loader.
        
        Args:
            config_dir: Directory containing model configurations. 
                       Defaults to project configs/models directory.
        """
        if config_dir is None:
            # Try to find the configs directory - first check relative to this module
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "configs" / "models"
            
            # If that doesn't exist, try from current working directory
            if not config_dir.exists():
                cwd_config = Path.cwd() / "configs" / "models"
                if cwd_config.exists():
                    config_dir = cwd_config
                else:
                    # Try other common locations
                    parent_configs = Path.cwd().parent / "configs" / "models"
                    if parent_configs.exists():
                        config_dir = parent_configs
        
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {config_dir}")
    
    def load_model_config(self, model_name: str) -> Dict[str, Any]:
        """Load configuration for a specific model.
        
        Args:
            model_name: Name of the model (e.g., 'smolvla', 'openvla', 'pi0')
            
        Returns:
            Dictionary containing the model configuration
            
        Raises:
            FileNotFoundError: If the model configuration file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
        """
        config_file = self.config_dir / f"{model_name}.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            if config is None:
                raise ValueError(f"Empty or invalid configuration file: {config_file}")
                
            logger.info(f"Successfully loaded configuration for model: {model_name}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_file}: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get list of available model configurations.
        
        Returns:
            List of model names that have configuration files
        """
        models = []
        for config_file in self.config_dir.glob("*.yaml"):
            if config_file.is_file():
                models.append(config_file.stem)
        
        return sorted(models)
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get basic information about a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary containing model basic info
        """
        config = self.load_model_config(model_name)
        
        model_section = config.get('model', {})
        
        return {
            'name': model_section.get('name', model_name),
            'display_name': model_section.get('display_name', model_name.title()),
            'version': model_section.get('version', 'unknown'),
            'description': model_section.get('description', 'No description available'),
            'architecture': model_section.get('architecture', {}),
            'hub': model_section.get('hub', {}),
            'capabilities': model_section.get('capabilities', {}),
            'performance': model_section.get('performance', {}),
        }
    
    def get_download_info(self, model_name: str) -> Dict[str, Any]:
        """Get download-related information for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary containing download information
        """
        config = self.load_model_config(model_name)
        model_section = config.get('model', {})
        
        hub_info = model_section.get('hub', {})
        architecture = model_section.get('architecture', {})
        
        return {
            'repo_id': hub_info.get('repo_id'),
            'model_type': hub_info.get('model_type', 'transformers'),
            'library': hub_info.get('library', 'transformers'),
            'size_gb': architecture.get('total_size_gb', 'unknown'),
            'dependencies': model_section.get('dependencies', {}),
            'installation': model_section.get('installation', {}),
        }
    
    def get_model_capabilities(self, model_name: str) -> Dict[str, Any]:
        """Get detailed capabilities of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary containing model capabilities
        """
        config = self.load_model_config(model_name)
        model_section = config.get('model', {})
        
        return {
            'capabilities': model_section.get('capabilities', {}),
            'performance': model_section.get('performance', {}),
            'input': model_section.get('input', {}),
            'output': model_section.get('output', {}),
            'compatibility': model_section.get('compatibility', {}),
        }
    
    def validate_config(self, model_name: str) -> Dict[str, Any]:
        """Validate a model configuration and return any issues.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary containing validation results
        """
        try:
            config = self.load_model_config(model_name)
            model_section = config.get('model', {})
            
            issues = []
            warnings = []
            
            # Check required fields
            required_fields = ['name', 'hub']
            for field in required_fields:
                if field not in model_section:
                    issues.append(f"Missing required field: model.{field}")
            
            # Check hub configuration
            hub_info = model_section.get('hub', {})
            if 'repo_id' not in hub_info:
                issues.append("Missing required field: model.hub.repo_id")
            
            # Check for recommended fields
            recommended_fields = ['description', 'architecture', 'capabilities']
            for field in recommended_fields:
                if field not in model_section:
                    warnings.append(f"Recommended field missing: model.{field}")
            
            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'warnings': warnings,
                'config_loaded': True
            }
            
        except Exception as e:
            return {
                'valid': False,
                'issues': [f"Failed to load configuration: {str(e)}"],
                'warnings': [],
                'config_loaded': False
            }


# Global instance for easy access
_config_loader = None

def get_config_loader(config_dir: Optional[str] = None) -> ConfigLoader:
    """Get a global config loader instance."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)
    return _config_loader
