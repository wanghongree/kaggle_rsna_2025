"""
Configuration management for RSNA Intracranial Aneurysm Detection project.

This module provides functions to load and access project configuration.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import os

def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path to the project root directory
    """
    # Get the directory containing this config.py file, then go up to project root
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    return project_root

def load_config(config_name: str = "paths.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_name: Name of the config file to load
        
    Returns:
        Dictionary containing configuration data
    """
    project_root = get_project_root()
    config_path = project_root / "config" / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

class PathConfig:
    """Configuration class for handling project paths."""
    
    def __init__(self, config_file: str = "paths.yaml"):
        self.config = load_config(config_file)
        self._project_root = Path(self.config['project_root'])
    
    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return self._project_root
    
    def get_path(self, key_path: str) -> Path:
        """Get a path from the configuration.
        
        Args:
            key_path: Dot-separated path to the configuration key (e.g., 'data.series')
            
        Returns:
            Absolute path constructed from project root and relative path
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if key not in value:
                raise KeyError(f"Configuration key '{key_path}' not found")
            value = value[key]
        
        if isinstance(value, str):
            # Convert relative path to absolute path
            return self.project_root / value
        else:
            raise ValueError(f"Configuration key '{key_path}' is not a path string")
    
    def get_data_root(self) -> Path:
        """Get the data root directory."""
        return self.get_path('data.root')
    
    def get_series_dir(self) -> Path:
        """Get the series directory."""
        return self.get_path('data.series')
    
    def get_segmentations_dir(self) -> Path:
        """Get the segmentations directory."""
        return self.get_path('data.segmentations')
    
    def get_train_csv(self) -> Path:
        """Get the train.csv file path."""
        return self.get_path('csv_files.train')
    
    def get_train_localizers_csv(self) -> Path:
        """Get the train_localizers.csv file path."""
        return self.get_path('csv_files.train_localizers')
    
    def get_dicom_metadata_output(self) -> Path:
        """Get the DICOM metadata output file path."""
        return self.get_path('outputs.dicom_metadata')
    
    def get_test_output_path(self) -> Path:
        """Get the test output file path (without extension)."""
        return self.get_path('outputs.test_results')
    
    def get_test_log_path(self) -> Path:
        """Get the test log file path."""
        return self.get_path('outputs.test_log')


# Global instance for easy access
path_config = PathConfig()

def get_config() -> PathConfig:
    """Get the global path configuration instance."""
    return path_config