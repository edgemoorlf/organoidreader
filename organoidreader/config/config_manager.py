"""
Configuration Management Module

Handles configuration settings for the OrganoidReader application.
Supports YAML configuration files with validation and default values.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import yaml
from dataclasses import dataclass, asdict, field


@dataclass
class ProcessingConfig:
    """Configuration for image processing parameters."""
    target_size: tuple = (512, 512)
    normalize: bool = True
    denoise: bool = True
    enhance_contrast: bool = True
    gaussian_blur_sigma: float = 0.5
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple = (8, 8)


@dataclass
class ModelConfig:
    """Configuration for deep learning models."""
    device: str = "auto"  # auto, cpu, cuda, mps
    batch_size: int = 8
    num_workers: int = 4
    precision: str = "float32"  # float32, float16
    model_cache_dir: str = "models"
    checkpoint_dir: str = "checkpoints"


@dataclass
class SegmentationConfig:
    """Configuration for segmentation parameters."""
    min_object_size: int = 100
    max_object_size: int = 50000
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    use_watershed: bool = True
    morphological_opening: bool = True
    remove_border_objects: bool = True


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    size_measurement_unit: str = "pixels"  # pixels, microns
    pixel_size_microns: Optional[float] = None
    calculate_shape_features: bool = True
    calculate_intensity_features: bool = True
    calculate_texture_features: bool = False
    min_area_threshold: float = 50.0
    max_area_threshold: float = 10000.0


@dataclass
class GUIConfig:
    """Configuration for GUI settings."""
    theme: str = "light"  # light, dark
    window_size: tuple = (1200, 800)
    auto_save: bool = True
    auto_save_interval: int = 300  # seconds
    show_tooltips: bool = True
    max_recent_files: int = 10


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_file: bool = True
    log_file: str = "organoidreader.log"
    max_log_size_mb: int = 10
    backup_count: int = 5


@dataclass
class Config:
    """Main configuration class combining all settings."""
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Application settings
    app_name: str = "OrganoidReader"
    version: str = "0.1.0"
    config_version: str = "1.0"
    data_dir: str = "data"
    temp_dir: str = "temp"
    export_dir: str = "exports"


class ConfigManager:
    """Manages configuration loading, saving, and validation."""
    
    DEFAULT_CONFIG_NAME = "config.yaml"
    DEFAULT_CONFIG_DIR = Path.home() / ".organoidreader"
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Path to configuration file. If None, uses default location.
        """
        if config_path is None:
            self.config_dir = self.DEFAULT_CONFIG_DIR
            self.config_path = self.config_dir / self.DEFAULT_CONFIG_NAME
        else:
            self.config_path = Path(config_path)
            self.config_dir = self.config_path.parent
        
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._config = None
        self.logger = logging.getLogger(__name__)
    
    def load_config(self) -> Config:
        """
        Load configuration from file or create default if not exists.
        
        Returns:
            Config object
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
                
                # Convert dict to Config object
                self._config = self._dict_to_config(config_dict)
                self.logger.info(f"Configuration loaded from {self.config_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to load config from {self.config_path}: {e}")
                self.logger.info("Using default configuration")
                self._config = Config()
        else:
            self.logger.info("No configuration file found, creating default")
            self._config = Config()
            self.save_config()
        
        return self._config
    
    def save_config(self, config: Optional[Config] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Config object to save. If None, saves current config.
        """
        if config is not None:
            self._config = config
        
        if self._config is None:
            raise ValueError("No configuration to save")
        
        try:
            config_dict = self._config_to_dict(self._config)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Configuration saved to {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config to {self.config_path}: {e}")
            raise
    
    def get_config(self) -> Config:
        """
        Get current configuration.
        
        Returns:
            Config object
        """
        if self._config is None:
            return self.load_config()
        return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary with configuration updates
        """
        if self._config is None:
            self._config = self.load_config()
        
        # Apply updates
        self._apply_updates(self._config, updates)
        self.save_config()
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> Config:
        """Convert dictionary to Config object."""
        # Extract nested configurations
        processing = ProcessingConfig(**config_dict.get('processing', {}))
        model = ModelConfig(**config_dict.get('model', {}))
        segmentation = SegmentationConfig(**config_dict.get('segmentation', {}))
        analysis = AnalysisConfig(**config_dict.get('analysis', {}))
        gui = GUIConfig(**config_dict.get('gui', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        # Extract main config parameters
        main_params = {k: v for k, v in config_dict.items() 
                      if k not in ['processing', 'model', 'segmentation', 'analysis', 'gui', 'logging']}
        
        return Config(
            processing=processing,
            model=model,
            segmentation=segmentation,
            analysis=analysis,
            gui=gui,
            logging=logging_config,
            **main_params
        )
    
    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        """Convert Config object to dictionary."""
        return {
            'processing': asdict(config.processing),
            'model': asdict(config.model),
            'segmentation': asdict(config.segmentation),
            'analysis': asdict(config.analysis),
            'gui': asdict(config.gui),
            'logging': asdict(config.logging),
            'app_name': config.app_name,
            'version': config.version,
            'config_version': config.config_version,
            'data_dir': config.data_dir,
            'temp_dir': config.temp_dir,
            'export_dir': config.export_dir
        }
    
    def _apply_updates(self, config: Config, updates: Dict[str, Any]) -> None:
        """Apply updates to configuration object."""
        for key, value in updates.items():
            if hasattr(config, key):
                if isinstance(value, dict) and hasattr(getattr(config, key), '__dict__'):
                    # Update nested configuration
                    nested_config = getattr(config, key)
                    for nested_key, nested_value in value.items():
                        if hasattr(nested_config, nested_key):
                            setattr(nested_config, nested_key, nested_value)
                else:
                    # Update main configuration
                    setattr(config, key, value)
    
    def validate_config(self, config: Config) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Args:
            config: Config object to validate
            
        Returns:
            List of validation error messages
        """
        issues = []
        
        # Validate processing config
        if config.processing.target_size[0] <= 0 or config.processing.target_size[1] <= 0:
            issues.append("Processing target_size must be positive")
        
        # Validate model config
        if config.model.batch_size <= 0:
            issues.append("Model batch_size must be positive")
        
        if config.model.device not in ['auto', 'cpu', 'cuda', 'mps']:
            issues.append("Model device must be one of: auto, cpu, cuda, mps")
        
        # Validate segmentation config
        if config.segmentation.confidence_threshold < 0 or config.segmentation.confidence_threshold > 1:
            issues.append("Segmentation confidence_threshold must be between 0 and 1")
        
        # Validate analysis config
        if config.analysis.size_measurement_unit not in ['pixels', 'microns']:
            issues.append("Analysis size_measurement_unit must be 'pixels' or 'microns'")
        
        # Validate GUI config
        if config.gui.theme not in ['light', 'dark']:
            issues.append("GUI theme must be 'light' or 'dark'")
        
        # Validate logging config
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
        if config.logging.level not in valid_log_levels:
            issues.append(f"Logging level must be one of: {valid_log_levels}")
        
        return issues
    
    def create_directories(self, config: Config) -> None:
        """Create necessary directories based on configuration."""
        directories = [
            config.data_dir,
            config.temp_dir,
            config.export_dir,
            config.model.model_cache_dir,
            config.model.checkpoint_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")


# Global configuration instance
_config_manager = None


def get_config_manager(config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None or config_path is not None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config() -> Config:
    """Get current configuration."""
    return get_config_manager().get_config()


def save_config(config: Optional[Config] = None) -> None:
    """Save configuration."""
    get_config_manager().save_config(config)


def update_config(updates: Dict[str, Any]) -> None:
    """Update configuration with new values."""
    get_config_manager().update_config(updates)