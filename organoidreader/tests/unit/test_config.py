"""
Unit tests for configuration management.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from organoidreader.config.config_manager import (
    Config, ProcessingConfig, ModelConfig, SegmentationConfig,
    AnalysisConfig, GUIConfig, LoggingConfig, ConfigManager,
    get_config_manager, get_config
)


class TestConfigDataClasses:
    """Test configuration data classes."""
    
    def test_processing_config_defaults(self):
        """Test ProcessingConfig default values."""
        config = ProcessingConfig()
        assert config.target_size == (512, 512)
        assert config.normalize is True
        assert config.denoise is True
        assert config.enhance_contrast is True
        assert config.gaussian_blur_sigma == 0.5
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        assert config.device == "auto"
        assert config.batch_size == 8
        assert config.num_workers == 4
        assert config.precision == "float32"
    
    def test_config_initialization(self):
        """Test main Config initialization."""
        config = Config()
        assert isinstance(config.processing, ProcessingConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.segmentation, SegmentationConfig)
        assert isinstance(config.analysis, AnalysisConfig)
        assert isinstance(config.gui, GUIConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert config.app_name == "OrganoidReader"
        assert config.version == "0.1.0"


class TestConfigManager:
    """Test ConfigManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        self.config_path = self.temp_path / "test_config.yaml"
        self.manager = ConfigManager(self.config_path)
    
    def teardown_method(self):
        """Clean up after tests."""
        import shutil
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)
    
    def test_init_with_custom_path(self):
        """Test ConfigManager initialization with custom path."""
        assert self.manager.config_path == self.config_path
        assert self.manager.config_dir == self.config_path.parent
    
    def test_init_with_default_path(self):
        """Test ConfigManager initialization with default path."""
        manager = ConfigManager()
        assert manager.config_path == ConfigManager.DEFAULT_CONFIG_DIR / ConfigManager.DEFAULT_CONFIG_NAME
    
    def test_load_config_creates_default(self):
        """Test loading config creates default when file doesn't exist."""
        config = self.manager.load_config()
        
        assert isinstance(config, Config)
        assert self.config_path.exists()
        assert config.app_name == "OrganoidReader"
    
    def test_load_config_from_file(self):
        """Test loading config from existing file."""
        # Create a test config file
        test_config = {
            'app_name': 'TestApp',
            'version': '1.0.0',
            'processing': {
                'target_size': [256, 256],
                'normalize': False
            },
            'model': {
                'batch_size': 16
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(test_config, f)
        
        config = self.manager.load_config()
        
        assert config.app_name == 'TestApp'
        assert config.version == '1.0.0'
        assert config.processing.target_size == (256, 256)
        assert config.processing.normalize is False
        assert config.model.batch_size == 16
    
    def test_save_config(self):
        """Test saving configuration to file."""
        config = Config()
        config.app_name = "SavedApp"
        config.processing.target_size = (1024, 1024)
        
        self.manager.save_config(config)
        
        assert self.config_path.exists()
        
        # Verify saved content
        with open(self.config_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data['app_name'] == "SavedApp"
        assert saved_data['processing']['target_size'] == [1024, 1024]
    
    def test_get_config_loads_if_none(self):
        """Test get_config loads config if none exists."""
        assert self.manager._config is None
        
        config = self.manager.get_config()
        
        assert config is not None
        assert self.manager._config is config
    
    def test_update_config(self):
        """Test updating configuration."""
        # Load initial config
        self.manager.load_config()
        
        updates = {
            'app_name': 'UpdatedApp',
            'processing': {
                'normalize': False,
                'target_size': [128, 128]
            }
        }
        
        self.manager.update_config(updates)
        
        config = self.manager.get_config()
        assert config.app_name == 'UpdatedApp'
        assert config.processing.normalize is False
        assert config.processing.target_size == (128, 128)
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid config."""
        config = Config()
        issues = self.manager.validate_config(config)
        assert issues == []
    
    def test_validate_config_invalid(self):
        """Test configuration validation with invalid config."""
        config = Config()
        config.processing.target_size = (-1, 0)  # Invalid
        config.model.batch_size = 0  # Invalid
        config.model.device = "invalid"  # Invalid
        config.segmentation.confidence_threshold = 1.5  # Invalid
        
        issues = self.manager.validate_config(config)
        
        assert len(issues) > 0
        assert any("target_size must be positive" in issue for issue in issues)
        assert any("batch_size must be positive" in issue for issue in issues)
        assert any("device must be one of" in issue for issue in issues)
        assert any("confidence_threshold must be between 0 and 1" in issue for issue in issues)
    
    @patch('pathlib.Path.mkdir')
    def test_create_directories(self, mock_mkdir):
        """Test directory creation."""
        config = Config()
        config.data_dir = "test_data"
        config.temp_dir = "test_temp"
        
        self.manager.create_directories(config)
        
        # Verify mkdir was called for expected directories
        expected_calls = len([
            config.data_dir,
            config.temp_dir,
            config.export_dir,
            config.model.model_cache_dir,
            config.model.checkpoint_dir
        ])
        
        assert mock_mkdir.call_count == expected_calls


class TestGlobalFunctions:
    """Test global configuration functions."""
    
    def test_get_config_manager_singleton(self):
        """Test get_config_manager returns same instance."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        assert manager1 is manager2
    
    def test_get_config_manager_new_path(self):
        """Test get_config_manager with new path creates new instance."""
        manager1 = get_config_manager()
        manager2 = get_config_manager("new_path.yaml")
        
        assert manager1 is not manager2
    
    @patch.object(ConfigManager, 'get_config')
    def test_get_config_function(self, mock_get_config):
        """Test get_config global function."""
        mock_config = Config()
        mock_get_config.return_value = mock_config
        
        config = get_config()
        
        assert config is mock_config
        mock_get_config.assert_called_once()