"""
Unit tests for image loading functionality.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os

# Import modules to test
from organoidreader.core.image_loader import ImageLoader, ImageLoadError, load_image, get_image_info


class TestImageLoader:
    """Test cases for ImageLoader class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.loader = ImageLoader()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Clean up temporary files
        import shutil
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)
    
    def test_init(self):
        """Test ImageLoader initialization."""
        assert isinstance(self.loader, ImageLoader)
        assert self.loader.metadata_cache == {}
        assert self.loader.SUPPORTED_FORMATS == {'.tiff', '.tif', '.jpg', '.jpeg', '.png', '.czi', '.nd2'}
    
    def test_get_supported_formats(self):
        """Test getting supported formats."""
        formats = self.loader.get_supported_formats()
        expected = ['.czi', '.jpeg', '.jpg', '.nd2', '.png', '.tif', '.tiff']
        assert formats == expected
    
    def test_is_supported_format(self):
        """Test format support checking."""
        assert self.loader.is_supported_format("test.jpg") is True
        assert self.loader.is_supported_format("test.tiff") is True
        assert self.loader.is_supported_format("test.bmp") is False
        assert self.loader.is_supported_format("test.xyz") is False
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(ImageLoadError, match="Image file not found"):
            self.loader.load_image("nonexistent.jpg")
    
    def test_load_unsupported_format(self):
        """Test loading unsupported format raises error."""
        # Create a temporary file with unsupported extension
        temp_file = self.temp_path / "test.xyz"
        temp_file.write_text("dummy content")
        
        with pytest.raises(ImageLoadError, match="Unsupported format"):
            self.loader.load_image(temp_file)
    
    @patch('organoidreader.core.image_loader.Image.open')
    def test_load_standard_image_success(self, mock_pil_open):
        """Test successful loading of standard image formats."""
        # Create a temporary JPEG file
        temp_file = self.temp_path / "test.jpg"
        temp_file.write_bytes(b"dummy jpeg content")
        
        # Mock PIL Image
        mock_image = MagicMock()
        mock_image.format = "JPEG"
        mock_image.mode = "RGB"
        mock_image.info = {"description": "test image"}
        mock_image.__array__ = MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8))
        
        mock_pil_open.return_value.__enter__ = MagicMock(return_value=mock_image)
        mock_pil_open.return_value.__exit__ = MagicMock(return_value=None)
        
        # Test loading
        image_array, metadata = self.loader.load_image(temp_file)
        
        assert isinstance(image_array, np.ndarray)
        assert image_array.shape == (100, 100, 3)
        assert metadata['format'] == 'JPEG'
        assert 'shape' in metadata
        assert 'dtype' in metadata
    
    def test_validate_image_none(self):
        """Test validation with None image."""
        with pytest.raises(ImageLoadError, match="Loaded image is None"):
            self.loader._validate_image(None, Path("test.jpg"))
    
    def test_validate_image_empty(self):
        """Test validation with empty image."""
        empty_array = np.array([])
        with pytest.raises(ImageLoadError, match="Loaded image is empty"):
            self.loader._validate_image(empty_array, Path("test.jpg"))
    
    def test_validate_image_1d(self):
        """Test validation with 1D image."""
        array_1d = np.array([1, 2, 3])
        with pytest.raises(ImageLoadError, match="Image must be at least 2D"):
            self.loader._validate_image(array_1d, Path("test.jpg"))
    
    def test_validate_image_valid(self):
        """Test validation with valid 2D image."""
        valid_array = np.zeros((100, 100))
        # Should not raise an exception
        self.loader._validate_image(valid_array, Path("test.jpg"))
    
    def test_batch_load_empty_list(self):
        """Test batch loading with empty list."""
        results = self.loader.batch_load([])
        assert results == []
    
    @patch.object(ImageLoader, 'load_image')
    def test_batch_load_with_failures(self, mock_load):
        """Test batch loading with some failures."""
        # Mock one success and one failure
        mock_load.side_effect = [
            (np.zeros((10, 10)), {"format": "JPEG"}),  # Success
            ImageLoadError("Failed to load")  # Failure
        ]
        
        paths = ["success.jpg", "failure.jpg"]
        results = self.loader.batch_load(paths)
        
        assert len(results) == 2
        assert results[0][0] is not None  # Success case
        assert results[1][0] is None      # Failure case


class TestUtilityFunctions:
    """Test utility functions."""
    
    @patch.object(ImageLoader, 'load_image')
    def test_load_image_function(self, mock_load):
        """Test load_image utility function."""
        mock_load.return_value = (np.zeros((10, 10)), {"format": "JPEG"})
        
        result = load_image("test.jpg")
        
        assert result[0].shape == (10, 10)
        assert result[1]["format"] == "JPEG"
        mock_load.assert_called_once_with("test.jpg")
    
    def test_get_image_info_nonexistent(self):
        """Test get_image_info with non-existent file."""
        with pytest.raises(ImageLoadError, match="Image file not found"):
            get_image_info("nonexistent.jpg")
    
    def test_get_image_info_success(self):
        """Test get_image_info with existing file."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(b"dummy content")
            tmp_path = tmp.name
        
        try:
            info = get_image_info(tmp_path)
            
            assert 'path' in info
            assert 'filename' in info
            assert 'format' in info
            assert 'file_size' in info
            assert 'supported' in info
            assert info['format'] == '.jpg'
            assert info['supported'] is True
            
        finally:
            os.unlink(tmp_path)


@pytest.fixture
def sample_image_array():
    """Fixture providing a sample image array."""
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_metadata():
    """Fixture providing sample metadata."""
    return {
        "format": "JPEG",
        "shape": (100, 100, 3),
        "dtype": "uint8",
        "file_size": 1024
    }