"""
Image Loading Module

This module provides functionality for loading and basic validation of microscopy images
in various formats commonly used in organoid research.

Supported formats:
- TIFF (including multi-page TIFF)
- JPEG
- PNG
- CZI (Zeiss microscopy format)
- ND2 (Nikon microscopy format)
"""

import os
import logging
from typing import Union, List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import tifffile
from skimage import io, img_as_float, img_as_ubyte

# Set up logging
logger = logging.getLogger(__name__)


class ImageLoadError(Exception):
    """Custom exception for image loading errors."""
    pass


class ImageLoader:
    """
    A comprehensive image loader for microscopy images used in organoid analysis.
    
    Supports multiple formats and provides metadata extraction capabilities.
    """
    
    SUPPORTED_FORMATS = {'.tiff', '.tif', '.jpg', '.jpeg', '.png', '.czi', '.nd2'}
    
    def __init__(self):
        """Initialize the ImageLoader."""
        self.metadata_cache = {}
    
    def load_image(self, image_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load an image from the specified path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (image_array, metadata_dict)
            
        Raises:
            ImageLoadError: If the image cannot be loaded
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise ImageLoadError(f"Image file not found: {image_path}")
        
        if image_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ImageLoadError(f"Unsupported format: {image_path.suffix}")
        
        try:
            image_array, metadata = self._load_by_format(image_path)
            
            # Validate loaded image
            self._validate_image(image_array, image_path)
            
            # Cache metadata
            self.metadata_cache[str(image_path)] = metadata
            
            logger.info(f"Successfully loaded image: {image_path} "
                       f"Shape: {image_array.shape}, Dtype: {image_array.dtype}")
            
            return image_array, metadata
            
        except Exception as e:
            raise ImageLoadError(f"Failed to load image {image_path}: {str(e)}")
    
    def _load_by_format(self, image_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load image based on file format."""
        suffix = image_path.suffix.lower()
        
        if suffix in {'.tiff', '.tif'}:
            return self._load_tiff(image_path)
        elif suffix in {'.jpg', '.jpeg', '.png'}:
            return self._load_standard(image_path)
        elif suffix == '.czi':
            return self._load_czi(image_path)
        elif suffix == '.nd2':
            return self._load_nd2(image_path)
        else:
            raise ImageLoadError(f"Unsupported format: {suffix}")
    
    def _load_tiff(self, image_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load TIFF image using tifffile."""
        try:
            with tifffile.TiffFile(image_path) as tif:
                image_array = tif.asarray()
                
                # Extract metadata
                metadata = {
                    'format': 'TIFF',
                    'shape': image_array.shape,
                    'dtype': str(image_array.dtype),
                    'pages': len(tif.pages),
                    'is_multipage': len(tif.pages) > 1,
                    'file_size': image_path.stat().st_size,
                    'tags': {}
                }
                
                # Extract TIFF tags if available
                if tif.pages:
                    page = tif.pages[0]
                    for tag in page.tags:
                        try:
                            metadata['tags'][tag.name] = tag.value
                        except:
                            continue
                
                return image_array, metadata
                
        except Exception as e:
            # Fallback to standard loading
            logger.warning(f"tifffile failed, trying standard loader: {e}")
            return self._load_standard(image_path)
    
    def _load_standard(self, image_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load standard image formats (JPEG, PNG) using multiple backends."""
        try:
            # Try PIL first
            with Image.open(image_path) as img:
                image_array = np.array(img)
                
                metadata = {
                    'format': img.format,
                    'mode': img.mode,
                    'shape': image_array.shape,
                    'dtype': str(image_array.dtype),
                    'file_size': image_path.stat().st_size,
                    'info': dict(img.info) if img.info else {}
                }
                
                return image_array, metadata
                
        except Exception as e1:
            try:
                # Fallback to OpenCV
                image_array = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image_array is None:
                    raise ImageLoadError("OpenCV failed to load image")
                
                # Convert BGR to RGB for color images
                if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                
                metadata = {
                    'format': image_path.suffix.upper().lstrip('.'),
                    'shape': image_array.shape,
                    'dtype': str(image_array.dtype),
                    'file_size': image_path.stat().st_size,
                    'loader': 'OpenCV'
                }
                
                return image_array, metadata
                
            except Exception as e2:
                # Final fallback to scikit-image
                try:
                    image_array = io.imread(str(image_path))
                    metadata = {
                        'format': image_path.suffix.upper().lstrip('.'),
                        'shape': image_array.shape,
                        'dtype': str(image_array.dtype),
                        'file_size': image_path.stat().st_size,
                        'loader': 'scikit-image'
                    }
                    
                    return image_array, metadata
                    
                except Exception as e3:
                    raise ImageLoadError(f"All loaders failed: PIL({e1}), OpenCV({e2}), scikit-image({e3})")
    
    def _load_czi(self, image_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load CZI format (Zeiss microscopy)."""
        try:
            import czifile
            
            with czifile.CziFile(image_path) as czi:
                image_array = czi.asarray()
                
                metadata = {
                    'format': 'CZI',
                    'shape': image_array.shape,
                    'dtype': str(image_array.dtype),
                    'file_size': image_path.stat().st_size,
                    'dimensions': czi.axes,
                    'pixel_size': getattr(czi, 'pixel_size', None)
                }
                
                return image_array, metadata
                
        except ImportError:
            raise ImageLoadError("czifile package not installed. Install with: pip install czifile")
        except Exception as e:
            raise ImageLoadError(f"Failed to load CZI file: {e}")
    
    def _load_nd2(self, image_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load ND2 format (Nikon microscopy)."""
        try:
            from nd2reader import ND2Reader
            
            with ND2Reader(str(image_path)) as nd2:
                # Get the first frame or all data
                if len(nd2.sizes) > 2:  # Multi-dimensional
                    image_array = np.array(nd2)
                else:  # Single image
                    image_array = nd2[0] if len(nd2) > 0 else np.array(nd2)
                
                metadata = {
                    'format': 'ND2',
                    'shape': image_array.shape,
                    'dtype': str(image_array.dtype),
                    'file_size': image_path.stat().st_size,
                    'sizes': dict(nd2.sizes),
                    'pixel_microns': getattr(nd2, 'pixel_microns', None),
                    'frame_rate': getattr(nd2, 'frame_rate', None)
                }
                
                return image_array, metadata
                
        except ImportError:
            raise ImageLoadError("nd2reader package not installed. Install with: pip install nd2reader")
        except Exception as e:
            raise ImageLoadError(f"Failed to load ND2 file: {e}")
    
    def _validate_image(self, image_array: np.ndarray, image_path: Path):
        """Validate loaded image array."""
        if image_array is None:
            raise ImageLoadError(f"Loaded image is None: {image_path}")
        
        if image_array.size == 0:
            raise ImageLoadError(f"Loaded image is empty: {image_path}")
        
        if len(image_array.shape) < 2:
            raise ImageLoadError(f"Image must be at least 2D: {image_path}")
        
        if len(image_array.shape) > 4:
            logger.warning(f"Image has more than 4 dimensions: {image_path} - {image_array.shape}")
    
    def batch_load(self, image_paths: List[Union[str, Path]]) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Load multiple images in batch.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of (image_array, metadata) tuples
        """
        results = []
        failed_loads = []
        
        for path in image_paths:
            try:
                result = self.load_image(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                failed_loads.append((path, str(e)))
                results.append((None, {}))
        
        if failed_loads:
            logger.warning(f"Failed to load {len(failed_loads)} images out of {len(image_paths)}")
        
        return results
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return sorted(list(self.SUPPORTED_FORMATS))
    
    def is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """Check if file format is supported."""
        return Path(file_path).suffix.lower() in self.SUPPORTED_FORMATS


# Utility functions
def load_image(image_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to load a single image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (image_array, metadata_dict)
    """
    loader = ImageLoader()
    return loader.load_image(image_path)


def batch_load_images(image_paths: List[Union[str, Path]]) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Convenience function to load multiple images.
    
    Args:
        image_paths: List of paths to image files
        
    Returns:
        List of (image_array, metadata) tuples
    """
    loader = ImageLoader()
    return loader.batch_load(image_paths)


def get_image_info(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get image information without loading the full image data.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with image information
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise ImageLoadError(f"Image file not found: {image_path}")
    
    info = {
        'path': str(image_path),
        'filename': image_path.name,
        'format': image_path.suffix.lower(),
        'file_size': image_path.stat().st_size,
        'supported': image_path.suffix.lower() in ImageLoader.SUPPORTED_FORMATS
    }
    
    return info