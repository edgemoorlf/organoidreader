"""
Image Preprocessing Module

This module provides comprehensive image preprocessing functionality for organoid analysis,
including noise reduction, contrast enhancement, normalization, and standardization.
"""

import logging
import numpy as np
from typing import Tuple, Optional, Union, Dict, Any
from pathlib import Path
import cv2
from skimage import (
    filters, morphology, exposure, transform, util, 
    restoration, measure, feature
)
from skimage.util import img_as_float, img_as_ubyte
from scipy import ndimage
from dataclasses import dataclass

from organoidreader.config.config_manager import ProcessingConfig

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Container for preprocessing results."""
    processed_image: np.ndarray
    original_shape: Tuple[int, ...]
    preprocessing_steps: list
    quality_metrics: Dict[str, float]
    metadata: Dict[str, Any]


class ImagePreprocessor:
    """
    Comprehensive image preprocessor for organoid microscopy images.
    
    Provides methods for noise reduction, contrast enhancement, normalization,
    and standardization of images for downstream analysis.
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Processing configuration. If None, uses defaults.
        """
        self.config = config if config is not None else ProcessingConfig()
        self.preprocessing_history = []
    
    def preprocess(self, 
                  image: np.ndarray, 
                  target_size: Optional[Tuple[int, int]] = None,
                  normalize: Optional[bool] = None,
                  enhance_contrast: Optional[bool] = None,
                  denoise: Optional[bool] = None) -> PreprocessingResult:
        """
        Apply complete preprocessing pipeline to an image.
        
        Args:
            image: Input image array
            target_size: Target size for resizing. If None, uses config default
            normalize: Whether to normalize. If None, uses config default
            enhance_contrast: Whether to enhance contrast. If None, uses config default
            denoise: Whether to apply denoising. If None, uses config default
            
        Returns:
            PreprocessingResult with processed image and metadata
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is None or empty")
        
        # Use config defaults if not specified
        target_size = target_size if target_size is not None else self.config.target_size
        normalize = normalize if normalize is not None else self.config.normalize
        enhance_contrast = enhance_contrast if enhance_contrast is not None else self.config.enhance_contrast
        denoise = denoise if denoise is not None else self.config.denoise
        
        original_shape = image.shape
        processed_image = image.copy()
        processing_steps = []
        quality_metrics = {}
        
        logger.info(f"Starting preprocessing: {original_shape} -> {target_size}")
        
        try:
            # Convert to float for processing
            if processed_image.dtype != np.float64:
                processed_image = img_as_float(processed_image)
                processing_steps.append("convert_to_float")
            
            # Calculate initial quality metrics
            quality_metrics['initial_snr'] = self._calculate_snr(processed_image)
            quality_metrics['initial_contrast'] = self._calculate_contrast(processed_image)
            
            # Denoising
            if denoise:
                processed_image = self._denoise_image(processed_image)
                processing_steps.append("denoise")
                quality_metrics['post_denoise_snr'] = self._calculate_snr(processed_image)
            
            # Contrast enhancement
            if enhance_contrast:
                processed_image = self._enhance_contrast(processed_image)
                processing_steps.append("enhance_contrast")
                quality_metrics['post_contrast_contrast'] = self._calculate_contrast(processed_image)
            
            # Normalization
            if normalize:
                processed_image = self._normalize_image(processed_image)
                processing_steps.append("normalize")
            
            # Resizing
            if target_size != original_shape[:2]:
                processed_image = self._resize_image(processed_image, target_size)
                processing_steps.append(f"resize_to_{target_size}")
            
            # Final quality metrics
            quality_metrics['final_snr'] = self._calculate_snr(processed_image)
            quality_metrics['final_contrast'] = self._calculate_contrast(processed_image)
            quality_metrics['sharpness'] = self._calculate_sharpness(processed_image)
            
            result = PreprocessingResult(
                processed_image=processed_image,
                original_shape=original_shape,
                preprocessing_steps=processing_steps,
                quality_metrics=quality_metrics,
                metadata={
                    'target_size': target_size,
                    'normalization_applied': normalize,
                    'contrast_enhancement_applied': enhance_contrast,
                    'denoising_applied': denoise
                }
            )
            
            logger.info(f"Preprocessing completed: {len(processing_steps)} steps applied")
            return result
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising to the image using multiple techniques.
        
        Args:
            image: Input image array
            
        Returns:
            Denoised image
        """
        # Gaussian denoising for general noise
        if self.config.gaussian_blur_sigma > 0:
            denoised = filters.gaussian(image, sigma=self.config.gaussian_blur_sigma)
        else:
            denoised = image.copy()
        
        # Non-local means denoising for better structure preservation
        if len(image.shape) == 2:  # Grayscale
            denoised = restoration.denoise_nl_means(
                denoised, 
                patch_size=7, 
                patch_distance=11, 
                h=0.1 * np.std(denoised)
            )
        elif len(image.shape) == 3:  # Color
            denoised = restoration.denoise_nl_means(
                denoised, 
                patch_size=7, 
                patch_distance=11, 
                h=0.1 * np.std(denoised),
                multichannel=True
            )
        
        # Bilateral filtering for edge preservation
        if len(image.shape) == 2:
            denoised = restoration.denoise_bilateral(
                denoised,
                sigma_color=0.1,
                sigma_spatial=1.0
            )
        
        logger.debug("Applied denoising: Gaussian + NL-means + Bilateral")
        return denoised
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using adaptive histogram equalization.
        
        Args:
            image: Input image array
            
        Returns:
            Contrast-enhanced image
        """
        if len(image.shape) == 2:  # Grayscale
            enhanced = exposure.equalize_adapthist(
                image,
                clip_limit=self.config.clahe_clip_limit / 100.0,
                kernel_size=self.config.clahe_tile_grid_size
            )
        elif len(image.shape) == 3:  # Color
            # Apply CLAHE to each channel
            enhanced = np.zeros_like(image)
            for i in range(image.shape[2]):
                enhanced[:, :, i] = exposure.equalize_adapthist(
                    image[:, :, i],
                    clip_limit=self.config.clahe_clip_limit / 100.0,
                    kernel_size=self.config.clahe_tile_grid_size
                )
        else:
            enhanced = image.copy()
        
        logger.debug(f"Applied CLAHE with clip_limit={self.config.clahe_clip_limit}")
        return enhanced
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image intensity values.
        
        Args:
            image: Input image array
            
        Returns:
            Normalized image
        """
        # Rescale to [0, 1] range
        normalized = exposure.rescale_intensity(image, out_range=(0, 1))
        
        # Z-score normalization for better model performance
        mean_val = np.mean(normalized)
        std_val = np.std(normalized)
        
        if std_val > 0:
            normalized = (normalized - mean_val) / std_val
            # Rescale back to [0, 1] range
            normalized = exposure.rescale_intensity(normalized, out_range=(0, 1))
        
        logger.debug("Applied intensity normalization")
        return normalized
    
    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target dimensions while preserving aspect ratio.
        
        Args:
            image: Input image array
            target_size: Target (height, width) dimensions
            
        Returns:
            Resized image
        """
        if len(image.shape) == 2:
            resized = transform.resize(
                image, 
                target_size, 
                mode='reflect', 
                anti_aliasing=True,
                preserve_range=True
            )
        elif len(image.shape) == 3:
            resized = transform.resize(
                image, 
                target_size + (image.shape[2],), 
                mode='reflect', 
                anti_aliasing=True,
                preserve_range=True
            )
        else:
            raise ValueError(f"Unsupported image dimensions: {image.shape}")
        
        logger.debug(f"Resized image: {image.shape} -> {resized.shape}")
        return resized
    
    def _calculate_snr(self, image: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio."""
        if len(image.shape) > 2:
            image = np.mean(image, axis=-1)  # Convert to grayscale
        
        # Calculate SNR as mean/std
        mean_signal = np.mean(image)
        noise_std = np.std(image)
        
        if noise_std > 0:
            snr = mean_signal / noise_std
        else:
            snr = float('inf')
        
        return float(snr)
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast using RMS contrast."""
        if len(image.shape) > 2:
            image = np.mean(image, axis=-1)  # Convert to grayscale
        
        mean_intensity = np.mean(image)
        contrast = np.sqrt(np.mean((image - mean_intensity) ** 2))
        
        return float(contrast)
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using gradient magnitude."""
        if len(image.shape) > 2:
            image = np.mean(image, axis=-1)  # Convert to grayscale
        
        # Calculate gradient magnitude
        grad_x = np.gradient(image, axis=1)
        grad_y = np.gradient(image, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Sharpness as mean gradient magnitude
        sharpness = np.mean(gradient_magnitude)
        
        return float(sharpness)
    
    def create_preprocessing_report(self, result: PreprocessingResult) -> str:
        """
        Create a detailed preprocessing report.
        
        Args:
            result: PreprocessingResult object
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=== IMAGE PREPROCESSING REPORT ===")
        report.append(f"Original shape: {result.original_shape}")
        report.append(f"Final shape: {result.processed_image.shape}")
        report.append("")
        
        report.append("Processing steps applied:")
        for i, step in enumerate(result.preprocessing_steps, 1):
            report.append(f"  {i}. {step}")
        report.append("")
        
        report.append("Quality metrics:")
        for metric, value in result.quality_metrics.items():
            report.append(f"  {metric}: {value:.4f}")
        report.append("")
        
        report.append("Configuration used:")
        for key, value in result.metadata.items():
            report.append(f"  {key}: {value}")
        
        return "\n".join(report)


# Utility functions for common preprocessing tasks
def preprocess_image(image: np.ndarray, 
                    config: Optional[ProcessingConfig] = None,
                    **kwargs) -> PreprocessingResult:
    """
    Convenience function for preprocessing a single image.
    
    Args:
        image: Input image array
        config: Processing configuration
        **kwargs: Additional preprocessing parameters
        
    Returns:
        PreprocessingResult with processed image and metadata
    """
    preprocessor = ImagePreprocessor(config)
    return preprocessor.preprocess(image, **kwargs)


def preprocess_batch(images: list, 
                    config: Optional[ProcessingConfig] = None,
                    **kwargs) -> list:
    """
    Convenience function for preprocessing multiple images.
    
    Args:
        images: List of image arrays
        config: Processing configuration
        **kwargs: Additional preprocessing parameters
        
    Returns:
        List of PreprocessingResult objects
    """
    preprocessor = ImagePreprocessor(config)
    results = []
    
    for i, image in enumerate(images):
        try:
            result = preprocessor.preprocess(image, **kwargs)
            results.append(result)
            logger.debug(f"Preprocessed image {i+1}/{len(images)}")
        except Exception as e:
            logger.error(f"Failed to preprocess image {i+1}: {e}")
            results.append(None)
    
    return results


def standardize_for_model(image: np.ndarray, 
                         target_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Standardize image for deep learning model input.
    
    Args:
        image: Input image array
        target_size: Target dimensions for model input
        
    Returns:
        Standardized image array
    """
    config = ProcessingConfig(
        target_size=target_size,
        normalize=True,
        enhance_contrast=True,
        denoise=True
    )
    
    result = preprocess_image(image, config)
    
    # Ensure proper format for model input
    processed = result.processed_image
    
    # Convert to proper data type
    if processed.dtype != np.float32:
        processed = processed.astype(np.float32)
    
    # Ensure proper channel dimension for models
    if len(processed.shape) == 2:  # Grayscale
        processed = np.expand_dims(processed, axis=-1)
    
    return processed