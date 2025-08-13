"""
Image Quality Assessment Module

This module provides comprehensive quality assessment metrics for organoid microscopy images,
including technical quality measures and biological relevance indicators.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import cv2
from skimage import (
    filters, measure, morphology, feature, exposure,
    segmentation, restoration
)
from scipy import ndimage, stats
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Container for image quality assessment results."""
    overall_score: float  # 0-100 scale
    technical_metrics: Dict[str, float]
    biological_metrics: Dict[str, float]
    recommendations: List[str]
    quality_issues: List[str]


class ImageQualityAssessment:
    """
    Comprehensive image quality assessment for organoid microscopy images.
    
    Evaluates both technical quality (focus, noise, contrast) and biological
    relevance (organoid presence, structure quality) of images.
    """
    
    def __init__(self):
        """Initialize the quality assessment system."""
        self.quality_thresholds = {
            'min_contrast': 0.1,
            'min_sharpness': 0.02,
            'max_noise_level': 0.3,
            'min_snr': 3.0,
            'min_organoid_area_ratio': 0.01,
            'max_saturation_ratio': 0.05
        }
    
    def assess_quality(self, image: np.ndarray, 
                      metadata: Optional[Dict] = None) -> QualityMetrics:
        """
        Perform comprehensive quality assessment of an image.
        
        Args:
            image: Input image array
            metadata: Optional metadata about the image
            
        Returns:
            QualityMetrics with assessment results
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is None or empty")
        
        logger.info("Starting image quality assessment")
        
        # Calculate technical metrics
        technical_metrics = self._assess_technical_quality(image)
        
        # Calculate biological metrics
        biological_metrics = self._assess_biological_quality(image)
        
        # Generate overall score
        overall_score = self._calculate_overall_score(technical_metrics, biological_metrics)
        
        # Generate recommendations and identify issues
        recommendations = self._generate_recommendations(technical_metrics, biological_metrics)
        quality_issues = self._identify_quality_issues(technical_metrics, biological_metrics)
        
        result = QualityMetrics(
            overall_score=overall_score,
            technical_metrics=technical_metrics,
            biological_metrics=biological_metrics,
            recommendations=recommendations,
            quality_issues=quality_issues
        )
        
        logger.info(f"Quality assessment completed. Overall score: {overall_score:.1f}/100")
        return result
    
    def _assess_technical_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Assess technical image quality parameters."""
        metrics = {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray = gray.astype(np.float64) / 255.0
        else:
            gray = image.astype(np.float64)
            if gray.max() > 1.0:
                gray = gray / gray.max()
        
        # Sharpness (focus quality)
        metrics['sharpness'] = self._calculate_sharpness(gray)
        
        # Contrast
        metrics['contrast'] = self._calculate_contrast(gray)
        
        # Signal-to-noise ratio
        metrics['snr'] = self._calculate_snr(gray)
        
        # Noise level estimation
        metrics['noise_level'] = self._estimate_noise_level(gray)
        
        # Brightness and exposure
        metrics['brightness'] = np.mean(gray)
        metrics['exposure_quality'] = self._assess_exposure(gray)
        
        # Saturation (for color images)
        if len(image.shape) == 3:
            metrics['saturation_ratio'] = self._calculate_saturation_ratio(image)
        else:
            metrics['saturation_ratio'] = 0.0
        
        # Blur detection
        metrics['blur_score'] = self._detect_blur(gray)
        
        # Uniformity of illumination
        metrics['illumination_uniformity'] = self._assess_illumination_uniformity(gray)
        
        return metrics
    
    def _assess_biological_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Assess biological relevance and organoid-specific quality."""
        metrics = {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray = gray.astype(np.float64) / 255.0
        else:
            gray = image.astype(np.float64)
            if gray.max() > 1.0:
                gray = gray / gray.max()
        
        # Organoid detection and coverage
        organoid_mask = self._detect_potential_organoids(gray)
        metrics['organoid_coverage'] = np.sum(organoid_mask) / organoid_mask.size
        
        # Structure quality
        metrics['structure_clarity'] = self._assess_structure_clarity(gray)
        
        # Edge definition quality
        metrics['edge_definition'] = self._assess_edge_definition(gray)
        
        # Texture richness (indicates cellular detail)
        metrics['texture_richness'] = self._calculate_texture_richness(gray)
        
        # Background uniformity
        metrics['background_quality'] = self._assess_background_quality(gray, organoid_mask)
        
        # Organoid shape quality
        if np.sum(organoid_mask) > 0:
            metrics['shape_quality'] = self._assess_organoid_shapes(organoid_mask)
        else:
            metrics['shape_quality'] = 0.0
        
        return metrics
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        sharpness = laplacian.var()
        return float(sharpness)
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate RMS contrast."""
        mean_intensity = np.mean(image)
        contrast = np.sqrt(np.mean((image - mean_intensity) ** 2))
        return float(contrast)
    
    def _calculate_snr(self, image: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        # Use Otsu's method to separate signal from background
        threshold = filters.threshold_otsu(image)
        signal_mask = image > threshold
        background_mask = ~signal_mask
        
        if np.sum(signal_mask) == 0 or np.sum(background_mask) == 0:
            return 1.0
        
        signal_mean = np.mean(image[signal_mask])
        noise_std = np.std(image[background_mask])
        
        if noise_std > 0:
            snr = signal_mean / noise_std
        else:
            snr = float('inf')
        
        return float(snr)
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level using robust statistics."""
        # Use median absolute deviation for robust noise estimation
        median_filtered = ndimage.median_filter(image, size=3)
        noise = np.abs(image - median_filtered)
        noise_level = np.median(noise) * 1.4826  # Scale factor for normal distribution
        return float(noise_level)
    
    def _assess_exposure(self, image: np.ndarray) -> float:
        """Assess exposure quality (0=poor, 1=good)."""
        hist, _ = np.histogram(image, bins=256, range=(0, 1))
        
        # Check for clipping
        underexposed = hist[0] / image.size  # Pixels at 0
        overexposed = hist[-1] / image.size  # Pixels at 1
        
        # Good exposure should have minimal clipping and good distribution
        clipping_penalty = (underexposed + overexposed) * 2
        
        # Check distribution spread
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        
        # Optimal exposure: mean around 0.3-0.7, good std
        exposure_score = 1.0 - clipping_penalty
        
        if mean_intensity < 0.1 or mean_intensity > 0.9:
            exposure_score *= 0.5  # Penalty for extreme brightness
        
        if std_intensity < 0.1:
            exposure_score *= 0.7  # Penalty for low contrast
        
        return max(0.0, float(exposure_score))
    
    def _calculate_saturation_ratio(self, image: np.ndarray) -> float:
        """Calculate ratio of saturated pixels in color image."""
        if len(image.shape) != 3:
            return 0.0
        
        # Check for saturation in any channel
        saturated = np.any(image >= 0.95 * image.max(), axis=2)
        saturation_ratio = np.sum(saturated) / image.size
        
        return float(saturation_ratio)
    
    def _detect_blur(self, image: np.ndarray) -> float:
        """Detect blur using gradient magnitude."""
        # Calculate gradient
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Blur score based on gradient magnitude distribution
        blur_score = np.mean(gradient_magnitude)
        return float(blur_score)
    
    def _assess_illumination_uniformity(self, image: np.ndarray) -> float:
        """Assess uniformity of illumination across the image."""
        # Divide image into regions and check intensity variation
        h, w = image.shape
        regions = []
        
        for i in range(0, h, h//4):
            for j in range(0, w, w//4):
                region = image[i:i+h//4, j:j+w//4]
                if region.size > 0:
                    regions.append(np.mean(region))
        
        if len(regions) == 0:
            return 1.0
        
        # Uniformity based on coefficient of variation
        mean_intensity = np.mean(regions)
        std_intensity = np.std(regions)
        
        if mean_intensity > 0:
            cv = std_intensity / mean_intensity
            uniformity = max(0.0, 1.0 - cv)
        else:
            uniformity = 0.0
        
        return float(uniformity)
    
    def _detect_potential_organoids(self, image: np.ndarray) -> np.ndarray:
        """Simple organoid detection for quality assessment."""
        # Apply Gaussian filter to reduce noise
        filtered = filters.gaussian(image, sigma=1.0)
        
        # Use multi-level Otsu thresholding
        thresholds = filters.threshold_multiotsu(filtered, classes=3)
        
        # Take middle threshold level (organoids are typically mid-intensity)
        organoid_mask = (filtered > thresholds[0]) & (filtered < thresholds[1])
        
        # Remove small objects
        organoid_mask = morphology.remove_small_objects(organoid_mask, min_size=100)
        
        # Fill holes
        organoid_mask = ndimage.binary_fill_holes(organoid_mask)
        
        return organoid_mask
    
    def _assess_structure_clarity(self, image: np.ndarray) -> float:
        """Assess clarity of cellular structures."""
        # Use local binary patterns for texture analysis
        try:
            lbp = feature.local_binary_pattern(image, P=8, R=1, method='uniform')
            structure_score = np.std(lbp) / (np.mean(lbp) + 1e-6)
        except:
            # Fallback to gradient-based measure
            grad = np.gradient(image)
            structure_score = np.mean(np.sqrt(grad[0]**2 + grad[1]**2))
        
        return float(min(1.0, structure_score))
    
    def _assess_edge_definition(self, image: np.ndarray) -> float:
        """Assess quality of edge definition in the image."""
        # Use Canny edge detection
        edges = feature.canny(image, sigma=1.0)
        edge_ratio = np.sum(edges) / edges.size
        
        # Quality based on edge density and connectivity
        edge_score = min(1.0, edge_ratio * 10)  # Scale factor
        
        return float(edge_score)
    
    def _calculate_texture_richness(self, image: np.ndarray) -> float:
        """Calculate texture richness indicating cellular detail."""
        # Use gray-level co-occurrence matrix properties
        try:
            # Quantize image for GLCM
            quantized = (image * 31).astype(int)
            glcm = feature.graycomatrix(quantized, [1], [0], levels=32, symmetric=True)
            
            # Calculate texture properties
            contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = feature.graycoprops(glcm, 'homogeneity')[0, 0]
            
            # Combine properties for richness score
            texture_score = (contrast + dissimilarity) * homogeneity
            
        except:
            # Fallback to standard deviation
            texture_score = np.std(image) * 2
        
        return float(min(1.0, texture_score))
    
    def _assess_background_quality(self, image: np.ndarray, organoid_mask: np.ndarray) -> float:
        """Assess background uniformity and cleanliness."""
        background_mask = ~organoid_mask
        
        if np.sum(background_mask) == 0:
            return 0.5
        
        background_pixels = image[background_mask]
        
        # Background should be uniform (low std) and clean (appropriate intensity)
        bg_mean = np.mean(background_pixels)
        bg_std = np.std(background_pixels)
        
        # Good background: low variation, appropriate intensity
        uniformity_score = max(0.0, 1.0 - bg_std * 5)  # Penalize high variation
        intensity_score = 1.0 - abs(bg_mean - 0.2) * 2  # Prefer dark background
        
        background_quality = (uniformity_score + max(0.0, intensity_score)) / 2
        
        return float(background_quality)
    
    def _assess_organoid_shapes(self, organoid_mask: np.ndarray) -> float:
        """Assess quality of detected organoid shapes."""
        labeled_mask = measure.label(organoid_mask)
        regions = measure.regionprops(labeled_mask)
        
        if len(regions) == 0:
            return 0.0
        
        shape_scores = []
        
        for region in regions:
            if region.area < 50:  # Skip very small regions
                continue
            
            # Assess circularity (organoids should be roughly circular)
            circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
            circularity_score = min(1.0, circularity)
            
            # Assess solidity (should be fairly solid)
            solidity_score = region.solidity
            
            # Combine scores
            shape_score = (circularity_score + solidity_score) / 2
            shape_scores.append(shape_score)
        
        if len(shape_scores) == 0:
            return 0.0
        
        return float(np.mean(shape_scores))
    
    def _calculate_overall_score(self, technical: Dict[str, float], 
                                biological: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics."""
        # Define weights for different aspects
        tech_weights = {
            'sharpness': 0.25,
            'contrast': 0.20,
            'snr': 0.15,
            'exposure_quality': 0.15,
            'illumination_uniformity': 0.10,
            'blur_score': 0.15
        }
        
        bio_weights = {
            'organoid_coverage': 0.25,
            'structure_clarity': 0.20,
            'edge_definition': 0.15,
            'texture_richness': 0.15,
            'background_quality': 0.15,
            'shape_quality': 0.10
        }
        
        # Calculate weighted technical score
        tech_score = 0.0
        tech_total_weight = 0.0
        
        for metric, weight in tech_weights.items():
            if metric in technical:
                value = technical[metric]
                
                # Normalize different metrics to 0-1 range
                if metric == 'sharpness':
                    normalized = min(1.0, value / 0.05)  # Good sharpness > 0.05
                elif metric == 'contrast':
                    normalized = min(1.0, value / 0.3)   # Good contrast > 0.3
                elif metric == 'snr':
                    normalized = min(1.0, value / 10.0)  # Good SNR > 10
                elif metric == 'blur_score':
                    normalized = min(1.0, value / 0.1)   # Good blur score > 0.1
                else:
                    normalized = value  # Already 0-1
                
                tech_score += normalized * weight
                tech_total_weight += weight
        
        if tech_total_weight > 0:
            tech_score /= tech_total_weight
        
        # Calculate weighted biological score
        bio_score = 0.0
        bio_total_weight = 0.0
        
        for metric, weight in bio_weights.items():
            if metric in biological:
                value = biological[metric]
                
                # Normalize metrics
                if metric == 'organoid_coverage':
                    normalized = min(1.0, value / 0.3)   # Good coverage > 0.3
                else:
                    normalized = value  # Already 0-1
                
                bio_score += normalized * weight
                bio_total_weight += weight
        
        if bio_total_weight > 0:
            bio_score /= bio_total_weight
        
        # Combine technical and biological scores
        overall_score = (tech_score * 0.6 + bio_score * 0.4) * 100
        
        return float(max(0.0, min(100.0, overall_score)))
    
    def _generate_recommendations(self, technical: Dict[str, float], 
                                 biological: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations based on metrics."""
        recommendations = []
        
        # Technical recommendations
        if technical.get('sharpness', 0) < 0.02:
            recommendations.append("Image appears blurry - check focus and camera stability")
        
        if technical.get('contrast', 0) < 0.1:
            recommendations.append("Low contrast - adjust lighting or camera settings")
        
        if technical.get('snr', 0) < 3.0:
            recommendations.append("High noise level - reduce ISO or increase exposure time")
        
        if technical.get('exposure_quality', 0) < 0.5:
            recommendations.append("Poor exposure - adjust brightness and avoid clipping")
        
        if technical.get('illumination_uniformity', 0) < 0.5:
            recommendations.append("Uneven illumination - check lighting setup")
        
        # Biological recommendations
        if biological.get('organoid_coverage', 0) < 0.05:
            recommendations.append("Low organoid content - ensure proper field of view")
        
        if biological.get('structure_clarity', 0) < 0.3:
            recommendations.append("Poor structure definition - improve contrast or focus")
        
        if biological.get('background_quality', 0) < 0.5:
            recommendations.append("Noisy background - improve sample preparation or imaging")
        
        if len(recommendations) == 0:
            recommendations.append("Image quality is acceptable for analysis")
        
        return recommendations
    
    def _identify_quality_issues(self, technical: Dict[str, float], 
                                biological: Dict[str, float]) -> List[str]:
        """Identify specific quality issues that may affect analysis."""
        issues = []
        
        # Critical technical issues
        if technical.get('sharpness', 0) < 0.01:
            issues.append("Severe blur - image may not be suitable for analysis")
        
        if technical.get('saturation_ratio', 0) > 0.1:
            issues.append("Significant pixel saturation detected")
        
        if technical.get('noise_level', 0) > 0.5:
            issues.append("Very high noise level")
        
        # Critical biological issues
        if biological.get('organoid_coverage', 0) < 0.01:
            issues.append("No clear organoid structures detected")
        
        if biological.get('background_quality', 0) < 0.2:
            issues.append("Very poor background quality")
        
        return issues


# Utility functions
def assess_image_quality(image: np.ndarray, 
                        metadata: Optional[Dict] = None) -> QualityMetrics:
    """
    Convenience function for assessing image quality.
    
    Args:
        image: Input image array
        metadata: Optional image metadata
        
    Returns:
        QualityMetrics with assessment results
    """
    assessor = ImageQualityAssessment()
    return assessor.assess_quality(image, metadata)


def batch_assess_quality(images: List[np.ndarray], 
                        metadata_list: Optional[List[Dict]] = None) -> List[QualityMetrics]:
    """
    Assess quality of multiple images in batch.
    
    Args:
        images: List of image arrays
        metadata_list: Optional list of metadata dictionaries
        
    Returns:
        List of QualityMetrics objects
    """
    assessor = ImageQualityAssessment()
    results = []
    
    if metadata_list is None:
        metadata_list = [None] * len(images)
    
    for i, (image, metadata) in enumerate(zip(images, metadata_list)):
        try:
            result = assessor.assess_quality(image, metadata)
            results.append(result)
            logger.debug(f"Assessed quality for image {i+1}/{len(images)}")
        except Exception as e:
            logger.error(f"Failed to assess quality for image {i+1}: {e}")
            results.append(None)
    
    return results