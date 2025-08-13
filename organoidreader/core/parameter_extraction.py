"""
Parameter Extraction Module

This module provides comprehensive parameter extraction and analysis
for segmented organoids, including morphological, intensity, and texture features.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import cv2
from skimage import measure, feature, filters, morphology
from scipy import ndimage, spatial
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OrganoidParameters:
    """Container for organoid parameter measurements."""
    label: int
    morphological: Dict[str, float]
    intensity: Dict[str, float]
    texture: Dict[str, float]
    spatial: Dict[str, float]
    metadata: Dict[str, Any]


class ParameterExtractor:
    """
    Comprehensive parameter extraction for segmented organoids.
    
    Extracts morphological, intensity, texture, and spatial features
    from segmented organoid regions for quantitative analysis.
    """
    
    def __init__(self, pixel_size_microns: Optional[float] = None):
        """
        Initialize parameter extractor.
        
        Args:
            pixel_size_microns: Size of one pixel in microns for real-world measurements
        """
        self.pixel_size_microns = pixel_size_microns
        self.conversion_factor = pixel_size_microns ** 2 if pixel_size_microns else 1.0
    
    def extract_parameters(self, 
                          image: np.ndarray,
                          labeled_mask: np.ndarray,
                          extract_texture: bool = True,
                          extract_intensity: bool = True) -> List[OrganoidParameters]:
        """
        Extract comprehensive parameters for all organoids in image.
        
        Args:
            image: Original image
            labeled_mask: Labeled segmentation mask
            extract_texture: Whether to calculate texture features
            extract_intensity: Whether to calculate intensity features
            
        Returns:
            List of OrganoidParameters for each organoid
        """
        if image is None or labeled_mask is None:
            raise ValueError("Image and labeled mask must not be None")
        
        # Ensure image is grayscale for consistent processing
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()
        
        # Normalize image to 0-1 range
        if gray_image.max() > 1.0:
            gray_image = gray_image.astype(np.float64) / gray_image.max()
        
        parameters_list = []
        regions = measure.regionprops(labeled_mask, intensity_image=gray_image)
        
        logger.info(f"Extracting parameters for {len(regions)} organoids")
        
        for region in regions:
            try:
                params = self._extract_single_organoid_parameters(
                    gray_image, region, extract_texture, extract_intensity
                )
                parameters_list.append(params)
            except Exception as e:
                logger.warning(f"Failed to extract parameters for organoid {region.label}: {e}")
                continue
        
        logger.info(f"Successfully extracted parameters for {len(parameters_list)} organoids")
        return parameters_list
    
    def _extract_single_organoid_parameters(self, 
                                           image: np.ndarray, 
                                           region: Any,
                                           extract_texture: bool,
                                           extract_intensity: bool) -> OrganoidParameters:
        """Extract parameters for a single organoid region."""
        
        # Morphological parameters
        morphological = self._extract_morphological_features(region)
        
        # Intensity parameters
        if extract_intensity:
            intensity = self._extract_intensity_features(image, region)
        else:
            intensity = {}
        
        # Texture parameters
        if extract_texture:
            texture = self._extract_texture_features(image, region)
        else:
            texture = {}
        
        # Spatial parameters
        spatial = self._extract_spatial_features(region)
        
        # Metadata
        metadata = {
            'pixel_size_microns': self.pixel_size_microns,
            'conversion_factor': self.conversion_factor,
            'has_texture_features': extract_texture,
            'has_intensity_features': extract_intensity
        }
        
        return OrganoidParameters(
            label=region.label,
            morphological=morphological,
            intensity=intensity,
            texture=texture,
            spatial=spatial,
            metadata=metadata
        )
    
    def _extract_morphological_features(self, region: Any) -> Dict[str, float]:
        """Extract morphological features from a region."""
        features = {}
        
        # Basic size measurements
        features['area_pixels'] = float(region.area)
        features['perimeter_pixels'] = float(region.perimeter)
        
        # Convert to real-world units if pixel size is known
        if self.pixel_size_microns:
            features['area_microns2'] = features['area_pixels'] * self.conversion_factor
            features['perimeter_microns'] = features['perimeter_pixels'] * self.pixel_size_microns
            features['equivalent_diameter_microns'] = np.sqrt(4 * features['area_microns2'] / np.pi)
        
        # Shape descriptors
        features['major_axis_length'] = float(region.major_axis_length)
        features['minor_axis_length'] = float(region.minor_axis_length)
        features['aspect_ratio'] = features['major_axis_length'] / max(features['minor_axis_length'], 1e-6)
        features['eccentricity'] = float(region.eccentricity)
        features['orientation'] = float(region.orientation)
        
        # Circularity and shape factors
        if region.perimeter > 0:
            features['circularity'] = 4 * np.pi * region.area / (region.perimeter ** 2)
        else:
            features['circularity'] = 0.0
        
        features['solidity'] = float(region.solidity)
        features['extent'] = float(region.extent)
        features['convex_area'] = float(region.convex_area)
        
        # Compactness
        if features['area_pixels'] > 0:
            features['compactness'] = features['perimeter_pixels'] ** 2 / (4 * np.pi * features['area_pixels'])
        else:
            features['compactness'] = 0.0
        
        # Roundness
        if features['major_axis_length'] > 0:
            features['roundness'] = 4 * features['area_pixels'] / (np.pi * features['major_axis_length'] ** 2)
        else:
            features['roundness'] = 0.0
        
        # Form factor
        if features['area_pixels'] > 0:
            features['form_factor'] = 4 * np.pi * features['area_pixels'] / (features['perimeter_pixels'] ** 2)
        else:
            features['form_factor'] = 0.0
        
        return features
    
    def _extract_intensity_features(self, image: np.ndarray, region: Any) -> Dict[str, float]:
        """Extract intensity-based features from a region."""
        features = {}
        
        # Get region pixels
        coords = region.coords
        region_pixels = image[coords[:, 0], coords[:, 1]]
        
        # Basic intensity statistics
        features['mean_intensity'] = float(np.mean(region_pixels))
        features['std_intensity'] = float(np.std(region_pixels))
        features['min_intensity'] = float(np.min(region_pixels))
        features['max_intensity'] = float(np.max(region_pixels))
        features['intensity_range'] = features['max_intensity'] - features['min_intensity']
        
        # Percentiles
        features['intensity_p25'] = float(np.percentile(region_pixels, 25))
        features['intensity_p50'] = float(np.percentile(region_pixels, 50))  # Median
        features['intensity_p75'] = float(np.percentile(region_pixels, 75))
        features['intensity_iqr'] = features['intensity_p75'] - features['intensity_p25']
        
        # Coefficient of variation
        if features['mean_intensity'] > 0:
            features['intensity_cv'] = features['std_intensity'] / features['mean_intensity']
        else:
            features['intensity_cv'] = 0.0
        
        # Skewness and kurtosis
        features['intensity_skewness'] = float(self._calculate_skewness(region_pixels))
        features['intensity_kurtosis'] = float(self._calculate_kurtosis(region_pixels))
        
        # Edge intensity analysis
        edge_intensities = self._get_edge_intensities(image, region)
        if len(edge_intensities) > 0:
            features['edge_mean_intensity'] = float(np.mean(edge_intensities))
            features['edge_std_intensity'] = float(np.std(edge_intensities))
            features['edge_contrast'] = features['mean_intensity'] - features['edge_mean_intensity']
        else:
            features['edge_mean_intensity'] = 0.0
            features['edge_std_intensity'] = 0.0
            features['edge_contrast'] = 0.0
        
        return features
    
    def _extract_texture_features(self, image: np.ndarray, region: Any) -> Dict[str, float]:
        """Extract texture features from a region."""
        features = {}
        
        try:
            # Get bounding box region
            minr, minc, maxr, maxc = region.bbox
            region_image = image[minr:maxr, minc:maxc].copy()
            region_mask = region.image
            
            # Apply mask to get only organoid pixels
            masked_region = region_image * region_mask
            
            # Gray-Level Co-occurrence Matrix (GLCM) features
            glcm_features = self._calculate_glcm_features(masked_region, region_mask)
            features.update(glcm_features)
            
            # Local Binary Pattern (LBP) features
            lbp_features = self._calculate_lbp_features(masked_region, region_mask)
            features.update(lbp_features)
            
            # Gabor filter responses
            gabor_features = self._calculate_gabor_features(masked_region, region_mask)
            features.update(gabor_features)
            
        except Exception as e:
            logger.warning(f"Failed to extract texture features for region {region.label}: {e}")
            # Return default values
            features.update({
                'glcm_contrast': 0.0, 'glcm_dissimilarity': 0.0, 'glcm_homogeneity': 0.0,
                'glcm_energy': 0.0, 'glcm_correlation': 0.0,
                'lbp_uniformity': 0.0, 'lbp_entropy': 0.0,
                'gabor_mean': 0.0, 'gabor_std': 0.0
            })
        
        return features
    
    def _extract_spatial_features(self, region: Any) -> Dict[str, float]:
        """Extract spatial features from a region."""
        features = {}
        
        # Centroid coordinates
        features['centroid_row'] = float(region.centroid[0])
        features['centroid_col'] = float(region.centroid[1])
        
        # Bounding box properties
        minr, minc, maxr, maxc = region.bbox
        features['bbox_area'] = float((maxr - minr) * (maxc - minc))
        features['bbox_width'] = float(maxc - minc)
        features['bbox_height'] = float(maxr - minr)
        features['bbox_aspect_ratio'] = features['bbox_width'] / max(features['bbox_height'], 1e-6)
        
        # Moments
        features['hu_moment_1'] = float(region.moments_hu[0])
        features['hu_moment_2'] = float(region.moments_hu[1])
        features['hu_moment_3'] = float(region.moments_hu[2])
        features['hu_moment_4'] = float(region.moments_hu[3])
        features['hu_moment_5'] = float(region.moments_hu[4])
        features['hu_moment_6'] = float(region.moments_hu[5])
        features['hu_moment_7'] = float(region.moments_hu[6])
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        if len(data) < 3:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return 0.0
        
        skewness = np.mean(((data - mean_val) / std_val) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        if len(data) < 4:
            return 0.0
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return 0.0
        
        kurtosis = np.mean(((data - mean_val) / std_val) ** 4) - 3
        return kurtosis
    
    def _get_edge_intensities(self, image: np.ndarray, region: Any) -> np.ndarray:
        """Get intensity values along the edge of a region."""
        try:
            # Create edge mask by dilating and subtracting original
            coords = region.coords
            mask = np.zeros(image.shape, dtype=bool)
            mask[coords[:, 0], coords[:, 1]] = True
            
            dilated = morphology.binary_dilation(mask, morphology.disk(1))
            edge_mask = dilated & ~mask
            
            edge_coords = np.where(edge_mask)
            if len(edge_coords[0]) > 0:
                return image[edge_coords]
            else:
                return np.array([])
        except:
            return np.array([])
    
    def _calculate_glcm_features(self, region_image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Calculate Gray-Level Co-occurrence Matrix features."""
        try:
            # Quantize image to reduce computation
            quantized = (region_image * mask * 31).astype(int)
            
            # Calculate GLCM for different angles
            glcm = feature.graycomatrix(quantized, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                                      levels=32, symmetric=True, normed=True)
            
            # Calculate properties
            contrast = np.mean(feature.graycoprops(glcm, 'contrast'))
            dissimilarity = np.mean(feature.graycoprops(glcm, 'dissimilarity'))
            homogeneity = np.mean(feature.graycoprops(glcm, 'homogeneity'))
            energy = np.mean(feature.graycoprops(glcm, 'energy'))
            correlation = np.mean(feature.graycoprops(glcm, 'correlation'))
            
            return {
                'glcm_contrast': float(contrast),
                'glcm_dissimilarity': float(dissimilarity),
                'glcm_homogeneity': float(homogeneity),
                'glcm_energy': float(energy),
                'glcm_correlation': float(correlation)
            }
        except:
            return {
                'glcm_contrast': 0.0,
                'glcm_dissimilarity': 0.0,
                'glcm_homogeneity': 0.0,
                'glcm_energy': 0.0,
                'glcm_correlation': 0.0
            }
    
    def _calculate_lbp_features(self, region_image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Calculate Local Binary Pattern features."""
        try:
            # Calculate LBP
            lbp = feature.local_binary_pattern(region_image, P=8, R=1, method='uniform')
            lbp_masked = lbp * mask
            
            # Get LBP values for masked region
            lbp_values = lbp_masked[mask > 0]
            
            if len(lbp_values) > 0:
                # Calculate histogram
                hist, _ = np.histogram(lbp_values, bins=np.arange(11), density=True)
                
                # Uniformity
                uniformity = np.sum(hist ** 2)
                
                # Entropy
                hist_nonzero = hist[hist > 0]
                entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))
            else:
                uniformity = 0.0
                entropy = 0.0
            
            return {
                'lbp_uniformity': float(uniformity),
                'lbp_entropy': float(entropy)
            }
        except:
            return {
                'lbp_uniformity': 0.0,
                'lbp_entropy': 0.0
            }
    
    def _calculate_gabor_features(self, region_image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Calculate Gabor filter response features."""
        try:
            # Apply Gabor filter
            gabor_real, gabor_imag = filters.gabor(region_image, frequency=0.1, theta=0)
            gabor_magnitude = np.sqrt(gabor_real**2 + gabor_imag**2)
            
            # Get responses for masked region
            gabor_values = gabor_magnitude * mask
            gabor_masked_values = gabor_values[mask > 0]
            
            if len(gabor_masked_values) > 0:
                gabor_mean = np.mean(gabor_masked_values)
                gabor_std = np.std(gabor_masked_values)
            else:
                gabor_mean = 0.0
                gabor_std = 0.0
            
            return {
                'gabor_mean': float(gabor_mean),
                'gabor_std': float(gabor_std)
            }
        except:
            return {
                'gabor_mean': 0.0,
                'gabor_std': 0.0
            }
    
    def create_parameters_dataframe(self, parameters_list: List[OrganoidParameters]) -> pd.DataFrame:
        """
        Create pandas DataFrame from parameters list.
        
        Args:
            parameters_list: List of OrganoidParameters
            
        Returns:
            DataFrame with all parameters
        """
        if not parameters_list:
            return pd.DataFrame()
        
        rows = []
        
        for params in parameters_list:
            row = {'label': params.label}
            
            # Add all feature dictionaries
            row.update(params.morphological)
            row.update(params.intensity)
            row.update(params.texture)
            row.update(params.spatial)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df
    
    def get_summary_statistics(self, parameters_list: List[OrganoidParameters]) -> Dict[str, Any]:
        """
        Calculate summary statistics for all organoids.
        
        Args:
            parameters_list: List of OrganoidParameters
            
        Returns:
            Dictionary with summary statistics
        """
        if not parameters_list:
            return {}
        
        df = self.create_parameters_dataframe(parameters_list)
        
        # Exclude non-numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        summary = {
            'count': len(parameters_list),
            'feature_count': len(numeric_columns),
            'mean_values': df[numeric_columns].mean().to_dict(),
            'std_values': df[numeric_columns].std().to_dict(),
            'min_values': df[numeric_columns].min().to_dict(),
            'max_values': df[numeric_columns].max().to_dict(),
            'median_values': df[numeric_columns].median().to_dict()
        }
        
        return summary


# Utility functions
def extract_organoid_parameters(image: np.ndarray,
                               labeled_mask: np.ndarray,
                               pixel_size_microns: Optional[float] = None,
                               extract_all: bool = True) -> List[OrganoidParameters]:
    """
    Convenience function for parameter extraction.
    
    Args:
        image: Original image
        labeled_mask: Labeled segmentation mask
        pixel_size_microns: Pixel size in microns
        extract_all: Whether to extract all features
        
    Returns:
        List of OrganoidParameters
    """
    extractor = ParameterExtractor(pixel_size_microns)
    return extractor.extract_parameters(
        image, labeled_mask, 
        extract_texture=extract_all, 
        extract_intensity=extract_all
    )


def parameters_to_csv(parameters_list: List[OrganoidParameters], 
                     output_path: str) -> None:
    """
    Save parameters to CSV file.
    
    Args:
        parameters_list: List of OrganoidParameters
        output_path: Output CSV file path
    """
    extractor = ParameterExtractor()
    df = extractor.create_parameters_dataframe(parameters_list)
    df.to_csv(output_path, index=False)
    logger.info(f"Parameters saved to {output_path}")


def compare_parameter_groups(group1: List[OrganoidParameters],
                           group2: List[OrganoidParameters],
                           feature_name: str) -> Dict[str, float]:
    """
    Compare a specific parameter between two groups of organoids.
    
    Args:
        group1: First group of organoids
        group2: Second group of organoids
        feature_name: Name of feature to compare
        
    Returns:
        Dictionary with comparison statistics
    """
    from scipy.stats import ttest_ind, mannwhitneyu
    
    extractor = ParameterExtractor()
    
    df1 = extractor.create_parameters_dataframe(group1)
    df2 = extractor.create_parameters_dataframe(group2)
    
    if feature_name not in df1.columns or feature_name not in df2.columns:
        raise ValueError(f"Feature '{feature_name}' not found in parameters")
    
    values1 = df1[feature_name].dropna().values
    values2 = df2[feature_name].dropna().values
    
    if len(values1) == 0 or len(values2) == 0:
        raise ValueError("Insufficient data for comparison")
    
    # T-test
    t_stat, t_pvalue = ttest_ind(values1, values2)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = mannwhitneyu(values1, values2, alternative='two-sided')
    
    return {
        'feature': feature_name,
        'group1_mean': float(np.mean(values1)),
        'group1_std': float(np.std(values1)),
        'group1_n': len(values1),
        'group2_mean': float(np.mean(values2)),
        'group2_std': float(np.std(values2)),
        'group2_n': len(values2),
        't_statistic': float(t_stat),
        't_pvalue': float(t_pvalue),
        'u_statistic': float(u_stat),
        'u_pvalue': float(u_pvalue)
    }