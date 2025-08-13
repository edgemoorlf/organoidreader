"""
Apoptosis Detection Module

This module provides comprehensive apoptosis detection and analysis for organoids,
including morphological feature recognition, TUNEL assay image analysis,
nuclear fragmentation detection, and apoptosis quantification.
"""

import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from skimage import filters, morphology, measure, segmentation, feature, restoration
from scipy import ndimage, stats
import torch
import torch.nn as nn

from organoidreader.core.parameter_extraction import OrganoidParameters

logger = logging.getLogger(__name__)


@dataclass
class ApoptosisResults:
    """Container for apoptosis detection results."""
    organoid_id: int
    apoptosis_score: float  # 0-1 scale
    apoptotic_cell_percentage: float
    nuclear_fragmentation_score: float
    condensed_chromatin_score: float
    membrane_blebbing_score: float
    apoptosis_stage: str  # "early", "mid", "late", "none"
    confidence: float
    apoptotic_features: Dict[str, float]
    detected_regions: List[Dict[str, Any]]  # Regions showing apoptotic features
    metadata: Dict[str, Any]


class ApoptosisDetector:
    """
    Comprehensive apoptosis detector for organoids.
    
    Detects apoptotic cells based on:
    1. Nuclear fragmentation patterns
    2. Chromatin condensation
    3. Cell shrinkage and rounding
    4. Membrane blebbing
    5. TUNEL assay specific features (if applicable)
    """
    
    def __init__(self, detection_mode: str = "morphological"):
        """
        Initialize apoptosis detector.
        
        Args:
            detection_mode: "morphological", "tunel", or "combined"
        """
        self.detection_mode = detection_mode
        
        # Apoptosis detection parameters
        self.params = {
            'min_fragment_size': 10,
            'max_fragment_size': 200,
            'condensation_threshold': 0.3,
            'fragmentation_threshold': 0.4,
            'blebbing_size_threshold': 5
        }
        
        # Stage classification thresholds
        self.stage_thresholds = {
            'early': (0.2, 0.4),    # Early apoptosis
            'mid': (0.4, 0.7),      # Mid-stage apoptosis
            'late': (0.7, 1.0),     # Late apoptosis/necrosis
        }
    
    def detect_apoptosis(self, 
                        image: np.ndarray,
                        organoid_mask: np.ndarray,
                        nuclear_channel: Optional[np.ndarray] = None,
                        organoid_params: Optional[OrganoidParameters] = None) -> ApoptosisResults:
        """
        Perform comprehensive apoptosis detection on a single organoid.
        
        Args:
            image: Original grayscale image (bright-field or fluorescence)
            organoid_mask: Binary mask of the organoid
            nuclear_channel: Optional nuclear staining channel (DAPI, etc.)
            organoid_params: Pre-computed organoid parameters
            
        Returns:
            ApoptosisResults with comprehensive analysis
        """
        logger.debug("Starting apoptosis detection")
        
        # Use nuclear channel if available, otherwise use main image
        analysis_image = nuclear_channel if nuclear_channel is not None else image
        
        # Extract organoid region
        organoid_region = analysis_image * organoid_mask
        
        # Detect apoptotic features
        nuclear_fragmentation = self._detect_nuclear_fragmentation(organoid_region, organoid_mask)
        chromatin_condensation = self._detect_chromatin_condensation(organoid_region, organoid_mask)
        membrane_blebbing = self._detect_membrane_blebbing(image, organoid_mask)
        
        # Calculate feature scores
        fragmentation_score = nuclear_fragmentation['fragmentation_score']
        condensation_score = chromatin_condensation['condensation_score']
        blebbing_score = membrane_blebbing['blebbing_score']
        
        # Calculate overall apoptosis score
        apoptosis_score = self._calculate_apoptosis_score(
            fragmentation_score, condensation_score, blebbing_score
        )
        
        # Estimate apoptotic cell percentage
        apoptotic_percentage = self._estimate_apoptotic_percentage(
            organoid_region, organoid_mask, apoptosis_score
        )
        
        # Classify apoptosis stage
        stage = self._classify_apoptosis_stage(apoptosis_score)
        
        # Calculate confidence
        confidence = self._calculate_confidence(fragmentation_score, condensation_score, blebbing_score)
        
        # Combine detected regions
        detected_regions = []
        detected_regions.extend(nuclear_fragmentation.get('fragments', []))
        detected_regions.extend(chromatin_condensation.get('condensed_regions', []))
        detected_regions.extend(membrane_blebbing.get('blebs', []))
        
        # Compile apoptotic features
        apoptotic_features = {
            'nuclear_fragments_count': len(nuclear_fragmentation.get('fragments', [])),
            'fragment_size_mean': nuclear_fragmentation.get('mean_fragment_size', 0),
            'fragment_size_std': nuclear_fragmentation.get('std_fragment_size', 0),
            'condensed_area_ratio': chromatin_condensation.get('condensed_ratio', 0),
            'condensation_intensity': chromatin_condensation.get('intensity_ratio', 0),
            'membrane_blebs_count': len(membrane_blebbing.get('blebs', [])),
            'blebbing_perimeter_ratio': membrane_blebbing.get('perimeter_ratio', 0),
            'overall_morphology_score': self._calculate_morphology_score(organoid_mask)
        }
        
        # Get organoid ID
        organoid_id = organoid_params.label if organoid_params else 1
        
        result = ApoptosisResults(
            organoid_id=organoid_id,
            apoptosis_score=apoptosis_score,
            apoptotic_cell_percentage=apoptotic_percentage,
            nuclear_fragmentation_score=fragmentation_score,
            condensed_chromatin_score=condensation_score,
            membrane_blebbing_score=blebbing_score,
            apoptosis_stage=stage,
            confidence=confidence,
            apoptotic_features=apoptotic_features,
            detected_regions=detected_regions,
            metadata={
                'detection_mode': self.detection_mode,
                'nuclear_channel_used': nuclear_channel is not None,
                'organoid_area': np.sum(organoid_mask)
            }
        )
        
        logger.debug(f"Apoptosis detection completed: {stage} ({apoptosis_score:.3f})")
        return result
    
    def _detect_nuclear_fragmentation(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Detect nuclear fragmentation patterns."""
        if np.sum(mask) == 0:
            return {'fragmentation_score': 0.0, 'fragments': []}
        
        # Enhance nuclear structures
        enhanced = self._enhance_nuclear_structures(image, mask)
        
        # Threshold to find nuclear regions
        try:
            # Use local thresholding for better fragmentation detection
            threshold = filters.threshold_local(enhanced, block_size=15, offset=0.01)
            nuclear_regions = enhanced > threshold
            nuclear_regions = nuclear_regions & mask
        except:
            # Fallback to Otsu thresholding
            masked_pixels = enhanced[mask > 0]
            if len(masked_pixels) > 0:
                threshold = filters.threshold_otsu(masked_pixels)
                nuclear_regions = (enhanced > threshold) & mask
            else:
                nuclear_regions = mask
        
        # Remove small noise
        nuclear_regions = morphology.remove_small_objects(nuclear_regions, min_size=self.params['min_fragment_size'])
        
        # Label connected components (potential fragments)
        labeled_nuclei = measure.label(nuclear_regions)
        fragments = measure.regionprops(labeled_nuclei)
        
        # Filter fragments by size
        valid_fragments = []
        for fragment in fragments:
            if self.params['min_fragment_size'] <= fragment.area <= self.params['max_fragment_size']:
                valid_fragments.append({
                    'centroid': fragment.centroid,
                    'area': fragment.area,
                    'eccentricity': fragment.eccentricity,
                    'solidity': fragment.solidity,
                    'bbox': fragment.bbox
                })
        
        # Calculate fragmentation score
        total_nuclear_area = np.sum(nuclear_regions)
        organoid_area = np.sum(mask)
        
        if organoid_area > 0:
            nuclear_density = total_nuclear_area / organoid_area
            fragment_count_normalized = len(valid_fragments) / max(organoid_area / 1000, 1)  # Per 1000 pixels
            
            # Higher fragment count and lower individual fragment sizes indicate more fragmentation
            if len(valid_fragments) > 0:
                mean_fragment_size = np.mean([f['area'] for f in valid_fragments])
                size_variability = np.std([f['area'] for f in valid_fragments]) / max(mean_fragment_size, 1)
                
                # Fragmentation score based on count, size variability, and nuclear density
                fragmentation_score = min(1.0, (fragment_count_normalized * 0.4 + 
                                               size_variability * 0.3 + 
                                               nuclear_density * 0.3))
            else:
                fragmentation_score = 0.0
                mean_fragment_size = 0
                size_variability = 0
        else:
            fragmentation_score = 0.0
            mean_fragment_size = 0
            size_variability = 0
        
        return {
            'fragmentation_score': float(fragmentation_score),
            'fragments': valid_fragments,
            'fragment_count': len(valid_fragments),
            'mean_fragment_size': float(mean_fragment_size) if valid_fragments else 0.0,
            'std_fragment_size': float(size_variability * mean_fragment_size) if valid_fragments else 0.0,
            'nuclear_density': float(nuclear_density) if organoid_area > 0 else 0.0
        }
    
    def _detect_chromatin_condensation(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Detect chromatin condensation patterns."""
        if np.sum(mask) == 0:
            return {'condensation_score': 0.0, 'condensed_regions': []}
        
        # Enhance contrast for chromatin detection
        enhanced = exposure.equalize_adapthist(image * mask, clip_limit=0.03)
        
        # Detect high-intensity regions (condensed chromatin)
        masked_pixels = enhanced[mask > 0]
        if len(masked_pixels) == 0:
            return {'condensation_score': 0.0, 'condensed_regions': []}
        
        # Use upper percentile as threshold for condensed regions
        condensation_threshold = np.percentile(masked_pixels, 85)
        condensed_regions = (enhanced > condensation_threshold) & mask
        
        # Remove small noise
        condensed_regions = morphology.remove_small_objects(condensed_regions, min_size=5)
        
        # Label and analyze condensed regions
        labeled_condensed = measure.label(condensed_regions)
        condensed_props = measure.regionprops(labeled_condensed)
        
        # Calculate condensation metrics
        condensed_area = np.sum(condensed_regions)
        organoid_area = np.sum(mask)
        condensed_ratio = condensed_area / organoid_area if organoid_area > 0 else 0
        
        # Intensity-based condensation score
        mean_organoid_intensity = np.mean(masked_pixels)
        condensed_pixels = enhanced[condensed_regions]
        
        if len(condensed_pixels) > 0:
            mean_condensed_intensity = np.mean(condensed_pixels)
            intensity_ratio = mean_condensed_intensity / max(mean_organoid_intensity, 1e-6)
        else:
            intensity_ratio = 1.0
        
        # Combine area ratio and intensity ratio for condensation score
        condensation_score = min(1.0, (condensed_ratio * 2 + (intensity_ratio - 1) * 0.5))
        condensation_score = max(0.0, condensation_score)
        
        # Extract condensed region information
        condensed_region_info = []
        for prop in condensed_props:
            condensed_region_info.append({
                'centroid': prop.centroid,
                'area': prop.area,
                'mean_intensity': prop.mean_intensity,
                'bbox': prop.bbox
            })
        
        return {
            'condensation_score': float(condensation_score),
            'condensed_regions': condensed_region_info,
            'condensed_ratio': float(condensed_ratio),
            'intensity_ratio': float(intensity_ratio),
            'condensed_count': len(condensed_region_info)
        }
    
    def _detect_membrane_blebbing(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """Detect membrane blebbing patterns."""
        if np.sum(mask) == 0:
            return {'blebbing_score': 0.0, 'blebs': []}
        
        # Get cell boundary
        boundary = mask ^ morphology.binary_erosion(mask, morphology.disk(1))
        
        if np.sum(boundary) == 0:
            return {'blebbing_score': 0.0, 'blebs': []}
        
        # Detect protrusions along the boundary
        # Use distance transform to find local maxima near boundary
        distance = ndimage.distance_transform_edt(mask)
        
        # Find local maxima in distance transform near boundary
        local_maxima = feature.peak_local_maxima(distance, min_distance=5, threshold_abs=2)
        
        # Filter maxima that are close to boundary (potential blebs)
        blebs = []
        boundary_coords = np.where(boundary)
        
        for peak in local_maxima:
            peak_coord = np.array([peak[0], peak[1]])
            
            # Check distance to nearest boundary pixel
            distances_to_boundary = np.sqrt(np.sum((np.array([boundary_coords[0], boundary_coords[1]]).T - peak_coord)**2, axis=1))
            min_distance_to_boundary = np.min(distances_to_boundary)
            
            # If peak is close to boundary, it might be a bleb
            if min_distance_to_boundary < 10:  # Within 10 pixels of boundary
                bleb_size = distance[peak]
                
                if bleb_size >= self.params['blebbing_size_threshold']:
                    blebs.append({
                        'centroid': tuple(peak),
                        'size': float(bleb_size),
                        'distance_to_boundary': float(min_distance_to_boundary)
                    })
        
        # Calculate blebbing score
        perimeter = measure.perimeter(mask)
        blebbing_density = len(blebs) / max(perimeter / 100, 1)  # Per 100 pixels of perimeter
        
        if blebs:
            mean_bleb_size = np.mean([b['size'] for b in blebs])
            blebbing_score = min(1.0, blebbing_density * 0.1 + mean_bleb_size / 20)
        else:
            blebbing_score = 0.0
        
        return {
            'blebbing_score': float(blebbing_score),
            'blebs': blebs,
            'bleb_count': len(blebs),
            'perimeter_ratio': float(blebbing_density),
            'mean_bleb_size': float(np.mean([b['size'] for b in blebs])) if blebs else 0.0
        }
    
    def _enhance_nuclear_structures(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Enhance nuclear structures for better detection."""
        # Apply mask
        masked_image = image * mask
        
        # Denoise
        denoised = restoration.denoise_nl_means(masked_image, patch_size=5, patch_distance=7, h=0.1)
        
        # Enhance contrast
        enhanced = exposure.equalize_adapthist(denoised, clip_limit=0.02)
        
        # Apply unsharp masking to enhance edges
        gaussian = filters.gaussian(enhanced, sigma=1)
        unsharp = enhanced - 0.3 * gaussian
        unsharp = np.clip(unsharp, 0, 1)
        
        return unsharp
    
    def _calculate_apoptosis_score(self, 
                                  fragmentation_score: float, 
                                  condensation_score: float, 
                                  blebbing_score: float) -> float:
        """Calculate overall apoptosis score from individual feature scores."""
        # Weight different features based on their importance for apoptosis detection
        weights = {
            'fragmentation': 0.4,  # Nuclear fragmentation is a key indicator
            'condensation': 0.35,  # Chromatin condensation is also important
            'blebbing': 0.25       # Membrane blebbing is supporting evidence
        }
        
        apoptosis_score = (
            weights['fragmentation'] * fragmentation_score +
            weights['condensation'] * condensation_score +
            weights['blebbing'] * blebbing_score
        )
        
        return float(np.clip(apoptosis_score, 0.0, 1.0))
    
    def _estimate_apoptotic_percentage(self, 
                                     image: np.ndarray, 
                                     mask: np.ndarray, 
                                     apoptosis_score: float) -> float:
        """Estimate percentage of apoptotic cells in the organoid."""
        if np.sum(mask) == 0:
            return 0.0
        
        # Use intensity-based segmentation to estimate cell regions
        masked_pixels = image[mask > 0]
        
        if len(masked_pixels) == 0:
            return 0.0
        
        # Segment into potential cell regions using watershed
        try:
            # Distance transform
            distance = ndimage.distance_transform_edt(mask)
            
            # Find local maxima (cell centers)
            local_maxima = feature.peak_local_maxima(distance, min_distance=10, threshold_abs=3)
            
            # Create markers for watershed
            markers = np.zeros(mask.shape, dtype=int)
            for i, peak in enumerate(local_maxima):
                markers[peak] = i + 1
            
            # Apply watershed
            labels = segmentation.watershed(-distance, markers, mask=mask)
            
            # Count regions and estimate apoptotic cells based on apoptosis score
            num_regions = len(local_maxima)
            apoptotic_cells = int(num_regions * apoptosis_score)
            apoptotic_percentage = (apoptotic_cells / max(num_regions, 1)) * 100
            
        except:
            # Fallback: direct estimation based on apoptosis score
            apoptotic_percentage = apoptosis_score * 100
        
        return float(apoptotic_percentage)
    
    def _classify_apoptosis_stage(self, apoptosis_score: float) -> str:
        """Classify apoptosis stage based on overall score."""
        if apoptosis_score < self.stage_thresholds['early'][0]:
            return "none"
        elif self.stage_thresholds['early'][0] <= apoptosis_score < self.stage_thresholds['early'][1]:
            return "early"
        elif self.stage_thresholds['mid'][0] <= apoptosis_score < self.stage_thresholds['mid'][1]:
            return "mid"
        else:
            return "late"
    
    def _calculate_morphology_score(self, mask: np.ndarray) -> float:
        """Calculate morphological changes associated with apoptosis."""
        if np.sum(mask) == 0:
            return 0.0
        
        props = measure.regionprops(mask.astype(int))[0]
        
        # Apoptotic cells tend to become more circular and compact
        circularity = 4 * np.pi * props.area / (props.perimeter ** 2) if props.perimeter > 0 else 0
        compactness = props.area / props.convex_area if props.convex_area > 0 else 0
        
        # Cell shrinkage indicator (relative to expected size)
        size_factor = min(props.area / 1000, 1.0)  # Normalize by expected size
        
        # Combine factors (higher values suggest apoptotic morphology)
        morphology_score = (circularity * 0.4 + compactness * 0.4 + (1 - size_factor) * 0.2)
        
        return float(np.clip(morphology_score, 0.0, 1.0))
    
    def _calculate_confidence(self, fragmentation: float, condensation: float, blebbing: float) -> float:
        """Calculate confidence in apoptosis detection."""
        # Confidence based on consistency between different features
        scores = [fragmentation, condensation, blebbing]
        
        # Higher consistency (lower std) gives higher confidence
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Confidence decreases with higher variability
        consistency_confidence = max(0.1, 1.0 - std_score)
        
        # Confidence also depends on signal strength
        signal_confidence = min(1.0, mean_score * 2)  # Strong signals give higher confidence
        
        overall_confidence = (consistency_confidence + signal_confidence) / 2
        
        return float(np.clip(overall_confidence, 0.1, 1.0))
    
    def detect_tunel_positive_cells(self, 
                                   tunel_image: np.ndarray, 
                                   organoid_mask: np.ndarray) -> Dict[str, Any]:
        """
        Detect TUNEL-positive cells in TUNEL assay images.
        
        Args:
            tunel_image: TUNEL staining image (fluorescence)
            organoid_mask: Organoid boundary mask
            
        Returns:
            Dictionary with TUNEL analysis results
        """
        if np.sum(organoid_mask) == 0:
            return {'tunel_positive_ratio': 0.0, 'tunel_positive_cells': []}
        
        # Enhance TUNEL signal
        enhanced_tunel = exposure.equalize_adapthist(tunel_image * organoid_mask)
        
        # Threshold for TUNEL-positive regions
        tunel_pixels = enhanced_tunel[organoid_mask > 0]
        
        if len(tunel_pixels) == 0:
            return {'tunel_positive_ratio': 0.0, 'tunel_positive_cells': []}
        
        # Use upper percentile as threshold for positive signal
        tunel_threshold = np.percentile(tunel_pixels, 75)
        tunel_positive = (enhanced_tunel > tunel_threshold) & organoid_mask
        
        # Label and analyze TUNEL-positive regions
        labeled_tunel = measure.label(tunel_positive)
        tunel_props = measure.regionprops(labeled_tunel)
        
        # Filter by size (remove noise)
        valid_tunel_cells = []
        for prop in tunel_props:
            if prop.area >= 10:  # Minimum cell size
                valid_tunel_cells.append({
                    'centroid': prop.centroid,
                    'area': prop.area,
                    'mean_intensity': prop.mean_intensity,
                    'bbox': prop.bbox
                })
        
        # Calculate TUNEL-positive ratio
        tunel_positive_area = sum([cell['area'] for cell in valid_tunel_cells])
        organoid_area = np.sum(organoid_mask)
        tunel_positive_ratio = tunel_positive_area / organoid_area if organoid_area > 0 else 0
        
        return {
            'tunel_positive_ratio': float(tunel_positive_ratio),
            'tunel_positive_cells': valid_tunel_cells,
            'tunel_cell_count': len(valid_tunel_cells)
        }


def create_apoptosis_summary(results: List[ApoptosisResults]) -> Dict[str, Any]:
    """
    Create summary statistics for apoptosis detection results.
    
    Args:
        results: List of ApoptosisResults
        
    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {}
    
    # Extract values
    apoptosis_scores = [r.apoptosis_score for r in results]
    apoptotic_percentages = [r.apoptotic_cell_percentage for r in results]
    confidence_scores = [r.confidence for r in results]
    
    # Count stages
    stages = [r.apoptosis_stage for r in results]
    stage_counts = {
        'none': stages.count('none'),
        'early': stages.count('early'),
        'mid': stages.count('mid'),
        'late': stages.count('late')
    }
    
    summary = {
        'total_organoids': len(results),
        'average_apoptosis_score': float(np.mean(apoptosis_scores)),
        'apoptosis_score_std': float(np.std(apoptosis_scores)),
        'average_apoptotic_percentage': float(np.mean(apoptotic_percentages)),
        'average_confidence': float(np.mean(confidence_scores)),
        'stage_counts': stage_counts,
        'stage_percentages': {
            k: (v / len(results)) * 100 for k, v in stage_counts.items()
        },
        'apoptotic_organoids_percentage': ((len(results) - stage_counts['none']) / len(results)) * 100
    }
    
    return summary


# Utility functions
def detect_organoid_apoptosis(image: np.ndarray, 
                             mask: np.ndarray,
                             nuclear_channel: Optional[np.ndarray] = None) -> ApoptosisResults:
    """
    Convenience function for single organoid apoptosis detection.
    
    Args:
        image: Grayscale image
        mask: Organoid binary mask
        nuclear_channel: Optional nuclear channel
        
    Returns:
        ApoptosisResults
    """
    detector = ApoptosisDetector()
    return detector.detect_apoptosis(image, mask, nuclear_channel)