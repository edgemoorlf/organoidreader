"""
Viability Analysis Module

This module provides comprehensive viability analysis for organoids,
including live/dead cell classification, membrane integrity assessment,
and viability scoring algorithms.
"""

import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from skimage import filters, morphology, measure, segmentation, feature, exposure
from scipy import ndimage, stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from organoidreader.core.parameter_extraction import OrganoidParameters
from organoidreader.models.unet import create_unet_model

logger = logging.getLogger(__name__)


@dataclass
class ViabilityResults:
    """Container for viability analysis results."""
    organoid_id: int
    viability_score: float  # 0-1 scale
    live_cell_percentage: float
    dead_cell_percentage: float
    membrane_integrity_score: float
    viability_classification: str  # "viable", "compromised", "non-viable"
    confidence: float
    features: Dict[str, float]
    metadata: Dict[str, Any]


class ViabilityClassifier(nn.Module):
    """
    Neural network for viability classification based on extracted features.
    
    Takes morphological and intensity features as input and predicts viability score.
    """
    
    def __init__(self, input_features: int = 50, hidden_dim: int = 128):
        """
        Initialize viability classifier.
        
        Args:
            input_features: Number of input features
            hidden_dim: Hidden layer dimension
        """
        super(ViabilityClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through classifier."""
        return self.classifier(x)


class ViabilityAnalyzer:
    """
    Comprehensive viability analyzer for organoids.
    
    Combines multiple approaches:
    1. Morphological analysis (shape, compactness)
    2. Intensity analysis (brightness patterns)
    3. Texture analysis (cellular detail preservation)
    4. Machine learning classification
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize viability analyzer.
        
        Args:
            model_path: Path to trained viability classifier model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize ML classifier
        self.classifier = ViabilityClassifier()
        if model_path:
            try:
                self.classifier.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded viability classifier from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load classifier: {e}, using untrained model")
        
        self.classifier.to(self.device)
        self.classifier.eval()
        
        # Viability thresholds
        self.viability_thresholds = {
            'viable': 0.7,      # > 0.7: viable
            'compromised': 0.3,  # 0.3-0.7: compromised
            # < 0.3: non-viable
        }
    
    def analyze_viability(self, 
                         image: np.ndarray,
                         organoid_mask: np.ndarray,
                         organoid_params: Optional[OrganoidParameters] = None) -> ViabilityResults:
        """
        Perform comprehensive viability analysis on a single organoid.
        
        Args:
            image: Original grayscale image
            organoid_mask: Binary mask of the organoid
            organoid_params: Pre-computed organoid parameters (optional)
            
        Returns:
            ViabilityResults with comprehensive analysis
        """
        logger.debug("Starting viability analysis")
        
        # Extract organoid region
        organoid_region = image * organoid_mask
        
        # Calculate viability features
        features = self._extract_viability_features(image, organoid_mask, organoid_region)
        
        # Calculate individual component scores
        morphological_score = self._calculate_morphological_viability(features)
        intensity_score = self._calculate_intensity_viability(features)
        texture_score = self._calculate_texture_viability(features)
        membrane_score = self._assess_membrane_integrity(image, organoid_mask)
        
        # ML-based classification if available
        ml_score = self._ml_viability_prediction(features)
        
        # Combine scores with weights
        viability_score = self._combine_viability_scores(
            morphological_score, intensity_score, texture_score, ml_score
        )
        
        # Calculate live/dead percentages
        live_percentage, dead_percentage = self._estimate_live_dead_ratio(
            image, organoid_mask, viability_score
        )
        
        # Classify viability
        classification = self._classify_viability(viability_score)
        confidence = self._calculate_confidence(features, viability_score)
        
        # Get organoid ID
        organoid_id = organoid_params.label if organoid_params else 1
        
        result = ViabilityResults(
            organoid_id=organoid_id,
            viability_score=viability_score,
            live_cell_percentage=live_percentage,
            dead_cell_percentage=dead_percentage,
            membrane_integrity_score=membrane_score,
            viability_classification=classification,
            confidence=confidence,
            features=features,
            metadata={
                'morphological_score': morphological_score,
                'intensity_score': intensity_score,
                'texture_score': texture_score,
                'ml_score': ml_score
            }
        )
        
        logger.debug(f"Viability analysis completed: {classification} ({viability_score:.3f})")
        return result
    
    def _extract_viability_features(self, 
                                   image: np.ndarray, 
                                   mask: np.ndarray, 
                                   region: np.ndarray) -> Dict[str, float]:
        """Extract features relevant to viability assessment."""
        features = {}
        
        # Get masked pixels
        masked_pixels = region[mask > 0]
        
        if len(masked_pixels) == 0:
            return {key: 0.0 for key in ['mean_intensity', 'std_intensity', 'compactness', 
                                        'circularity', 'texture_variance', 'edge_sharpness']}
        
        # Intensity features
        features['mean_intensity'] = float(np.mean(masked_pixels))
        features['std_intensity'] = float(np.std(masked_pixels))
        features['intensity_range'] = float(np.max(masked_pixels) - np.min(masked_pixels))
        features['intensity_cv'] = features['std_intensity'] / max(features['mean_intensity'], 1e-6)
        
        # Morphological features
        features.update(self._calculate_morphological_features(mask))
        
        # Texture features
        features.update(self._calculate_texture_features(region, mask))
        
        # Edge features
        features.update(self._calculate_edge_features(image, mask))
        
        # Gradient features
        features.update(self._calculate_gradient_features(region, mask))
        
        return features
    
    def _calculate_morphological_features(self, mask: np.ndarray) -> Dict[str, float]:
        """Calculate morphological features from mask."""
        features = {}
        
        # Basic shape properties
        props = measure.regionprops(mask.astype(int))[0] if np.any(mask) else None
        
        if props:
            features['area'] = float(props.area)
            features['perimeter'] = float(props.perimeter)
            features['compactness'] = 4 * np.pi * props.area / (props.perimeter ** 2) if props.perimeter > 0 else 0
            features['circularity'] = features['compactness']  # Same calculation
            features['solidity'] = float(props.solidity)
            features['extent'] = float(props.extent)
            features['eccentricity'] = float(props.eccentricity)
        else:
            features.update({
                'area': 0.0, 'perimeter': 0.0, 'compactness': 0.0,
                'circularity': 0.0, 'solidity': 0.0, 'extent': 0.0, 'eccentricity': 0.0
            })
        
        return features
    
    def _calculate_texture_features(self, region: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Calculate texture-based features."""
        features = {}
        
        masked_region = region * mask
        
        if np.sum(mask) == 0:
            return {'texture_variance': 0.0, 'texture_entropy': 0.0, 'texture_uniformity': 0.0}
        
        # Texture variance
        features['texture_variance'] = float(np.var(masked_region[mask > 0]))
        
        # Local binary patterns for texture analysis
        try:
            lbp = feature.local_binary_pattern(region, P=8, R=1, method='uniform')
            lbp_values = lbp[mask > 0]
            
            if len(lbp_values) > 0:
                hist, _ = np.histogram(lbp_values, bins=10, density=True)
                
                # Entropy
                hist_nonzero = hist[hist > 0]
                features['texture_entropy'] = -np.sum(hist_nonzero * np.log2(hist_nonzero))
                
                # Uniformity
                features['texture_uniformity'] = np.sum(hist ** 2)
            else:
                features['texture_entropy'] = 0.0
                features['texture_uniformity'] = 0.0
                
        except:
            features['texture_entropy'] = 0.0
            features['texture_uniformity'] = 0.0
        
        return features
    
    def _calculate_edge_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Calculate edge-based features."""
        features = {}
        
        # Edge detection
        edges = feature.canny(image, sigma=1.0)
        edge_pixels = edges & mask
        
        features['edge_density'] = np.sum(edge_pixels) / max(np.sum(mask), 1)
        
        # Edge sharpness using gradient magnitude
        grad_x = np.gradient(image, axis=1)
        grad_y = np.gradient(image, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        if np.sum(mask) > 0:
            features['edge_sharpness'] = np.mean(gradient_magnitude[mask > 0])
        else:
            features['edge_sharpness'] = 0.0
        
        return features
    
    def _calculate_gradient_features(self, region: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Calculate gradient-based features."""
        features = {}
        
        if np.sum(mask) == 0:
            return {'gradient_mean': 0.0, 'gradient_std': 0.0}
        
        # Calculate gradients
        grad_x = np.gradient(region, axis=1)
        grad_y = np.gradient(region, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        gradient_values = gradient_magnitude[mask > 0]
        
        features['gradient_mean'] = float(np.mean(gradient_values))
        features['gradient_std'] = float(np.std(gradient_values))
        
        return features
    
    def _calculate_morphological_viability(self, features: Dict[str, float]) -> float:
        """Calculate viability score based on morphological features."""
        # Viable organoids tend to be:
        # - More circular/compact
        # - Higher solidity (fewer holes)
        # - Appropriate size range
        
        circularity_score = min(features.get('circularity', 0) * 2, 1.0)  # Higher is better
        solidity_score = features.get('solidity', 0)
        compactness_score = min(features.get('compactness', 0) * 2, 1.0)
        
        # Size-based scoring (avoid very small or very large)
        area = features.get('area', 0)
        if 100 < area < 10000:  # Reasonable size range
            size_score = 1.0
        elif area < 50:
            size_score = 0.0
        else:
            size_score = max(0.0, 1.0 - (area - 10000) / 10000)
        
        morphological_score = (circularity_score + solidity_score + compactness_score + size_score) / 4
        return float(morphological_score)
    
    def _calculate_intensity_viability(self, features: Dict[str, float]) -> float:
        """Calculate viability score based on intensity features."""
        # Viable organoids typically have:
        # - Moderate brightness (not too dark, not oversaturated)
        # - Good contrast (moderate std)
        # - Reasonable intensity range
        
        mean_intensity = features.get('mean_intensity', 0)
        std_intensity = features.get('std_intensity', 0)
        intensity_cv = features.get('intensity_cv', 0)
        
        # Optimal intensity range (assuming 0-1 normalized)
        if 0.2 < mean_intensity < 0.8:
            intensity_score = 1.0
        else:
            intensity_score = max(0.0, 1.0 - abs(mean_intensity - 0.5) * 2)
        
        # Contrast score (moderate std is good)
        contrast_score = min(std_intensity * 5, 1.0)  # Scale factor
        
        # Coefficient of variation score
        cv_score = min(intensity_cv * 2, 1.0) if intensity_cv < 0.5 else max(0.0, 2.0 - intensity_cv * 2)
        
        intensity_viability = (intensity_score + contrast_score + cv_score) / 3
        return float(intensity_viability)
    
    def _calculate_texture_viability(self, features: Dict[str, float]) -> float:
        """Calculate viability score based on texture features."""
        # Viable organoids show:
        # - Rich texture (high variance, moderate entropy)
        # - Organized structure (reasonable uniformity)
        
        texture_variance = features.get('texture_variance', 0)
        texture_entropy = features.get('texture_entropy', 0)
        texture_uniformity = features.get('texture_uniformity', 0)
        edge_sharpness = features.get('edge_sharpness', 0)
        
        # Texture richness (normalized)
        variance_score = min(texture_variance * 10, 1.0)
        entropy_score = min(texture_entropy / 3.0, 1.0)  # Entropy typically 0-3
        sharpness_score = min(edge_sharpness * 20, 1.0)
        
        texture_viability = (variance_score + entropy_score + sharpness_score) / 3
        return float(texture_viability)
    
    def _assess_membrane_integrity(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Assess membrane integrity based on edge definition."""
        # Detect edges and assess their continuity
        edges = feature.canny(image, sigma=1.0)
        
        # Create boundary mask
        boundary = mask ^ morphology.binary_erosion(mask, morphology.disk(1))
        
        if np.sum(boundary) == 0:
            return 0.0
        
        # Check edge continuity along boundary
        edge_boundary_overlap = np.sum(edges & boundary) / np.sum(boundary)
        
        # Check for gaps in the boundary (indicates membrane damage)
        boundary_props = measure.regionprops(boundary.astype(int))
        if boundary_props:
            solidity = boundary_props[0].solidity
            membrane_integrity = (edge_boundary_overlap + solidity) / 2
        else:
            membrane_integrity = edge_boundary_overlap
        
        return float(membrane_integrity)
    
    def _ml_viability_prediction(self, features: Dict[str, float]) -> float:
        """Get ML-based viability prediction."""
        try:
            # Select relevant features for ML model
            feature_names = [
                'mean_intensity', 'std_intensity', 'intensity_cv',
                'compactness', 'solidity', 'circularity',
                'texture_variance', 'texture_entropy', 'edge_sharpness',
                'gradient_mean'
            ]
            
            # Create feature vector
            feature_vector = []
            for name in feature_names:
                feature_vector.append(features.get(name, 0.0))
            
            # Pad or truncate to expected input size
            while len(feature_vector) < 50:
                feature_vector.append(0.0)
            feature_vector = feature_vector[:50]
            
            # Convert to tensor and predict
            x = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                prediction = self.classifier(x)
                ml_score = prediction.item()
            
            return ml_score
            
        except Exception as e:
            logger.debug(f"ML prediction failed: {e}")
            return 0.5  # Neutral score if ML fails
    
    def _combine_viability_scores(self, 
                                 morphological: float, 
                                 intensity: float, 
                                 texture: float, 
                                 ml: float) -> float:
        """Combine different viability scores with appropriate weights."""
        weights = {
            'morphological': 0.3,
            'intensity': 0.25,
            'texture': 0.2,
            'ml': 0.25
        }
        
        combined_score = (
            weights['morphological'] * morphological +
            weights['intensity'] * intensity +
            weights['texture'] * texture +
            weights['ml'] * ml
        )
        
        return float(np.clip(combined_score, 0.0, 1.0))
    
    def _estimate_live_dead_ratio(self, 
                                 image: np.ndarray, 
                                 mask: np.ndarray, 
                                 viability_score: float) -> Tuple[float, float]:
        """Estimate live/dead cell ratio based on intensity patterns."""
        if np.sum(mask) == 0:
            return 0.0, 100.0
        
        # Extract organoid pixels
        organoid_pixels = image[mask > 0]
        
        # Use Otsu's method to separate bright (live) and dark (dead) regions
        try:
            threshold = filters.threshold_otsu(organoid_pixels)
            live_pixels = organoid_pixels > threshold
            
            live_percentage = np.sum(live_pixels) / len(organoid_pixels) * 100
            dead_percentage = 100 - live_percentage
            
            # Adjust based on overall viability score
            live_percentage *= viability_score
            dead_percentage = 100 - live_percentage
            
        except:
            # Fallback based on viability score
            live_percentage = viability_score * 100
            dead_percentage = 100 - live_percentage
        
        return float(live_percentage), float(dead_percentage)
    
    def _classify_viability(self, viability_score: float) -> str:
        """Classify viability based on score."""
        if viability_score >= self.viability_thresholds['viable']:
            return "viable"
        elif viability_score >= self.viability_thresholds['compromised']:
            return "compromised"
        else:
            return "non-viable"
    
    def _calculate_confidence(self, features: Dict[str, float], viability_score: float) -> float:
        """Calculate confidence in the viability assessment."""
        # Confidence based on feature quality and consistency
        
        # Check if features are in expected ranges
        feature_quality = 0.0
        quality_checks = 0
        
        # Intensity features quality
        mean_intensity = features.get('mean_intensity', 0)
        if 0.1 < mean_intensity < 0.9:
            feature_quality += 1.0
        quality_checks += 1
        
        # Morphological features quality
        area = features.get('area', 0)
        if 50 < area < 20000:
            feature_quality += 1.0
        quality_checks += 1
        
        # Texture features quality
        texture_variance = features.get('texture_variance', 0)
        if texture_variance > 0.001:  # Has some texture
            feature_quality += 1.0
        quality_checks += 1
        
        # Overall feature quality
        feature_confidence = feature_quality / max(quality_checks, 1)
        
        # Distance from decision boundaries affects confidence
        boundary_distances = [
            abs(viability_score - self.viability_thresholds['viable']),
            abs(viability_score - self.viability_thresholds['compromised'])
        ]
        boundary_confidence = min(boundary_distances) * 2  # Scale factor
        
        # Combine confidences
        overall_confidence = (feature_confidence + min(boundary_confidence, 1.0)) / 2
        
        return float(np.clip(overall_confidence, 0.1, 1.0))
    
    def analyze_batch_viability(self, 
                               images: List[np.ndarray],
                               masks: List[np.ndarray],
                               organoid_params: Optional[List[OrganoidParameters]] = None) -> List[ViabilityResults]:
        """
        Analyze viability for multiple organoids.
        
        Args:
            images: List of grayscale images
            masks: List of organoid masks
            organoid_params: Optional list of pre-computed parameters
            
        Returns:
            List of ViabilityResults
        """
        results = []
        
        for i, (image, mask) in enumerate(zip(images, masks)):
            params = organoid_params[i] if organoid_params and i < len(organoid_params) else None
            
            try:
                result = self.analyze_viability(image, mask, params)
                results.append(result)
            except Exception as e:
                logger.error(f"Viability analysis failed for organoid {i}: {e}")
                # Add placeholder result
                results.append(ViabilityResults(
                    organoid_id=i,
                    viability_score=0.0,
                    live_cell_percentage=0.0,
                    dead_cell_percentage=100.0,
                    membrane_integrity_score=0.0,
                    viability_classification="non-viable",
                    confidence=0.0,
                    features={},
                    metadata={'error': str(e)}
                ))
        
        return results


def create_viability_summary(results: List[ViabilityResults]) -> Dict[str, Any]:
    """
    Create summary statistics for viability analysis results.
    
    Args:
        results: List of ViabilityResults
        
    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {}
    
    # Extract values
    viability_scores = [r.viability_score for r in results]
    live_percentages = [r.live_cell_percentage for r in results]
    confidence_scores = [r.confidence for r in results]
    
    # Count classifications
    classifications = [r.viability_classification for r in results]
    class_counts = {
        'viable': classifications.count('viable'),
        'compromised': classifications.count('compromised'),
        'non-viable': classifications.count('non-viable')
    }
    
    summary = {
        'total_organoids': len(results),
        'average_viability_score': float(np.mean(viability_scores)),
        'viability_score_std': float(np.std(viability_scores)),
        'average_live_percentage': float(np.mean(live_percentages)),
        'average_confidence': float(np.mean(confidence_scores)),
        'classification_counts': class_counts,
        'classification_percentages': {
            k: (v / len(results)) * 100 for k, v in class_counts.items()
        },
        'viable_organoids_percentage': (class_counts['viable'] / len(results)) * 100
    }
    
    return summary


# Utility functions
def analyze_organoid_viability(image: np.ndarray, 
                              mask: np.ndarray,
                              model_path: Optional[str] = None) -> ViabilityResults:
    """
    Convenience function for single organoid viability analysis.
    
    Args:
        image: Grayscale image
        mask: Organoid binary mask
        model_path: Optional path to trained classifier
        
    Returns:
        ViabilityResults
    """
    analyzer = ViabilityAnalyzer(model_path)
    return analyzer.analyze_viability(image, mask)