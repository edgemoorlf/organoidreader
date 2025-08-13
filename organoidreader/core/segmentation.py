"""
Segmentation Engine Module

This module provides the main interface for organoid segmentation,
integrating preprocessing, model inference, and post-processing.
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Dict, Any, List
from pathlib import Path
import cv2
from skimage import morphology, measure, segmentation, feature
from scipy import ndimage

from organoidreader.models.unet import UNet, MultiScaleUNet, create_unet_model
from organoidreader.core.preprocessing import ImagePreprocessor, standardize_for_model
from organoidreader.config.config_manager import SegmentationConfig, ModelConfig

logger = logging.getLogger(__name__)


class SegmentationEngine:
    """
    Main segmentation engine for organoid detection and segmentation.
    
    Combines preprocessing, deep learning inference, and post-processing
    to provide robust organoid segmentation results.
    """
    
    def __init__(self, 
                 model_config: Optional[ModelConfig] = None,
                 segmentation_config: Optional[SegmentationConfig] = None):
        """
        Initialize segmentation engine.
        
        Args:
            model_config: Model configuration
            segmentation_config: Segmentation configuration
        """
        self.model_config = model_config if model_config else ModelConfig()
        self.seg_config = segmentation_config if segmentation_config else SegmentationConfig()
        
        self.device = self._setup_device()
        self.model = None
        self.preprocessor = ImagePreprocessor()
        
        logger.info(f"SegmentationEngine initialized with device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.model_config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.model_config.device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def load_model(self, 
                   model_path: Optional[Union[str, Path]] = None,
                   model_type: str = "attention") -> None:
        """
        Load segmentation model.
        
        Args:
            model_path: Path to saved model weights. If None, creates new model
            model_type: Type of model to create ("standard", "attention", "multiscale")
        """
        try:
            # Create model
            self.model = create_unet_model(
                model_type=model_type,
                in_channels=1,  # Assuming grayscale input
                out_channels=1,  # Binary segmentation
                dropout_rate=0.1
            )
            
            self.model = self.model.to(self.device)
            
            # Load weights if provided
            if model_path is not None:
                model_path = Path(model_path)
                if model_path.exists():
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"Loaded model weights from {model_path}")
                else:
                    logger.warning(f"Model file not found: {model_path}, using untrained model")
            else:
                logger.info("Created new untrained model")
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def segment_image(self, 
                     image: np.ndarray,
                     preprocess: bool = True,
                     postprocess: bool = True) -> Dict[str, Any]:
        """
        Segment organoids in an image.
        
        Args:
            image: Input image array
            preprocess: Whether to apply preprocessing
            postprocess: Whether to apply post-processing
            
        Returns:
            Dictionary containing segmentation results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.info("Starting organoid segmentation")
        
        try:
            original_shape = image.shape
            
            # Preprocessing
            if preprocess:
                processed_image = standardize_for_model(image, target_size=(512, 512))
            else:
                processed_image = image
            
            # Ensure proper format for model
            if len(processed_image.shape) == 2:
                processed_image = np.expand_dims(processed_image, axis=-1)
            
            # Convert to tensor
            input_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).unsqueeze(0)
            input_tensor = input_tensor.float().to(self.device)
            
            # Model inference
            with torch.no_grad():
                output = self.model(input_tensor)
                prediction = torch.sigmoid(output).cpu().numpy()
            
            # Remove batch and channel dimensions
            prediction = prediction[0, 0]
            
            # Resize back to original size if needed
            if prediction.shape != original_shape[:2]:
                prediction = cv2.resize(prediction, 
                                     (original_shape[1], original_shape[0]), 
                                     interpolation=cv2.INTER_LINEAR)
            
            # Post-processing
            if postprocess:
                binary_mask, labeled_mask, stats = self._postprocess_prediction(
                    prediction, original_shape[:2]
                )
            else:
                binary_mask = prediction > self.seg_config.confidence_threshold
                labeled_mask = measure.label(binary_mask)
                stats = self._calculate_region_stats(labeled_mask)
            
            result = {
                'prediction': prediction,
                'binary_mask': binary_mask,
                'labeled_mask': labeled_mask,
                'statistics': stats,
                'num_organoids': len(stats),
                'original_shape': original_shape,
                'processed_shape': processed_image.shape
            }
            
            logger.info(f"Segmentation completed. Found {len(stats)} organoids")
            return result
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            raise
    
    def _postprocess_prediction(self, 
                               prediction: np.ndarray, 
                               target_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Post-process segmentation prediction.
        
        Args:
            prediction: Raw model prediction
            target_shape: Target shape for output
            
        Returns:
            Tuple of (binary_mask, labeled_mask, statistics)
        """
        # Threshold prediction
        binary_mask = prediction > self.seg_config.confidence_threshold
        
        # Morphological operations
        if self.seg_config.morphological_opening:
            kernel = morphology.disk(2)
            binary_mask = morphology.binary_opening(binary_mask, kernel)
        
        # Remove small objects
        binary_mask = morphology.remove_small_objects(
            binary_mask, 
            min_size=self.seg_config.min_object_size
        )
        
        # Fill holes
        binary_mask = ndimage.binary_fill_holes(binary_mask)
        
        # Remove objects touching border
        if self.seg_config.remove_border_objects:
            binary_mask = segmentation.clear_border(binary_mask)
        
        # Watershed segmentation for overlapping objects
        if self.seg_config.use_watershed:
            binary_mask = self._apply_watershed(binary_mask)
        
        # Label connected components
        labeled_mask = measure.label(binary_mask)
        
        # Filter by size
        labeled_mask = self._filter_by_size(labeled_mask)
        
        # Calculate statistics
        stats = self._calculate_region_stats(labeled_mask)
        
        return binary_mask, labeled_mask, stats
    
    def _apply_watershed(self, binary_mask: np.ndarray) -> np.ndarray:
        """Apply watershed segmentation to separate touching objects."""
        # Distance transform
        distance = ndimage.distance_transform_edt(binary_mask)
        
        # Find local maxima
        from skimage.feature import peak_local_maxima
        coords = peak_local_maxima(distance, min_distance=10, threshold_abs=0.3)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndimage.label(mask)
        
        # Apply watershed
        labels = segmentation.watershed(-distance, markers, mask=binary_mask)
        
        return labels > 0
    
    def _filter_by_size(self, labeled_mask: np.ndarray) -> np.ndarray:
        """Filter labeled regions by size constraints."""
        filtered_mask = np.zeros_like(labeled_mask)
        current_label = 1
        
        for region in measure.regionprops(labeled_mask):
            area = region.area
            
            if (self.seg_config.min_object_size <= area <= self.seg_config.max_object_size):
                coords = region.coords
                filtered_mask[coords[:, 0], coords[:, 1]] = current_label
                current_label += 1
        
        return filtered_mask
    
    def _calculate_region_stats(self, labeled_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Calculate statistics for each segmented region."""
        stats = []
        
        for region in measure.regionprops(labeled_mask):
            region_stats = {
                'label': region.label,
                'area': region.area,
                'perimeter': region.perimeter,
                'centroid': region.centroid,
                'bbox': region.bbox,
                'circularity': 4 * np.pi * region.area / (region.perimeter ** 2) if region.perimeter > 0 else 0,
                'solidity': region.solidity,
                'eccentricity': region.eccentricity,
                'major_axis_length': region.major_axis_length,
                'minor_axis_length': region.minor_axis_length,
                'orientation': region.orientation,
                'extent': region.extent
            }
            stats.append(region_stats)
        
        return stats
    
    def segment_batch(self, 
                     images: List[np.ndarray],
                     preprocess: bool = True,
                     postprocess: bool = True) -> List[Dict[str, Any]]:
        """
        Segment multiple images in batch.
        
        Args:
            images: List of image arrays
            preprocess: Whether to apply preprocessing
            postprocess: Whether to apply post-processing
            
        Returns:
            List of segmentation results
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                result = self.segment_image(image, preprocess, postprocess)
                results.append(result)
                logger.debug(f"Segmented image {i+1}/{len(images)}")
            except Exception as e:
                logger.error(f"Failed to segment image {i+1}: {e}")
                results.append(None)
        
        return results
    
    def visualize_segmentation(self, 
                              image: np.ndarray, 
                              result: Dict[str, Any],
                              show_labels: bool = True) -> np.ndarray:
        """
        Create visualization of segmentation results.
        
        Args:
            image: Original image
            result: Segmentation result dictionary
            show_labels: Whether to show region labels
            
        Returns:
            Visualization image
        """
        # Convert image to RGB if grayscale
        if len(image.shape) == 2:
            vis_image = np.stack([image] * 3, axis=-1)
        else:
            vis_image = image.copy()
        
        # Normalize to 0-255 range
        vis_image = ((vis_image - vis_image.min()) / (vis_image.max() - vis_image.min()) * 255).astype(np.uint8)
        
        # Overlay segmentation mask
        binary_mask = result['binary_mask']
        labeled_mask = result['labeled_mask']
        
        # Create colored overlay
        overlay = np.zeros_like(vis_image)
        overlay[binary_mask] = [0, 255, 0]  # Green for organoids
        
        # Blend with original image
        vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
        
        # Add contours
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_image, contours, -1, (255, 0, 0), 2)  # Blue contours
        
        # Add labels if requested
        if show_labels:
            for stat in result['statistics']:
                centroid = stat['centroid']
                label_text = str(stat['label'])
                cv2.putText(vis_image, label_text, 
                           (int(centroid[1]), int(centroid[0])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return vis_image
    
    def get_segmentation_summary(self, result: Dict[str, Any]) -> str:
        """
        Generate text summary of segmentation results.
        
        Args:
            result: Segmentation result dictionary
            
        Returns:
            Formatted summary string
        """
        stats = result['statistics']
        num_organoids = len(stats)
        
        if num_organoids == 0:
            return "No organoids detected in the image."
        
        # Calculate aggregate statistics
        areas = [s['area'] for s in stats]
        circularities = [s['circularity'] for s in stats]
        
        summary = []
        summary.append(f"=== SEGMENTATION SUMMARY ===")
        summary.append(f"Number of organoids detected: {num_organoids}")
        summary.append(f"Total area covered: {sum(areas):.1f} pixels")
        summary.append(f"Average organoid area: {np.mean(areas):.1f} ± {np.std(areas):.1f} pixels")
        summary.append(f"Area range: {min(areas):.1f} - {max(areas):.1f} pixels")
        summary.append(f"Average circularity: {np.mean(circularities):.3f} ± {np.std(circularities):.3f}")
        summary.append(f"Image coverage: {sum(areas) / (result['original_shape'][0] * result['original_shape'][1]) * 100:.2f}%")
        
        return "\n".join(summary)


# Utility functions
def segment_image(image: np.ndarray, 
                 model_path: Optional[str] = None,
                 model_type: str = "attention") -> Dict[str, Any]:
    """
    Convenience function for single image segmentation.
    
    Args:
        image: Input image array
        model_path: Path to model weights
        model_type: Type of model to use
        
    Returns:
        Segmentation result dictionary
    """
    engine = SegmentationEngine()
    engine.load_model(model_path, model_type)
    return engine.segment_image(image)


def create_segmentation_engine(model_config: Optional[ModelConfig] = None,
                              segmentation_config: Optional[SegmentationConfig] = None) -> SegmentationEngine:
    """
    Factory function to create segmentation engine.
    
    Args:
        model_config: Model configuration
        segmentation_config: Segmentation configuration
        
    Returns:
        Configured SegmentationEngine instance
    """
    return SegmentationEngine(model_config, segmentation_config)