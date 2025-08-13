"""
Training Metrics Module

Provides comprehensive metrics for evaluating segmentation model performance
during training and validation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import logging

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """
    Comprehensive metrics for segmentation evaluation.
    
    Computes standard segmentation metrics including Dice score, IoU,
    precision, recall, and F1 score.
    """
    
    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        """
        Initialize metrics calculator.
        
        Args:
            threshold: Threshold for binary classification
            eps: Small epsilon to avoid division by zero
        """
        self.threshold = threshold
        self.eps = eps
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics."""
        self.total_dice = 0.0
        self.total_iou = 0.0
        self.total_precision = 0.0
        self.total_recall = 0.0
        self.total_f1 = 0.0
        self.total_samples = 0
        
        # For confusion matrix calculation
        self.all_predictions = []
        self.all_targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with a batch of predictions and targets.
        
        Args:
            predictions: Model predictions (batch_size, channels, height, width)
            targets: Ground truth masks (batch_size, channels, height, width)
        """
        # Apply sigmoid if predictions are logits
        if predictions.min() < 0 or predictions.max() > 1:
            predictions = torch.sigmoid(predictions)
        
        # Flatten tensors
        predictions_flat = predictions.view(-1)
        targets_flat = targets.view(-1)
        
        # Binary predictions
        predictions_binary = (predictions_flat > self.threshold).float()
        
        # Calculate metrics for this batch
        dice = self._calculate_dice(predictions_flat, targets_flat)
        iou = self._calculate_iou(predictions_flat, targets_flat)
        precision, recall, f1 = self._calculate_precision_recall_f1(predictions_binary, targets_flat)
        
        # Accumulate metrics
        batch_size = predictions.size(0)
        self.total_dice += dice * batch_size
        self.total_iou += iou * batch_size
        self.total_precision += precision * batch_size
        self.total_recall += recall * batch_size
        self.total_f1 += f1 * batch_size
        self.total_samples += batch_size
        
        # Store for confusion matrix
        self.all_predictions.extend(predictions_binary.cpu().numpy())
        self.all_targets.extend(targets_flat.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        if self.total_samples == 0:
            return {
                'dice_score': 0.0,
                'iou': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
        
        metrics = {
            'dice_score': self.total_dice / self.total_samples,
            'iou': self.total_iou / self.total_samples,
            'precision': self.total_precision / self.total_samples,
            'recall': self.total_recall / self.total_samples,
            'f1_score': self.total_f1 / self.total_samples
        }
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix for all accumulated predictions."""
        if len(self.all_predictions) == 0:
            return np.zeros((2, 2))
        
        return confusion_matrix(self.all_targets, self.all_predictions)
    
    def _calculate_dice(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate Dice coefficient."""
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        if union == 0:
            return 1.0  # Perfect score if both are empty
        
        dice = (2.0 * intersection + self.eps) / (union + self.eps)
        return dice.item()
    
    def _calculate_iou(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate Intersection over Union (IoU)."""
        # Convert to binary
        predictions_binary = (predictions > self.threshold).float()
        
        intersection = (predictions_binary * targets).sum()
        union = predictions_binary.sum() + targets.sum() - intersection
        
        if union == 0:
            return 1.0  # Perfect score if both are empty
        
        iou = (intersection + self.eps) / (union + self.eps)
        return iou.item()
    
    def _calculate_precision_recall_f1(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        # Convert to numpy for sklearn
        pred_np = predictions.cpu().numpy().astype(int)
        target_np = targets.cpu().numpy().astype(int)
        
        # Handle edge case where all predictions are the same
        if len(np.unique(pred_np)) == 1 or len(np.unique(target_np)) == 1:
            if np.array_equal(pred_np, target_np):
                return 1.0, 1.0, 1.0
            else:
                return 0.0, 0.0, 0.0
        
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                target_np, pred_np, average='binary', zero_division=0
            )
            return float(precision), float(recall), float(f1)
        except:
            return 0.0, 0.0, 0.0


class LossTracker:
    """Track and compute running averages of loss values."""
    
    def __init__(self):
        """Initialize loss tracker."""
        self.reset()
    
    def reset(self):
        """Reset accumulated losses."""
        self.total_loss = 0.0
        self.total_samples = 0
        self.loss_history = []
    
    def update(self, loss: float, batch_size: int = 1):
        """
        Update with a new loss value.
        
        Args:
            loss: Loss value
            batch_size: Size of the batch
        """
        self.total_loss += loss * batch_size
        self.total_samples += batch_size
        self.loss_history.append(loss)
    
    def compute(self) -> float:
        """Compute average loss."""
        if self.total_samples == 0:
            return 0.0
        return self.total_loss / self.total_samples
    
    def get_recent_average(self, n: int = 10) -> float:
        """Get average of last n loss values."""
        if len(self.loss_history) == 0:
            return 0.0
        
        recent_losses = self.loss_history[-n:]
        return sum(recent_losses) / len(recent_losses)


class MetricsLogger:
    """
    Logger for training metrics with support for various backends.
    """
    
    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory to save logs (optional)
        """
        self.log_dir = log_dir
        self.metrics_history = {
            'train': [],
            'val': []
        }
        
        # Try to initialize TensorBoard writer if available
        self.tb_writer = None
        if log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir)
                logger.info(f"TensorBoard logging enabled: {log_dir}")
            except ImportError:
                logger.warning("TensorBoard not available, using basic logging")
    
    def log_metrics(self, metrics: Dict[str, float], step: int, phase: str = 'train'):
        """
        Log metrics for a given step and phase.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step/epoch
            phase: Phase ('train' or 'val')
        """
        # Store in history
        metrics_with_step = {'step': step, **metrics}
        self.metrics_history[phase].append(metrics_with_step)
        
        # Log to TensorBoard if available
        if self.tb_writer:
            for name, value in metrics.items():
                self.tb_writer.add_scalar(f"{phase}/{name}", value, step)
        
        # Log to console
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {step} ({phase}): {metrics_str}")
    
    def log_images(self, images: torch.Tensor, step: int, tag: str = 'images'):
        """
        Log images to TensorBoard.
        
        Args:
            images: Tensor of images (batch_size, channels, height, width)
            step: Training step
            tag: Tag for the images
        """
        if self.tb_writer and images.numel() > 0:
            # Take first few images from batch
            images_to_log = images[:4]  # Log up to 4 images
            self.tb_writer.add_images(tag, images_to_log, step)
    
    def close(self):
        """Close the logger and clean up resources."""
        if self.tb_writer:
            self.tb_writer.close()
    
    def get_best_metric(self, metric_name: str, phase: str = 'val') -> Tuple[float, int]:
        """
        Get best value and step for a specific metric.
        
        Args:
            metric_name: Name of the metric
            phase: Phase to check ('train' or 'val')
            
        Returns:
            Tuple of (best_value, best_step)
        """
        if phase not in self.metrics_history or not self.metrics_history[phase]:
            return 0.0, 0
        
        history = self.metrics_history[phase]
        
        # Find entry with best metric (assuming higher is better for most metrics)
        best_entry = max(history, key=lambda x: x.get(metric_name, 0))
        
        return best_entry.get(metric_name, 0.0), best_entry['step']
    
    def save_metrics_history(self, filepath: str):
        """Save metrics history to a file."""
        import json
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            logger.info(f"Metrics history saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save metrics history: {e}")


def calculate_metrics(predictions: torch.Tensor, 
                     targets: torch.Tensor, 
                     threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate segmentation metrics for a batch of predictions.
    
    Args:
        predictions: Model predictions
        targets: Ground truth masks
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of computed metrics
    """
    metrics_calculator = SegmentationMetrics(threshold=threshold)
    metrics_calculator.update(predictions, targets)
    return metrics_calculator.compute()


def print_metrics_summary(metrics: Dict[str, float], phase: str = ""):
    """
    Print a formatted summary of metrics.
    
    Args:
        metrics: Dictionary of metrics
        phase: Optional phase identifier
    """
    if phase:
        print(f"\n{phase.upper()} METRICS:")
    else:
        print("\nMETRICS SUMMARY:")
    
    print("-" * 40)
    
    for metric_name, value in metrics.items():
        print(f"{metric_name.replace('_', ' ').title():<15}: {value:.4f}")
    
    print("-" * 40)