"""
Training Engine Module

Provides the main training loop with support for validation, checkpointing,
early stopping, and comprehensive logging.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from organoidreader.models.unet import create_unet_model, SegmentationLoss
from organoidreader.training.config import TrainingConfig, DataConfig, get_optimizer, get_scheduler
from organoidreader.training.dataset import create_data_loaders
from organoidreader.training.metrics import SegmentationMetrics, LossTracker, MetricsLogger

logger = logging.getLogger(__name__)


class TrainingEngine:
    """
    Main training engine for organoid segmentation models.
    
    Handles the complete training pipeline including model initialization,
    data loading, training loops, validation, and checkpointing.
    """
    
    def __init__(self, 
                 train_config: TrainingConfig,
                 data_config: DataConfig,
                 model: Optional[nn.Module] = None):
        """
        Initialize training engine.
        
        Args:
            train_config: Training configuration
            data_config: Data configuration  
            model: Pre-initialized model (optional)
        """
        self.train_config = train_config
        self.data_config = data_config
        
        # Setup device
        self.device = self._setup_device()
        
        # Initialize model
        if model is not None:
            self.model = model
        else:
            self.model = create_unet_model(
                model_type=train_config.model_type,
                in_channels=train_config.input_channels,
                out_channels=train_config.output_channels
            )
        
        self.model = self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = get_optimizer(self.model.parameters(), train_config)
        self.scheduler = get_scheduler(self.optimizer, train_config)
        
        # Initialize loss function
        self.loss_fn = SegmentationLoss(
            dice_weight=train_config.dice_weight,
            bce_weight=train_config.bce_weight
        )
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if train_config.use_amp else None
        
        # Initialize metrics and logging
        self.metrics_logger = MetricsLogger(train_config.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_dice = 0.0
        self.patience_counter = 0
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(train_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TrainingEngine initialized on device: {self.device}")
        logger.info(f"Model: {train_config.model_type}, Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.train_config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.train_config.device)
        
        logger.info(f"Using device: {device}")
        return device
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader) -> Dict[str, Any]:
        """
        Run complete training process.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting training process")
        start_time = time.time()
        
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        try:
            for epoch in range(self.current_epoch, self.train_config.num_epochs):
                self.current_epoch = epoch
                
                logger.info(f"\nEpoch {epoch + 1}/{self.train_config.num_epochs}")
                logger.info("-" * 50)
                
                # Training phase
                train_loss, train_metrics = self._train_epoch(train_loader)
                training_history['train_losses'].append(train_loss)
                training_history['train_metrics'].append(train_metrics)
                
                # Log training metrics
                self.metrics_logger.log_metrics(
                    {'loss': train_loss, **train_metrics}, 
                    epoch, 
                    'train'
                )
                
                # Validation phase
                if (epoch + 1) % self.train_config.validation_frequency == 0:
                    val_loss, val_metrics = self._validate_epoch(val_loader)
                    training_history['val_losses'].append(val_loss)
                    training_history['val_metrics'].append(val_metrics)
                    
                    # Log validation metrics
                    self.metrics_logger.log_metrics(
                        {'loss': val_loss, **val_metrics}, 
                        epoch, 
                        'val'
                    )
                    
                    # Check for improvement
                    current_val_dice = val_metrics.get('dice_score', 0.0)
                    
                    if current_val_dice > self.best_val_dice:
                        self.best_val_dice = current_val_dice
                        self.patience_counter = 0
                        
                        # Save best model
                        if self.train_config.save_best_only:
                            self._save_checkpoint(epoch, is_best=True)
                    else:
                        self.patience_counter += 1
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if len(training_history['val_losses']) > 0:
                            self.scheduler.step(training_history['val_losses'][-1])
                    else:
                        self.scheduler.step()
                
                # Save checkpoint periodically
                if (epoch + 1) % self.train_config.save_frequency == 0:
                    self._save_checkpoint(epoch)
                
                # Early stopping check
                if self.patience_counter >= self.train_config.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
                
                # Print epoch summary
                self._print_epoch_summary(epoch, train_loss, train_metrics, 
                                        training_history.get('val_losses', [None])[-1], 
                                        training_history.get('val_metrics', [{}])[-1])
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            # Save final checkpoint
            self._save_checkpoint(self.current_epoch, is_final=True)
            
            # Close logger
            self.metrics_logger.close()
        
        training_time = time.time() - start_time
        
        results = {
            'total_epochs': self.current_epoch + 1,
            'best_val_dice': self.best_val_dice,
            'training_time': training_time,
            'history': training_history
        }
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best validation Dice score: {self.best_val_dice:.4f}")
        
        return results
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Run one training epoch."""
        self.model.train()
        
        loss_tracker = LossTracker()
        metrics_calculator = SegmentationMetrics()
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    predictions = self.model(images)
                    loss = self.loss_fn(predictions, masks)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(images)
                loss = self.loss_fn(predictions, masks)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            loss_tracker.update(loss.item(), images.size(0))
            
            with torch.no_grad():
                metrics_calculator.update(predictions, masks)
            
            # Log batch progress
            if batch_idx % 10 == 0:
                logger.debug(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = loss_tracker.compute()
        metrics = metrics_calculator.compute()
        
        return avg_loss, metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Run one validation epoch."""
        self.model.eval()
        
        loss_tracker = LossTracker()
        metrics_calculator = SegmentationMetrics()
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                loss = self.loss_fn(predictions, masks)
                
                # Update metrics
                loss_tracker.update(loss.item(), images.size(0))
                metrics_calculator.update(predictions, masks)
        
        avg_loss = loss_tracker.compute()
        metrics = metrics_calculator.compute()
        
        return avg_loss, metrics
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_dice': self.best_val_dice,
            'train_config': self.train_config,
            'data_config': self.data_config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pth"
            logger.info(f"Saving best model checkpoint: {checkpoint_path}")
        elif is_final:
            checkpoint_path = self.checkpoint_dir / "final_model.pth"
            logger.info(f"Saving final model checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> int:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            
        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load scaler state
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch + 1}")
        
        return self.current_epoch
    
    def _print_epoch_summary(self, epoch: int, train_loss: float, train_metrics: Dict[str, float],
                           val_loss: Optional[float], val_metrics: Dict[str, float]):
        """Print epoch summary."""
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_metrics.get('dice_score', 0):.4f}")
        
        if val_loss is not None:
            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_metrics.get('dice_score', 0):.4f}")
        
        if self.scheduler is not None:
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.6f}")
        
        print(f"Best Val Dice: {self.best_val_dice:.4f}")


def train_model(train_config: TrainingConfig, 
               data_config: DataConfig,
               model: Optional[nn.Module] = None) -> Dict[str, Any]:
    """
    Convenience function to train a model.
    
    Args:
        train_config: Training configuration
        data_config: Data configuration
        model: Optional pre-initialized model
        
    Returns:
        Training results dictionary
    """
    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_config, data_config)
    
    # Create training engine
    engine = TrainingEngine(train_config, data_config, model)
    
    # Run training
    results = engine.train(train_loader, val_loader)
    
    return results


def resume_training(checkpoint_path: str,
                   train_config: Optional[TrainingConfig] = None,
                   data_config: Optional[DataConfig] = None) -> Dict[str, Any]:
    """
    Resume training from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        train_config: Optional new training config
        data_config: Optional new data config
        
    Returns:
        Training results dictionary
    """
    # Load checkpoint to get configs if not provided
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if train_config is None:
        train_config = checkpoint['train_config']
    if data_config is None:
        data_config = checkpoint['data_config']
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_config, data_config)
    
    # Create training engine
    engine = TrainingEngine(train_config, data_config)
    
    # Load checkpoint
    engine.load_checkpoint(checkpoint_path)
    
    # Resume training
    results = engine.train(train_loader, val_loader)
    
    return results