"""
Training Configuration Module

Defines training-specific configurations and hyperparameters
for organoid segmentation model training.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch.optim as optim


@dataclass
class TrainingConfig:
    """Configuration for model training parameters."""
    
    # Model settings
    model_type: str = "attention"  # "standard", "attention", "multiscale"
    input_channels: int = 1
    output_channels: int = 1
    
    # Training hyperparameters
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Optimizer settings
    optimizer_type: str = "adam"  # "adam", "sgd", "adamw"
    momentum: float = 0.9  # For SGD
    beta1: float = 0.9     # For Adam
    beta2: float = 0.999   # For Adam
    
    # Loss function settings
    dice_weight: float = 0.7
    bce_weight: float = 0.3
    
    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # "step", "cosine", "plateau"
    step_size: int = 20
    gamma: float = 0.5
    patience: int = 10  # For plateau scheduler
    
    # Data augmentation
    use_augmentation: bool = True
    rotation_range: float = 15.0
    brightness_range: float = 0.2
    contrast_range: float = 0.2
    noise_std: float = 0.01
    
    # Validation settings
    validation_split: float = 0.2
    validation_frequency: int = 5  # Every N epochs
    
    # Checkpointing
    save_frequency: int = 10  # Every N epochs
    save_best_only: bool = True
    early_stopping_patience: int = 20
    
    # Hardware settings
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    num_workers: int = 4
    pin_memory: bool = True
    
    # Paths
    dataset_path: str = "data/training"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Metrics to track
    metrics_to_track: List[str] = field(default_factory=lambda: [
        "dice_score", "iou", "precision", "recall", "f1_score"
    ])
    
    # Mixed precision training
    use_amp: bool = True  # Automatic Mixed Precision
    
    # Reproducibility
    random_seed: int = 42


@dataclass  
class DataConfig:
    """Configuration for dataset handling."""
    
    # Dataset paths
    train_images_path: str = "data/train/images"
    train_masks_path: str = "data/train/masks"
    val_images_path: str = "data/val/images"
    val_masks_path: str = "data/val/masks"
    
    # Image preprocessing
    image_size: tuple = (512, 512)
    normalize_images: bool = True
    normalize_mean: List[float] = field(default_factory=lambda: [0.485])
    normalize_std: List[float] = field(default_factory=lambda: [0.229])
    
    # Data loading
    cache_dataset: bool = False  # Cache preprocessed data in memory
    shuffle_train: bool = True
    shuffle_val: bool = False
    
    # File extensions
    image_extensions: List[str] = field(default_factory=lambda: [
        ".jpg", ".jpeg", ".png", ".tiff", ".tif"
    ])
    mask_extensions: List[str] = field(default_factory=lambda: [
        ".png", ".tiff", ".tif"
    ])


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    
    # U-Net specific settings
    features: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    use_attention: bool = True
    dropout_rate: float = 0.1
    
    # Multi-scale U-Net settings
    scales: List[float] = field(default_factory=lambda: [1.0, 0.75, 0.5])
    
    # Model initialization
    init_type: str = "kaiming"  # "kaiming", "xavier", "normal"
    init_gain: float = 0.02


def get_optimizer(model_parameters, config: TrainingConfig):
    """
    Create optimizer based on configuration.
    
    Args:
        model_parameters: Model parameters to optimize
        config: Training configuration
        
    Returns:
        Configured optimizer
    """
    if config.optimizer_type.lower() == "adam":
        return optim.Adam(
            model_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
    elif config.optimizer_type.lower() == "adamw":
        return optim.AdamW(
            model_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )
    elif config.optimizer_type.lower() == "sgd":
        return optim.SGD(
            model_parameters,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer_type}")


def get_scheduler(optimizer, config: TrainingConfig):
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer instance
        config: Training configuration
        
    Returns:
        Configured scheduler or None
    """
    if not config.use_scheduler:
        return None
    
    if config.scheduler_type.lower() == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma
        )
    elif config.scheduler_type.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs
        )
    elif config.scheduler_type.lower() == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=config.patience,
            factor=config.gamma,
            verbose=True
        )
    else:
        raise ValueError(f"Unsupported scheduler: {config.scheduler_type}")


def create_training_config(**kwargs) -> TrainingConfig:
    """
    Create training configuration with custom overrides.
    
    Args:
        **kwargs: Configuration overrides
        
    Returns:
        TrainingConfig instance
    """
    config = TrainingConfig()
    
    # Update with provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")
    
    return config