"""
Training Script

Main script for training organoid segmentation models.
Provides command-line interface and example usage.
"""

import argparse
import logging
import yaml
from pathlib import Path
import torch

from organoidreader.training.config import TrainingConfig, DataConfig
from organoidreader.training.trainer import train_model
from organoidreader.training.dataset import create_synthetic_dataset, validate_dataset_structure
from organoidreader.utils.logging_setup import setup_logging


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train OrganoidReader segmentation model')
    
    # Configuration arguments
    parser.add_argument('--config', type=str, help='Path to training config YAML file')
    parser.add_argument('--data-dir', type=str, default='data', help='Root directory of dataset')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory for logs')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model-type', type=str, default='attention', 
                       choices=['standard', 'attention', 'multiscale'], 
                       help='Type of U-Net model')
    
    # Data parameters
    parser.add_argument('--image-size', type=int, nargs=2, default=[512, 512], help='Input image size')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda', 'mps'], help='Device to use')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    
    # Utilities
    parser.add_argument('--create-synthetic-data', action='store_true', 
                       help='Create synthetic dataset for testing')
    parser.add_argument('--num-synthetic-samples', type=int, default=100, 
                       help='Number of synthetic samples to create')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--validate-data', action='store_true', help='Validate dataset structure')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Create synthetic dataset if requested
        if args.create_synthetic_data:
            logger.info("Creating synthetic dataset...")
            create_synthetic_dataset(
                output_dir=args.data_dir,
                num_samples=args.num_synthetic_samples,
                image_size=tuple(args.image_size)
            )
            logger.info("Synthetic dataset created successfully")
            return
        
        # Validate dataset structure if requested
        if args.validate_data:
            logger.info("Validating dataset structure...")
            if validate_dataset_structure(args.data_dir):
                logger.info("Dataset structure is valid")
            else:
                logger.error("Dataset structure validation failed")
                return
        
        # Load configuration
        if args.config and Path(args.config).exists():
            logger.info(f"Loading configuration from {args.config}")
            with open(args.config, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            train_config = TrainingConfig(**config_dict.get('training', {}))
            data_config = DataConfig(**config_dict.get('data', {}))
        else:
            logger.info("Using default configuration")
            train_config = TrainingConfig()
            data_config = DataConfig()
        
        # Override with command line arguments
        train_config.num_epochs = args.epochs
        train_config.batch_size = args.batch_size
        train_config.learning_rate = args.learning_rate
        train_config.model_type = args.model_type
        train_config.device = args.device
        train_config.num_workers = args.num_workers
        train_config.checkpoint_dir = args.checkpoint_dir
        train_config.log_dir = args.log_dir
        
        data_config.image_size = tuple(args.image_size)
        
        # Set data paths
        data_root = Path(args.data_dir)
        data_config.train_images_path = str(data_root / "train" / "images")
        data_config.train_masks_path = str(data_root / "train" / "masks")
        data_config.val_images_path = str(data_root / "val" / "images")
        data_config.val_masks_path = str(data_root / "val" / "masks")
        
        # Create directories
        Path(train_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(train_config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Print configuration
        logger.info("Training Configuration:")
        logger.info(f"  Model: {train_config.model_type}")
        logger.info(f"  Epochs: {train_config.num_epochs}")
        logger.info(f"  Batch Size: {train_config.batch_size}")
        logger.info(f"  Learning Rate: {train_config.learning_rate}")
        logger.info(f"  Device: {train_config.device}")
        logger.info(f"  Image Size: {data_config.image_size}")
        
        # Resume training if checkpoint provided
        if args.resume:
            logger.info(f"Resuming training from {args.resume}")
            from organoidreader.training.trainer import resume_training
            results = resume_training(args.resume, train_config, data_config)
        else:
            # Start new training
            logger.info("Starting new training...")
            results = train_model(train_config, data_config)
        
        # Print results
        logger.info("Training Results:")
        logger.info(f"  Total Epochs: {results['total_epochs']}")
        logger.info(f"  Best Validation Dice: {results['best_val_dice']:.4f}")
        logger.info(f"  Training Time: {results['training_time']:.2f} seconds")
        
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def create_training_config_template(output_path: str):
    """Create a template training configuration file."""
    template_config = {
        'training': {
            'model_type': 'attention',
            'batch_size': 8,
            'num_epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'optimizer_type': 'adam',
            'use_scheduler': True,
            'scheduler_type': 'cosine',
            'use_augmentation': True,
            'validation_frequency': 5,
            'save_frequency': 10,
            'early_stopping_patience': 20,
            'use_amp': True,
            'random_seed': 42
        },
        'data': {
            'train_images_path': 'data/train/images',
            'train_masks_path': 'data/train/masks',
            'val_images_path': 'data/val/images',
            'val_masks_path': 'data/val/masks',
            'image_size': [512, 512],
            'normalize_images': True,
            'cache_dataset': False,
            'shuffle_train': True
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(template_config, f, default_flow_style=False, indent=2)
    
    print(f"Training configuration template saved to {output_path}")


if __name__ == "__main__":
    main()