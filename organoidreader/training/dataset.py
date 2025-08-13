"""
Dataset Module

Handles dataset loading, preprocessing, and augmentation for organoid segmentation training.
Supports various image formats and provides PyTorch DataLoader integration.
"""

import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

from organoidreader.core.image_loader import ImageLoader, ImageLoadError
from organoidreader.training.config import DataConfig, TrainingConfig

logger = logging.getLogger(__name__)


class OrganoidDataset(Dataset):
    """
    Dataset class for organoid segmentation training.
    
    Handles loading of images and corresponding segmentation masks,
    with support for data augmentation and preprocessing.
    """
    
    def __init__(self, 
                 images_dir: str,
                 masks_dir: str,
                 config: DataConfig,
                 transform: Optional[Callable] = None,
                 is_training: bool = True):
        """
        Initialize dataset.
        
        Args:
            images_dir: Directory containing training images
            masks_dir: Directory containing segmentation masks
            config: Data configuration
            transform: Optional transform function
            is_training: Whether this is training data (affects augmentation)
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.config = config
        self.transform = transform
        self.is_training = is_training
        
        self.image_loader = ImageLoader()
        
        # Find all image files
        self.image_files = self._find_image_files()
        
        # Validate that corresponding masks exist
        self.valid_pairs = self._validate_image_mask_pairs()
        
        logger.info(f"Dataset initialized: {len(self.valid_pairs)} image-mask pairs")
        
        if len(self.valid_pairs) == 0:
            raise ValueError("No valid image-mask pairs found!")
    
    def _find_image_files(self) -> List[Path]:
        """Find all valid image files in the images directory."""
        image_files = []
        
        for ext in self.config.image_extensions:
            pattern = f"*{ext}"
            files = list(self.images_dir.glob(pattern))
            image_files.extend(files)
        
        return sorted(image_files)
    
    def _validate_image_mask_pairs(self) -> List[Tuple[Path, Path]]:
        """Validate that each image has a corresponding mask."""
        valid_pairs = []
        
        for image_file in self.image_files:
            # Look for corresponding mask file
            mask_file = None
            
            for ext in self.config.mask_extensions:
                potential_mask = self.masks_dir / f"{image_file.stem}{ext}"
                if potential_mask.exists():
                    mask_file = potential_mask
                    break
            
            if mask_file is not None:
                valid_pairs.append((image_file, mask_file))
            else:
                logger.warning(f"No mask found for image: {image_file}")
        
        return valid_pairs
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.valid_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of sample to retrieve
            
        Returns:
            Dictionary containing image and mask tensors
        """
        image_path, mask_path = self.valid_pairs[idx]
        
        try:
            # Load image
            image, _ = self.image_loader.load_image(image_path)
            
            # Load mask
            mask, _ = self.image_loader.load_image(mask_path)
            
            # Convert to proper format
            image = self._prepare_image(image)
            mask = self._prepare_mask(mask)
            
            # Apply transforms if provided
            if self.transform is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            
            # Convert to tensors
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).float()
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).float()
            
            # Ensure proper dimensions
            if len(image.shape) == 2:
                image = image.unsqueeze(0)  # Add channel dimension
            elif len(image.shape) == 3 and image.shape[2] in [1, 3]:
                image = image.permute(2, 0, 1)  # HWC -> CHW
            
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)  # Add channel dimension
            elif len(mask.shape) == 3:
                if mask.shape[2] == 1:
                    mask = mask.permute(2, 0, 1)  # HWC -> CHW
                else:
                    mask = mask[:, :, 0].unsqueeze(0)  # Take first channel
            
            return {
                'image': image,
                'mask': mask,
                'image_path': str(image_path),
                'mask_path': str(mask_path)
            }
            
        except Exception as e:
            logger.error(f"Error loading sample {idx} ({image_path}): {e}")
            # Return a dummy sample to prevent training interruption
            dummy_image = torch.zeros((1, *self.config.image_size))
            dummy_mask = torch.zeros((1, *self.config.image_size))
            
            return {
                'image': dummy_image,
                'mask': dummy_mask,
                'image_path': str(image_path),
                'mask_path': str(mask_path)
            }
    
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for training."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:  # RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                image = image[:, :, 0]  # Take first channel
        
        # Normalize to [0, 1] range
        if image.max() > 1.0:
            image = image.astype(np.float32) / image.max()
        else:
            image = image.astype(np.float32)
        
        # Resize if needed
        if image.shape[:2] != self.config.image_size:
            image = cv2.resize(image, self.config.image_size, interpolation=cv2.INTER_LINEAR)
        
        return image
    
    def _prepare_mask(self, mask: np.ndarray) -> np.ndarray:
        """Prepare mask for training."""
        # Convert to grayscale if needed
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]  # Take first channel
        
        # Resize if needed
        if mask.shape[:2] != self.config.image_size:
            mask = cv2.resize(mask, self.config.image_size, interpolation=cv2.INTER_NEAREST)
        
        # Binarize mask
        mask = (mask > 0.5).astype(np.float32)
        
        return mask


class AugmentationPipeline:
    """Handles data augmentation for training."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize augmentation pipeline.
        
        Args:
            config: Training configuration with augmentation parameters
        """
        self.config = config
    
    def get_training_transforms(self) -> A.Compose:
        """Get training augmentation pipeline."""
        if not self.config.use_augmentation:
            return A.Compose([
                A.Normalize(mean=[0.0], std=[1.0]),
                ToTensorV2()
            ])
        
        transforms_list = [
            # Geometric transforms
            A.Rotate(
                limit=self.config.rotation_range,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT,
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            # Intensity transforms
            A.RandomBrightnessContrast(
                brightness_limit=self.config.brightness_range,
                contrast_limit=self.config.contrast_range,
                p=0.5
            ),
            A.GaussNoise(
                var_limit=(0, self.config.noise_std**2),
                p=0.3
            ),
            
            # Additional augmentations
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT,
                p=0.3
            ),
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.1,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT,
                p=0.3
            ),
            
            # Normalization and tensor conversion
            A.Normalize(mean=[0.0], std=[1.0]),
            ToTensorV2()
        ]
        
        return A.Compose(transforms_list)
    
    def get_validation_transforms(self) -> A.Compose:
        """Get validation transforms (no augmentation)."""
        return A.Compose([
            A.Normalize(mean=[0.0], std=[1.0]),
            ToTensorV2()
        ])


def create_data_loaders(train_config: TrainingConfig, 
                       data_config: DataConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_config: Training configuration
        data_config: Data configuration
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Set random seeds for reproducibility
    torch.manual_seed(train_config.random_seed)
    np.random.seed(train_config.random_seed)
    random.seed(train_config.random_seed)
    
    # Create augmentation pipeline
    aug_pipeline = AugmentationPipeline(train_config)
    train_transforms = aug_pipeline.get_training_transforms()
    val_transforms = aug_pipeline.get_validation_transforms()
    
    # Create datasets
    train_dataset = OrganoidDataset(
        images_dir=data_config.train_images_path,
        masks_dir=data_config.train_masks_path,
        config=data_config,
        transform=train_transforms,
        is_training=True
    )
    
    val_dataset = OrganoidDataset(
        images_dir=data_config.val_images_path,
        masks_dir=data_config.val_masks_path,
        config=data_config,
        transform=val_transforms,
        is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=data_config.shuffle_train,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        drop_last=True  # Drop last batch if incomplete
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=data_config.shuffle_val,
        num_workers=train_config.num_workers,
        pin_memory=train_config.pin_memory,
        drop_last=False
    )
    
    logger.info(f"Created data loaders: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    return train_loader, val_loader


def create_synthetic_dataset(output_dir: str,
                           num_samples: int = 100,
                           image_size: Tuple[int, int] = (512, 512)) -> None:
    """
    Create a synthetic dataset for testing training pipeline.
    
    Args:
        output_dir: Directory to save synthetic data
        num_samples: Number of synthetic samples to generate
        image_size: Size of generated images
    """
    output_path = Path(output_dir)
    
    # Create directories
    train_images_dir = output_path / "train" / "images"
    train_masks_dir = output_path / "train" / "masks"
    val_images_dir = output_path / "val" / "images"
    val_masks_dir = output_path / "val" / "masks"
    
    for dir_path in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Split samples between train and validation
    train_samples = int(num_samples * 0.8)
    val_samples = num_samples - train_samples
    
    logger.info(f"Generating {num_samples} synthetic samples ({train_samples} train, {val_samples} val)")
    
    # Generate training samples
    for i in range(train_samples):
        image, mask = _generate_synthetic_sample(image_size)
        
        # Save image and mask
        cv2.imwrite(str(train_images_dir / f"sample_{i:04d}.png"), (image * 255).astype(np.uint8))
        cv2.imwrite(str(train_masks_dir / f"sample_{i:04d}.png"), (mask * 255).astype(np.uint8))
    
    # Generate validation samples
    for i in range(val_samples):
        image, mask = _generate_synthetic_sample(image_size)
        
        # Save image and mask
        cv2.imwrite(str(val_images_dir / f"sample_{i:04d}.png"), (image * 255).astype(np.uint8))
        cv2.imwrite(str(val_masks_dir / f"sample_{i:04d}.png"), (mask * 255).astype(np.uint8))
    
    logger.info(f"Synthetic dataset created in {output_dir}")


def _generate_synthetic_sample(image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a single synthetic image-mask pair."""
    height, width = image_size
    
    # Create background
    image = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.float32)
    
    # Add background gradient
    y, x = np.ogrid[:height, :width]
    background = 0.1 + 0.05 * np.sin(x / width * 2 * np.pi) * np.sin(y / height * 2 * np.pi)
    image += background
    
    # Generate random organoids
    num_organoids = np.random.randint(1, 8)
    
    for _ in range(num_organoids):
        # Random position
        center_x = np.random.randint(width // 6, 5 * width // 6)
        center_y = np.random.randint(height // 6, 5 * height // 6)
        
        # Random size
        radius = np.random.randint(15, 50)
        
        # Create organoid
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        organoid_mask = distance <= radius
        
        # Add to image with intensity profile
        intensity = np.exp(-(distance / radius)**2) * 0.6
        intensity[distance > radius] = 0
        image += intensity
        
        # Add to mask
        mask[organoid_mask] = 1.0
    
    # Add noise
    noise = np.random.normal(0, 0.05, image.shape)
    image += noise
    
    # Clip values
    image = np.clip(image, 0, 1)
    
    return image, mask


# Utility functions
def validate_dataset_structure(data_dir: str) -> bool:
    """
    Validate that dataset has the expected structure.
    
    Args:
        data_dir: Root directory of dataset
        
    Returns:
        True if structure is valid
    """
    data_path = Path(data_dir)
    
    required_dirs = [
        "train/images", "train/masks",
        "val/images", "val/masks"
    ]
    
    for dir_name in required_dirs:
        dir_path = data_path / dir_name
        if not dir_path.exists():
            logger.error(f"Missing directory: {dir_path}")
            return False
        
        # Check if directory has files
        files = list(dir_path.iterdir())
        if len(files) == 0:
            logger.warning(f"Empty directory: {dir_path}")
    
    logger.info("Dataset structure validation passed")
    return True


def calculate_dataset_statistics(data_loader: DataLoader) -> Dict[str, float]:
    """
    Calculate dataset statistics for normalization.
    
    Args:
        data_loader: DataLoader to analyze
        
    Returns:
        Dictionary with mean and std statistics
    """
    mean = 0.0
    std = 0.0
    total_samples = 0
    
    for batch in data_loader:
        images = batch['image']
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'total_samples': total_samples
    }