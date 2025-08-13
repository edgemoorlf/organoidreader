"""
U-Net Model Implementation for Organoid Segmentation

This module implements a U-Net architecture optimized for organoid segmentation
with configurable depth, attention mechanisms, and multi-scale processing.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0):
        """
        Initialize double convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dropout_rate: Dropout rate for regularization
        """
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through double convolution."""
        return self.double_conv(x)


class AttentionGate(nn.Module):
    """Attention gate for improved feature selection."""
    
    def __init__(self, gate_channels: int, in_channels: int, inter_channels: int):
        """
        Initialize attention gate.
        
        Args:
            gate_channels: Number of channels in gate signal
            in_channels: Number of input channels
            inter_channels: Number of intermediate channels
        """
        super(AttentionGate, self).__init__()
        
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention gate.
        
        Args:
            gate: Gate signal from deeper layer
            x: Input feature map
            
        Returns:
            Attention-weighted feature map
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class UNet(nn.Module):
    """
    U-Net architecture for organoid segmentation.
    
    Features:
    - Configurable depth and channels
    - Optional attention gates
    - Multi-scale feature processing
    - Dropout for regularization
    """
    
    def __init__(self, 
                 in_channels: int = 1, 
                 out_channels: int = 1,
                 features: List[int] = None,
                 use_attention: bool = True,
                 dropout_rate: float = 0.1):
        """
        Initialize U-Net model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            features: List of feature dimensions for each level
            use_attention: Whether to use attention gates
            dropout_rate: Dropout rate for regularization
        """
        super(UNet, self).__init__()
        
        if features is None:
            features = [64, 128, 256, 512, 1024]
        
        self.use_attention = use_attention
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder (down-sampling path)
        in_ch = in_channels
        for feature in features:
            self.encoder.append(DoubleConv(in_ch, feature, dropout_rate))
            in_ch = feature
        
        # Decoder (up-sampling path)
        for feature in reversed(features[:-1]):
            self.decoder.append(
                nn.ConvTranspose2d(features[-1], feature, kernel_size=2, stride=2)
            )
            
            if use_attention:
                self.decoder.append(
                    AttentionGate(feature, feature, feature // 2)
                )
            
            self.decoder.append(
                DoubleConv(feature * 2, feature, dropout_rate)
            )
            features[-1] = feature
        
        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Segmentation output tensor
        """
        skip_connections = []
        
        # Encoder path
        for encode in self.encoder:
            x = encode(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        skip_connections = skip_connections[:-1]  # Remove last for bottleneck
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        # Decoder path
        idx = 0
        for i in range(0, len(self.decoder), 3 if self.use_attention else 2):
            # Transpose convolution
            x = self.decoder[i](x)
            
            # Get skip connection
            skip_connection = skip_connections[idx]
            
            # Handle size mismatch
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            
            if self.use_attention:
                # Apply attention gate
                attention_gate = self.decoder[i + 1]
                skip_connection = attention_gate(x, skip_connection)
                
                # Double convolution
                concat_skip = torch.cat((skip_connection, x), dim=1)
                x = self.decoder[i + 2](concat_skip)
            else:
                # Double convolution without attention
                concat_skip = torch.cat((skip_connection, x), dim=1)
                x = self.decoder[i + 1](concat_skip)
            
            idx += 1
        
        # Final output
        output = self.final_conv(x)
        return output


class MultiScaleUNet(nn.Module):
    """
    Multi-scale U-Net for improved organoid segmentation.
    
    Processes images at multiple scales and combines predictions
    for better detection of organoids of varying sizes.
    """
    
    def __init__(self, 
                 in_channels: int = 1, 
                 out_channels: int = 1,
                 scales: List[float] = None):
        """
        Initialize multi-scale U-Net.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels  
            scales: List of scale factors for multi-scale processing
        """
        super(MultiScaleUNet, self).__init__()
        
        if scales is None:
            scales = [1.0, 0.75, 0.5]
        
        self.scales = scales
        self.base_unet = UNet(in_channels, out_channels, use_attention=True)
        
        # Scale fusion network
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(scales), out_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale U-Net.
        
        Args:
            x: Input tensor
            
        Returns:
            Fused segmentation output
        """
        original_size = x.shape[2:]
        outputs = []
        
        # Process at each scale
        for scale in self.scales:
            if scale == 1.0:
                scaled_x = x
            else:
                # Resize input
                new_size = [int(s * scale) for s in original_size]
                scaled_x = F.interpolate(x, size=new_size, mode='bilinear', align_corners=False)
            
            # Get prediction
            pred = self.base_unet(scaled_x)
            
            # Resize prediction back to original size
            if scale != 1.0:
                pred = F.interpolate(pred, size=original_size, mode='bilinear', align_corners=False)
            
            outputs.append(pred)
        
        # Concatenate and fuse predictions
        fused_input = torch.cat(outputs, dim=1)
        final_output = self.fusion_conv(fused_input)
        
        return final_output


class SegmentationLoss(nn.Module):
    """
    Combined loss function for organoid segmentation.
    
    Combines Dice loss and Binary Cross Entropy for balanced training.
    """
    
    def __init__(self, dice_weight: float = 0.7, bce_weight: float = 0.3):
        """
        Initialize combined loss.
        
        Args:
            dice_weight: Weight for Dice loss
            bce_weight: Weight for BCE loss
        """
        super(SegmentationLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
        """
        Calculate Dice loss.
        
        Args:
            pred: Predicted tensor
            target: Target tensor
            smooth: Smoothing factor
            
        Returns:
            Dice loss value
        """
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        
        return 1 - dice
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            pred: Predicted tensor
            target: Target tensor
            
        Returns:
            Combined loss value
        """
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        
        return self.dice_weight * dice + self.bce_weight * bce


def create_unet_model(model_type: str = "standard",
                     in_channels: int = 1,
                     out_channels: int = 1,
                     **kwargs) -> nn.Module:
    """
    Factory function to create U-Net models.
    
    Args:
        model_type: Type of model ("standard", "attention", "multiscale")
        in_channels: Number of input channels
        out_channels: Number of output channels
        **kwargs: Additional model parameters
        
    Returns:
        U-Net model instance
    """
    if model_type == "standard":
        return UNet(in_channels, out_channels, use_attention=False, **kwargs)
    elif model_type == "attention":
        return UNet(in_channels, out_channels, use_attention=True, **kwargs)
    elif model_type == "multiscale":
        return MultiScaleUNet(in_channels, out_channels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
        'model_type': model.__class__.__name__
    }