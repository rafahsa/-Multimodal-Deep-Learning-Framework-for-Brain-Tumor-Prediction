"""
Swin UNETR Encoder for 3D Brain Tumor Classification

This module provides a Swin UNETR-based classification model that uses only the encoder
portion of Swin UNETR, removing the segmentation decoder and adding a classification head.

Architecture:
    Multi-modal 3D input (4 channels: T1, T1ce, T2, FLAIR)
    → Swin UNETR Encoder (Transformer-based feature extraction)
    → Global Pooling (Adaptive Average Pooling)
    → Classification Head (FC layer + Dropout)
    → Binary classification logits (LGG vs HGG)

Author: Medical Imaging Pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from monai.networks.nets import SwinUNETR
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("Warning: MONAI not installed. Install with: pip install monai")


class SwinUNETREncoderClassifier(nn.Module):
    """
    Swin UNETR Encoder-based Classifier for Brain Tumor Grading.
    
    Uses Swin UNETR encoder (swinViT) to extract features, then applies global pooling
    and a classification head for binary classification (HGG vs LGG).
    
    Args:
        img_size: Input image size (D, H, W). Default: (128, 128, 128)
        in_channels: Number of input channels. Default: 4 (multi-modal)
        num_classes: Number of output classes. Default: 2 (binary classification)
        patch_size: Patch size for Swin Transformer. Default: 2
        feature_size: Base feature size. Default: 48 (memory-efficient)
        depths: Number of layers in each stage. Default: [2, 2, 2, 2]
        num_heads: Number of attention heads in each stage. Default: [3, 6, 12, 24]
        window_size: Window size for Swin Transformer. Default: 7
        dropout: Dropout rate in classification head. Default: 0.4
        use_checkpoint: Enable gradient checkpointing for memory efficiency. Default: False
        use_hidden_layer: Whether to use hidden layer in classification head. Default: False
    """
    
    def __init__(
        self,
        img_size: tuple = (128, 128, 128),
        in_channels: int = 4,
        num_classes: int = 2,
        patch_size: int = 2,
        feature_size: int = 48,
        depths: list = [2, 2, 2, 2],
        num_heads: list = [3, 6, 12, 24],
        window_size: int = 7,
        dropout: float = 0.4,
        use_checkpoint: bool = False,
        use_hidden_layer: bool = False,
        logger=None
    ):
        super().__init__()
        
        if not MONAI_AVAILABLE:
            raise ImportError(
                "MONAI is required for Swin UNETR. Install with: pip install monai"
            )
        
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.depths = depths
        self.num_heads = num_heads
        
        # Create Swin UNETR model (we'll use only the encoder)
        # Note: We create full SwinUNETR but only use encoder part
        self.swin_unetr = SwinUNETR(
            in_channels=in_channels,
            out_channels=feature_size,  # Not used for classification, but required
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            feature_size=feature_size,
            norm_name="instance",
            drop_rate=0.0,  # Dropout handled in classification head
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            normalize=True,
            use_checkpoint=use_checkpoint,
            spatial_dims=3
        )
        
        # Extract encoder (swinViT) from SwinUNETR
        # SwinUNETR uses swinViT as its encoder
        self.encoder = self.swin_unetr.swinViT
        
        # Calculate encoder output dimension
        # SwinTransformer outputs features at multiple resolutions
        # The deepest (final) feature map has dimension: feature_size * (2 ** num_stages)
        # Swin UNETR has 4 stages (depths), so final feature dim = feature_size * 16
        # Actually, SwinTransformer outputs 5 feature maps (including initial), 
        # so final dim = feature_size * (2 ** len(depths)) = feature_size * 16
        num_stages = len(depths)
        self.encoder_output_dim = feature_size * (2 ** num_stages)
        
        if logger:
            logger.info(f"Swin UNETR Encoder output dimension: {self.encoder_output_dim}")
        
        # Global pooling: Adaptive Average Pooling
        # Reduces spatial dimensions to (1, 1, 1) regardless of input size
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification head
        if use_hidden_layer:
            # Two-layer head: encoder_dim -> hidden_dim -> num_classes
            hidden_dim = 256
            self.classifier = nn.Sequential(
                nn.Linear(self.encoder_output_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            # Single-layer head: encoder_dim -> num_classes
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.encoder_output_dim, num_classes)
            )
        
        if logger:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logger.info(f"Swin UNETR Encoder Classifier: {total_params/1e6:.2f}M total params, "
                       f"{trainable_params/1e6:.2f}M trainable")
            logger.info(f"Classification head: {'Two-layer' if use_hidden_layer else 'Single-layer'}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, D, H, W)
               Expected: (B, 4, 128, 128, 128) for multi-modal BraTS
        
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Encoder forward pass
        # SwinTransformer (swinViT) returns features at multiple resolutions
        # We use the deepest (final) feature map
        encoder_features = self.encoder(x)
        
        # encoder_features is a list of feature maps at different resolutions
        # Use the deepest (last) feature map for classification
        if isinstance(encoder_features, (list, tuple)):
            final_features = encoder_features[-1]  # Deepest resolution: (B, C, D', H', W')
        else:
            final_features = encoder_features  # Single output: (B, C, D', H', W')
        
        # Global pooling: (B, C, D', H', W') -> (B, C, 1, 1, 1)
        pooled = self.global_pool(final_features)
        
        # Flatten: (B, C, 1, 1, 1) -> (B, C)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Classification: (B, C) -> (B, num_classes)
        logits = self.classifier(pooled)
        
        return logits
    
    def get_backbone_params(self):
        """
        Get parameters from the encoder (backbone).
        
        Returns:
            Generator of encoder parameters
        """
        for name, param in self.named_parameters():
            if 'classifier' not in name:
                yield param
    
    def get_classifier_params(self):
        """
        Get parameters from the classification head.
        
        Returns:
            Generator of classifier parameters
        """
        for name, param in self.named_parameters():
            if 'classifier' in name:
                yield param


# Convenience function for model creation (matches ResNet50-3D interface)
def create_swin_unetr_classifier(
    num_classes: int = 2,
    in_channels: int = 4,
    img_size: tuple = (128, 128, 128),
    feature_size: int = 48,
    depths: list = [2, 2, 2, 2],
    num_heads: list = [3, 6, 12, 24],
    dropout: float = 0.4,
    use_checkpoint: bool = False,
    use_hidden_layer: bool = False,
    logger=None
) -> SwinUNETREncoderClassifier:
    """
    Create Swin UNETR Encoder Classifier.
    
    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels
        img_size: Input image size
        feature_size: Base feature size (48 for memory efficiency)
        depths: Number of layers per stage
        num_heads: Number of attention heads per stage
        dropout: Dropout rate in classification head
        use_checkpoint: Enable gradient checkpointing
        use_hidden_layer: Whether to use hidden layer in classifier
        logger: Optional logger for information
        
    Returns:
        SwinUNETREncoderClassifier instance
    """
    return SwinUNETREncoderClassifier(
        img_size=img_size,
        in_channels=in_channels,
        num_classes=num_classes,
        feature_size=feature_size,
        depths=depths,
        num_heads=num_heads,
        dropout=dropout,
        use_checkpoint=use_checkpoint,
        use_hidden_layer=use_hidden_layer,
        logger=logger
    )


if __name__ == "__main__":
    """
    Test Swin UNETR Encoder Classifier.
    """
    print("Testing Swin UNETR Encoder Classifier")
    print("=" * 60)
    
    if not MONAI_AVAILABLE:
        print("ERROR: MONAI not installed.")
        print("Install with: pip install monai")
        exit(1)
    
    # Create model
    model = SwinUNETREncoderClassifier(
        img_size=(128, 128, 128),
        in_channels=4,
        num_classes=2,
        feature_size=48,  # Smaller for memory efficiency
        use_checkpoint=False
    )
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 4, 128, 128, 128)
    
    print(f"Input shape: {x.shape}")
    
    model.eval()
    with torch.no_grad():
        logits = model(x)
    
    print(f"Output shape: {logits.shape}")
    print(f"Expected: ({batch_size}, 2)")
    
    # Verify output shape
    assert logits.shape == (batch_size, 2), f"Expected ({batch_size}, 2), got {logits.shape}"
    
    # Test parameter groups
    backbone_params = list(model.get_backbone_params())
    classifier_params = list(model.get_classifier_params())
    
    print(f"\nBackbone parameters: {sum(p.numel() for p in backbone_params)/1e6:.2f}M")
    print(f"Classifier parameters: {sum(p.numel() for p in classifier_params)/1e6:.2f}M")
    
    print("\n✓ Model test passed!")

