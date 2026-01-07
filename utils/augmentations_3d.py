"""
Stage 5: Geometric Data Augmentation for 3D Medical Imaging

This module provides on-the-fly augmentation transforms for 3D MRI volumes.
Augmentation is applied ONLY during training, NOT during validation or testing.

Key Principles:
- Medical-safe: Preserves anatomical plausibility
- On-the-fly: Applied in DataLoader, no files created on disk
- Train-only: No augmentation during validation/testing
- Compatible: Works with MIL (slice-based), ResNet50-3D, Swin UNETR

Requirements:
    pip install monai

Author: Medical Imaging Pipeline
"""

from __future__ import annotations

from typing import Optional, Tuple

try:
    from monai.transforms import (
        Compose,
        RandRotateD,
        RandFlipD,
        RandZoomD,
        RandAffineD,
        ToTensorD,
        EnsureChannelFirstD,
        NormalizeIntensityD,
    )
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("Warning: MONAI not installed. Install with: pip install monai")


def get_train_transforms_3d(
    rotation_range: Tuple[float, float, float] = (0.2, 0.2, 0.2),  # ±11.5 degrees in radians (slightly reduced for stability)
    flip_prob: float = 0.5,
    zoom_range: Tuple[float, float] = (0.92, 1.08),  # ±8% (slightly reduced for stability)
    translation_range: Tuple[float, float, float] = (0.08, 0.08, 0.08),  # ±8% of size (slightly reduced)
    prob: float = 0.6,  # Increased probability for more augmentation
    spatial_dims: int = 3,
    num_channels: int = 1  # Number of input channels (1 for single-modality, 4 for multi-modality)
) -> Compose:
    """
    Get training augmentation transforms for 3D medical imaging.
    
    Applies mild geometric augmentations that preserve anatomical plausibility:
    - Random rotation (±15 degrees)
    - Random flip (x, y, z axes)
    - Random zoom (±10%)
    - Random translation (±10% of volume size)
    
    Args:
        rotation_range: Rotation range in radians for (x, y, z) axes.
                        Default (0.26, 0.26, 0.26) ≈ ±15 degrees
        flip_prob: Probability of applying flip for each axis
        zoom_range: Zoom range (min, max). Default (0.9, 1.1) = ±10%
        translation_range: Translation range as fraction of size for (x, y, z)
        prob: Overall probability of applying augmentation
        spatial_dims: Number of spatial dimensions (3 for 3D)
        
    Returns:
        MONAI Compose transform pipeline
        
    Example:
        >>> transforms = get_train_transforms_3d()
        >>> augmented_volume = transforms(volume_dict)
    """
    if not MONAI_AVAILABLE:
        raise ImportError("MONAI is required for augmentation. Install with: pip install monai")
    
    transforms = []
    
    # Ensure channel dimension is first (required for MONAI)
    # For single-modality (1 channel), add channel dim if missing
    # For multi-modal (4 channels), channels are ALREADY first from dataset (4, D, H, W)
    # DO NOT apply EnsureChannelFirstD for multi-modal - it would incorrectly rearrange dimensions
    if num_channels == 1:
        # Single modality: add channel dim if missing (for 3D: (D,H,W) -> (1,D,H,W))
        transforms.append(EnsureChannelFirstD(keys='image', channel_dim='no_channel'))
    else:
        # Multi-modal: channels are already first (4, D, H, W) from dataset
        # Skip EnsureChannelFirstD to preserve correct channel order
        # MONAI transforms work correctly with channels-first format
        pass
    
    # Add augmentation transforms
    transforms.extend([
        # Random rotation (±15 degrees ≈ 0.26 radians)
        # Rotates around x, y, z axes independently
        RandRotateD(
            keys='image',
            range_x=rotation_range[0],
            range_y=rotation_range[1],
            range_z=rotation_range[2],
            prob=prob,
            keep_size=True,  # Maintain output size
            mode='bilinear',  # Bilinear interpolation for smooth rotation
            padding_mode='constant'  # Constant padding (zeros) for boundaries
        ),
        
        # Random flip along x-axis
        RandFlipD(
            keys='image',
            spatial_axis=0,
            prob=flip_prob
        ),
        
        # Random flip along y-axis
        RandFlipD(
            keys='image',
            spatial_axis=1,
            prob=flip_prob
        ),
        
        # Random flip along z-axis
        RandFlipD(
            keys='image',
            spatial_axis=2,
            prob=flip_prob
        ),
        
        # Random zoom (±10%)
        RandZoomD(
            keys='image',
            min_zoom=zoom_range[0],
            max_zoom=zoom_range[1],
            prob=prob,
            mode='trilinear',  # Trilinear interpolation for 3D
            padding_mode='constant',  # Constant padding (zeros) for boundaries
            keep_size=True  # Crop/pad to maintain size
        ),
        
        # Random translation (±10% of volume size)
        # Combined with rotation for more realistic augmentation
        RandAffineD(
            keys='image',
            prob=prob,
            translate_range=translation_range,
            mode='bilinear',
            padding_mode='constant',  # Constant padding (zeros) for boundaries
            spatial_size=None  # Keep original size
        ),
        
        # Convert to tensor (required for PyTorch)
        ToTensorD(keys='image', dtype=None)  # Preserves original dtype
    ])
    
    return Compose(transforms)


def get_val_transforms_3d(
    normalize: bool = True,
    spatial_dims: int = 3,
    num_channels: int = 1  # Number of input channels (1 for single-modality, 4 for multi-modality)
) -> Compose:
    """
    Get validation/test transforms (NO augmentation).
    
    Only applies:
    - Channel dimension addition (if needed)
    - Tensor conversion
    - Optional normalization (if not already normalized)
    
    Args:
        normalize: Whether to apply intensity normalization
                   (default: True, but Stage 2 already normalized)
        spatial_dims: Number of spatial dimensions (3 for 3D)
        num_channels: Number of input channels (1 for single-modality, 4 for multi-modality)
        
    Returns:
        MONAI Compose transform pipeline (no augmentation)
        
    Example:
        >>> transforms = get_val_transforms_3d()
        >>> volume = transforms(volume_dict)  # No augmentation applied
    """
    if not MONAI_AVAILABLE:
        raise ImportError(
            "MONAI is required for transforms. Install with: pip install monai"
        )
    
    transforms = []
    
    # Ensure channel dimension is first
    # For single-modality (1 channel), add channel dim if missing
    # For multi-modal (4 channels), channels are ALREADY first from dataset (4, D, H, W)
    # DO NOT apply EnsureChannelFirstD for multi-modal - it would incorrectly rearrange dimensions
    if num_channels == 1:
        # Single modality: add channel dim if missing (for 3D: (D,H,W) -> (1,D,H,W))
        transforms.append(EnsureChannelFirstD(keys='image', channel_dim='no_channel'))
    else:
        # Multi-modal: channels are already first (4, D, H, W) from dataset
        # Skip EnsureChannelFirstD to preserve correct channel order
        # MONAI transforms work correctly with channels-first format
        pass
    
    # Optional normalization (usually not needed if Stage 2 already normalized)
    if normalize:
        transforms.append(NormalizeIntensityD(keys='image', subtrahend=0.0, divisor=1.0))
    
    # Convert to tensor
    transforms.append(ToTensorD(keys='image', dtype=None))
    
    return Compose(transforms)


def get_transforms_3d(mode: str = "train", **kwargs) -> Compose:
    """
    Convenience function to get transforms based on mode.
    
    Args:
        mode: "train" or "val"/"test"
        **kwargs: Additional arguments passed to get_train_transforms_3d or get_val_transforms_3d
        
    Returns:
        Appropriate transform pipeline
        
    Example:
        >>> train_transforms = get_transforms_3d("train")
        >>> val_transforms = get_transforms_3d("val")
    """
    if mode.lower() in ["train", "training"]:
        return get_train_transforms_3d(**kwargs)
    elif mode.lower() in ["val", "validation", "test", "testing"]:
        return get_val_transforms_3d(**kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'train' or 'val'/'test'")


# Example usage for different model architectures
def get_mil_transforms_3d(mode: str = "train") -> Compose:
    """
    Get transforms for MIL (Multiple Instance Learning) models.
    
    MIL models work with 2D slices extracted from 3D volumes.
    This function returns 3D transforms that can be applied before slicing,
    or can be adapted for slice-level augmentation.
    
    Args:
        mode: "train" or "val"
        
    Returns:
        Transform pipeline
    """
    if mode.lower() == "train":
        # For MIL, we might want slightly different augmentation
        # since slices are extracted later
        return get_train_transforms_3d(
            rotation_range=(0.2, 0.2, 0.2),  # Slightly less rotation for slices
            flip_prob=0.5,
            zoom_range=(0.95, 1.05),  # Less zoom for slice consistency
            prob=0.5
        )
    else:
        return get_val_transforms_3d()


def get_resnet3d_transforms_3d(mode: str = "train", num_channels: int = 1) -> Compose:
    """
    Get transforms for ResNet50-3D models.
    
    ResNet50-3D processes full 3D volumes, so standard 3D augmentation applies.
    
    Args:
        mode: "train" or "val"
        num_channels: Number of input channels (1 for single-modality, 4 for multi-modality)
        
    Returns:
        Transform pipeline
    """
    if mode == "train":
        return get_train_transforms_3d(num_channels=num_channels)
    elif mode == "val":
        return get_val_transforms_3d(num_channels=num_channels)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'train' or 'val'")


def get_swin_unetr_transforms_3d(mode: str = "train") -> Compose:
    """
    Get transforms for Swin UNETR models.
    
    Swin UNETR also processes full 3D volumes, so standard 3D augmentation applies.
    
    Args:
        mode: "train" or "val"
        
    Returns:
        Transform pipeline
    """
    return get_transforms_3d(mode=mode)


# Medical justification constants
MEDICAL_AUGMENTATION_PARAMS = {
    "rotation_max_degrees": 15,  # ±15 degrees preserves anatomical structure
    "zoom_range_percent": 10,    # ±10% maintains relative proportions
    "translation_percent": 10,   # ±10% keeps brain within field of view
    "flip_probability": 0.5,      # 50% chance per axis (realistic variation)
}


if __name__ == "__main__":
    """
    Example usage and testing.
    """
    print("Stage 5: Geometric Data Augmentation")
    print("=" * 60)
    
    if not MONAI_AVAILABLE:
        print("ERROR: MONAI not installed.")
        print("Install with: pip install monai")
        exit(1)
    
    # Example: Get train transforms
    train_transforms = get_train_transforms_3d()
    print(f"Train transforms: {len(train_transforms.transforms)} transforms")
    
    # Example: Get validation transforms
    val_transforms = get_val_transforms_3d()
    print(f"Val transforms: {len(val_transforms.transforms)} transforms (no augmentation)")
    
    # Example: Model-specific transforms
    mil_transforms = get_mil_transforms_3d("train")
    resnet_transforms = get_resnet3d_transforms_3d("train")
    swin_transforms = get_swin_unetr_transforms_3d("train")
    
    print("\nModel-specific transforms available:")
    print("  - MIL (slice-based)")
    print("  - ResNet50-3D")
    print("  - Swin UNETR")
    
    print("\nMedical augmentation parameters:")
    for key, value in MEDICAL_AUGMENTATION_PARAMS.items():
        print(f"  {key}: {value}")

