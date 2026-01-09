"""
2D Slice Augmentation for MIL (Multiple Instance Learning)

This module provides augmentation transforms for 2D slices used in MIL models.
Augmentations are applied per-slice independently, maintaining slice-level consistency.

Key Principles:
- Medical-safe: Preserves anatomical plausibility at slice level
- Per-slice: Applied to individual slices (not full volumes)
- Train-only: No augmentation during validation/testing
- Multi-modal: Applied consistently across all 4 modalities (T1, T1ce, T2, FLAIR)

Author: Medical Imaging Pipeline
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class SliceAugmentation:
    """
    Augmentation transforms for 2D slices.
    
    Applies mild geometric and intensity augmentations that preserve
    anatomical plausibility at the slice level.
    """
    
    def __init__(
        self,
        flip_prob: float = 0.5,
        intensity_scale_range: Tuple[float, float] = (0.95, 1.05),
        apply_prob: float = 0.8
    ):
        self.flip_prob = flip_prob
        self.intensity_scale_range = intensity_scale_range
        self.apply_prob = apply_prob
    
    def __call__(self, slice_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to a single slice.
        
        Args:
            slice_tensor: (4, H, W) multi-modal slice [T1, T1ce, T2, FLAIR]
        
        Returns:
            augmented_slice: (4, H, W) augmented slice
        """
        if torch.rand(1).item() > self.apply_prob:
            return slice_tensor
        
        # Random horizontal flip
        if torch.rand(1).item() < self.flip_prob:
            slice_tensor = torch.flip(slice_tensor, dims=[2])  # Flip along width
        
        # Random vertical flip
        if torch.rand(1).item() < self.flip_prob:
            slice_tensor = torch.flip(slice_tensor, dims=[1])  # Flip along height
        
        # Random intensity scaling (per-channel, maintains relative intensities)
        if self.intensity_scale_range[0] < 1.0 or self.intensity_scale_range[1] > 1.0:
            scale = torch.empty(4).uniform_(self.intensity_scale_range[0], self.intensity_scale_range[1])
            slice_tensor = slice_tensor * scale.view(4, 1, 1)
        
        return slice_tensor


def get_mil_slice_transforms(mode: str = "train") -> Optional[SliceAugmentation]:
    """
    Get augmentation transforms for MIL slices.
    
    Args:
        mode: "train" or "val"
    
    Returns:
        SliceAugmentation instance for training, None for validation
    """
    if mode == "train":
        return SliceAugmentation(
            flip_prob=0.5,
            intensity_scale_range=(0.95, 1.05),  # ±5% intensity variation
            apply_prob=0.8
        )
    else:
        return None  # No augmentation for validation


def normalize_slice(slice_tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize slice to [0, 1] range per channel.
    
    Args:
        slice_tensor: (4, H, W) multi-modal slice
    
    Returns:
        normalized_slice: (4, H, W) normalized slice
    """
    normalized = torch.zeros_like(slice_tensor)
    for c in range(slice_tensor.shape[0]):
        channel = slice_tensor[c]
        min_val = channel.min()
        max_val = channel.max()
        if max_val > min_val:
            normalized[c] = (channel - min_val) / (max_val - min_val)
    return normalized

