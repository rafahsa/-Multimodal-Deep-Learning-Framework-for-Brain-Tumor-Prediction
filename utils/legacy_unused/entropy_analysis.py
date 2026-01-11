"""
Entropy-based Slice Informativeness Analysis for MIL

This module provides entropy computation utilities for analyzing slice informativeness
in 3D MRI volumes. This is used exclusively with Multiple Instance Learning (MIL) models
to identify the most informative slices for training.

Key Principles:
- Metadata-only: Computes entropy scores, no image modification
- MIL-specific: Designed for slice-based MIL models
- GPU-compatible: Supports CUDA acceleration
- Deterministic: Reproducible results with fixed seeds

Requirements:
    torch >= 1.8.0
    numpy

Author: Medical Imaging Pipeline
"""

from typing import List, Optional, Union
import warnings

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. Install with: pip install torch")


def compute_slice_entropy(
    volume: Union[torch.Tensor, np.ndarray],
    axis: str = "axial",
    num_bins: int = 256,
    normalize: bool = True
) -> List[float]:
    """
    Compute Shannon entropy for each 2D slice along the given axis.
    
    Entropy is computed using histogram-based probability estimation:
    H(X) = -Σ p(x) * log2(p(x))
    
    Higher entropy indicates more informative slices (more uniform distribution).
    Lower entropy indicates less informative slices (more uniform or sparse).
    
    Args:
        volume: 3D volume tensor/array of shape (D, H, W) or (C, D, H, W)
        axis: Slice axis ("axial", "coronal", "sagittal")
              - "axial": slices along z-axis (depth)
              - "coronal": slices along y-axis (height)
              - "sagittal": slices along x-axis (width)
        num_bins: Number of bins for histogram computation
        normalize: Whether to normalize entropy to [0, 1] range
        
    Returns:
        List of entropy values, one per slice
        
    Example:
        >>> volume = torch.randn(128, 128, 128)  # 3D volume
        >>> entropies = compute_slice_entropy(volume, axis="axial")
        >>> # Returns list of 128 entropy values (one per slice)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    
    # Convert numpy to torch if needed
    if isinstance(volume, np.ndarray):
        volume = torch.from_numpy(volume).float()
    elif not isinstance(volume, torch.Tensor):
        volume = torch.tensor(volume, dtype=torch.float32)
    
    # Ensure float type
    volume = volume.float()
    
    # Handle channel dimension: (C, D, H, W) -> (D, H, W)
    if volume.dim() == 4:
        # Take first channel or average across channels
        volume = volume[0] if volume.shape[0] == 1 else volume.mean(dim=0)
    
    if volume.dim() != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
    
    # Determine slice axis
    axis_map = {
        "axial": 0,    # z-axis (depth)
        "coronal": 1,  # y-axis (height)
        "sagittal": 2  # x-axis (width)
    }
    
    if axis not in axis_map:
        raise ValueError(f"Unknown axis: {axis}. Must be one of: {list(axis_map.keys())}")
    
    slice_dim = axis_map[axis]
    num_slices = volume.shape[slice_dim]
    
    # Get min/max for histogram range
    v_min = volume.min().item()
    v_max = volume.max().item()
    
    # Avoid division by zero
    if v_max - v_min < 1e-10:
        # Constant volume, return zeros
        return [0.0] * num_slices
    
    # Compute entropy for each slice
    entropies = []
    
    for i in range(num_slices):
        # Extract slice
        if slice_dim == 0:
            slice_2d = volume[i, :, :]
        elif slice_dim == 1:
            slice_2d = volume[:, i, :]
        else:  # slice_dim == 2
            slice_2d = volume[:, :, i]
        
        # Flatten slice
        slice_flat = slice_2d.flatten()
        
        # Compute histogram
        # Move to CPU for histogram computation (more stable)
        slice_flat_cpu = slice_flat.cpu().numpy()
        hist, bin_edges = np.histogram(
            slice_flat_cpu,
            bins=num_bins,
            range=(v_min, v_max),
            density=False
        )
        
        # Normalize histogram to get probabilities
        hist_sum = hist.sum()
        if hist_sum == 0:
            entropy = 0.0
        else:
            probs = hist / hist_sum
            # Remove zeros to avoid log(0)
            probs = probs[probs > 0]
            # Compute Shannon entropy: H = -Σ p * log2(p)
            entropy = -np.sum(probs * np.log2(probs))
        
        entropies.append(float(entropy))
    
    # Normalize to [0, 1] if requested
    if normalize and len(entropies) > 0:
        max_entropy = max(entropies)
        if max_entropy > 0:
            entropies = [e / max_entropy for e in entropies]
    
    return entropies


def select_top_k_slices(
    entropy_scores: List[float],
    k: int,
    return_scores: bool = False
) -> Union[List[int], tuple]:
    """
    Select top-k most informative slices based on entropy scores.
    
    Args:
        entropy_scores: List of entropy values (one per slice)
        k: Number of top slices to select
        return_scores: If True, return (indices, scores) tuple
        
    Returns:
        List of slice indices (sorted by entropy, descending)
        If return_scores=True: (indices, scores) tuple
        
    Example:
        >>> entropies = [0.5, 0.8, 0.3, 0.9, 0.2]
        >>> top_indices = select_top_k_slices(entropies, k=2)
        >>> # Returns: [3, 1]  # Indices of slices with highest entropy
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    
    if len(entropy_scores) == 0:
        return [] if not return_scores else ([], [])
    
    k = min(k, len(entropy_scores))
    
    # Get indices sorted by entropy (descending)
    sorted_indices = sorted(
        range(len(entropy_scores)),
        key=lambda i: entropy_scores[i],
        reverse=True
    )
    
    top_indices = sorted_indices[:k]
    top_scores = [entropy_scores[i] for i in top_indices]
    
    # Sort indices in ascending order (slice order)
    top_indices_sorted = sorted(top_indices)
    
    if return_scores:
        return (top_indices_sorted, top_scores)
    else:
        return top_indices_sorted


def compute_volume_entropy_stats(
    entropy_scores: List[float]
) -> dict:
    """
    Compute statistics about entropy distribution.
    
    Args:
        entropy_scores: List of entropy values
        
    Returns:
        Dictionary with statistics (mean, std, min, max, median)
    """
    if len(entropy_scores) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0
        }
    
    scores_array = np.array(entropy_scores)
    
    return {
        'mean': float(np.mean(scores_array)),
        'std': float(np.std(scores_array)),
        'min': float(np.min(scores_array)),
        'max': float(np.max(scores_array)),
        'median': float(np.median(scores_array))
    }


# Example usage
if __name__ == "__main__":
    """
    Example usage and testing.
    """
    print("Entropy-based Slice Informativeness Analysis")
    print("=" * 60)
    
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not installed.")
        print("Install with: pip install torch")
        exit(1)
    
    # Example: Simulate a 3D volume
    volume = torch.randn(128, 128, 128)
    
    # Compute entropy for axial slices
    entropies = compute_slice_entropy(volume, axis="axial", normalize=False)
    print(f"\nComputed entropy for {len(entropies)} slices")
    print(f"Entropy range: [{min(entropies):.4f}, {max(entropies):.4f}]")
    
    # Select top-k slices
    top_indices = select_top_k_slices(entropies, k=16)
    print(f"\nTop 16 slice indices: {top_indices[:10]}... (showing first 10)")
    
    # Statistics
    stats = compute_volume_entropy_stats(entropies)
    print(f"\nEntropy statistics:")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std: {stats['std']:.4f}")
    print(f"  Min: {stats['min']:.4f}, Max: {stats['max']:.4f}")
    
    print("\n" + "=" * 60)
    print("This is a metadata-only analysis stage for MIL models.")

