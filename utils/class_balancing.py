"""
Stage 6: Class Balancing (Oversampling) for 3D Medical Imaging

This module provides runtime class balancing utilities for handling class imbalance
(HGG vs LGG) in the BraTS2018 dataset. Class balancing is applied ONLY during training
via weighted sampling in the DataLoader.

Key Principles:
- Runtime-only: No files created on disk, applied in DataLoader
- Training-only: Never applied to validation or test sets
- Sampling-based: Uses WeightedRandomSampler (no image synthesis)
- Deterministic: Supports reproducible sampling with seeds
- Patient-safe: Maintains patient-level atomicity

Requirements:
    torch >= 1.8.0
    numpy

Author: Medical Imaging Pipeline
"""

from typing import List, Optional, Tuple, Union
import warnings

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. Install with: pip install torch")


def compute_class_weights(
    labels: Union[List[int], np.ndarray, torch.Tensor],
    strategy: str = "inverse_freq",
    smooth: float = 1e-6
) -> np.ndarray:
    """
    Compute class weights from label distribution.
    
    Args:
        labels: Array-like of class labels (0 for LGG, 1 for HGG typically)
        strategy: Weight computation strategy:
            - "inverse_freq": Weight = total_samples / (num_classes * class_freq)
            - "balanced": Weight = total_samples / (num_classes * class_count)
            - "uniform": All classes have weight 1.0
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Array of class weights (length = num_classes)
        
    Example:
        >>> labels = [0, 0, 0, 1, 1, 1, 1, 1]  # 3 LGG, 5 HGG
        >>> weights = compute_class_weights(labels, strategy="inverse_freq")
        >>> # weights[0] > weights[1] (LGG is minority class)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    
    # Convert to numpy array
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    labels = np.asarray(labels)
    
    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    num_classes = len(unique_classes)
    total_samples = len(labels)
    
    # Initialize weights array (indexed by class value)
    max_class = int(unique_classes.max())
    weights = np.ones(max_class + 1, dtype=np.float32)
    
    if strategy == "uniform":
        # All classes have equal weight
        return weights
    
    elif strategy == "balanced" or strategy == "inverse_freq":
        # Compute weights: total_samples / (num_classes * class_count)
        for class_idx, count in zip(unique_classes, class_counts):
            weight = total_samples / (num_classes * (count + smooth))
            weights[int(class_idx)] = weight
        
        return weights
    
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Must be one of: 'inverse_freq', 'balanced', 'uniform'"
        )


def get_weighted_sampler(
    labels: Union[List[int], np.ndarray, torch.Tensor],
    strategy: str = "inverse_freq",
    num_samples: Optional[int] = None,
    replacement: bool = True,
    generator: Optional[torch.Generator] = None,
    seed: Optional[int] = None
) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler for balanced class sampling.
    
    This sampler ensures that during training, minority classes are sampled
    more frequently, effectively balancing the class distribution.
    
    Args:
        labels: Array-like of class labels (per sample in dataset)
        strategy: Weight computation strategy (see compute_class_weights)
        num_samples: Number of samples to draw per epoch.
                    If None, uses len(labels)
        replacement: Whether to sample with replacement (required for balancing)
        generator: Optional PyTorch generator for reproducibility
        seed: Optional random seed for reproducibility (creates generator if provided)
              If both generator and seed are provided, generator takes precedence
        
    Returns:
        WeightedRandomSampler instance
        
    Example:
        >>> dataset = BraTSDataset(...)
        >>> labels = [dataset[i][1] for i in range(len(dataset))]  # Get all labels
        >>> sampler = get_weighted_sampler(labels, strategy="inverse_freq", seed=42)
        >>> train_loader = DataLoader(dataset, batch_size=8, sampler=sampler)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    
    # Convert to numpy array
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    labels = np.asarray(labels)
    
    # Compute class weights
    class_weights = compute_class_weights(labels, strategy=strategy)
    
    # Map each sample to its class weight
    sample_weights = class_weights[labels.astype(int)]
    
    # Convert to torch tensor
    sample_weights_tensor = torch.from_numpy(sample_weights).float()
    
    # Determine number of samples
    if num_samples is None:
        num_samples = len(labels)
    
    # Handle seed: create generator from seed if provided and generator is None
    if generator is None and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=num_samples,
        replacement=replacement,
        generator=generator
    )
    
    return sampler


def get_balanced_dataloader(
    dataset: Dataset,
    labels: Optional[Union[List[int], np.ndarray, torch.Tensor]] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    strategy: str = "inverse_freq",
    shuffle: Optional[bool] = None,
    pin_memory: bool = True,
    drop_last: bool = False,
    seed: Optional[int] = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader with class-balanced sampling for training.
    
    This function automatically creates a WeightedRandomSampler if labels are provided,
    or uses standard shuffling if labels are None (useful for validation/test).
    
    Args:
        dataset: PyTorch Dataset instance
        labels: Optional array of class labels. If provided, uses WeightedRandomSampler.
                If None, uses standard DataLoader behavior (for validation/test)
        batch_size: Batch size
        num_workers: Number of worker processes
        strategy: Weight computation strategy (only used if labels provided)
        shuffle: Whether to shuffle. If None, auto-determined:
                - True if labels provided (uses sampler)
                - False if labels None (validation/test)
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        seed: Optional random seed for reproducibility
        **kwargs: Additional arguments passed to DataLoader
        
    Returns:
        DataLoader instance
        
    Example:
        >>> # Training: with class balancing
        >>> train_dataset = BraTSDataset(data_path, mode="train")
        >>> train_labels = train_dataset.get_all_labels()  # Get labels
        >>> train_loader = get_balanced_dataloader(
        ...     train_dataset, 
        ...     labels=train_labels,
        ...     batch_size=8,
        ...     seed=42
        ... )
        >>> 
        >>> # Validation: no balancing
        >>> val_dataset = BraTSDataset(data_path, mode="val")
        >>> val_loader = get_balanced_dataloader(
        ...     val_dataset,
        ...     labels=None,  # No balancing
        ...     batch_size=8,
        ...     shuffle=False
        ... )
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install torch")
    
    # Determine shuffle behavior
    if shuffle is None:
        shuffle = (labels is not None)  # Shuffle if using balanced sampling
    
    # Create sampler if labels provided
    sampler = None
    if labels is not None:
        # Create generator for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        
        sampler = get_weighted_sampler(
            labels=labels,
            strategy=strategy,
            num_samples=len(dataset),
            replacement=True,
            generator=generator
        )
        
        # Cannot use shuffle with sampler
        if shuffle:
            warnings.warn(
                "shuffle=True ignored when using WeightedRandomSampler. "
                "Sampler handles randomization.",
                UserWarning
            )
        shuffle = False
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator if (seed is not None and labels is None) else None,
        **kwargs
    )
    
    return dataloader


def get_class_distribution(
    labels: Union[List[int], np.ndarray, torch.Tensor],
    class_names: Optional[List[str]] = None
) -> dict:
    """
    Get class distribution statistics.
    
    Args:
        labels: Array-like of class labels
        class_names: Optional list of class names (e.g., ["LGG", "HGG"])
        
    Returns:
        Dictionary with class distribution statistics
        
    Example:
        >>> labels = [0, 0, 0, 1, 1, 1, 1, 1]
        >>> stats = get_class_distribution(labels, class_names=["LGG", "HGG"])
        >>> # Returns: {'LGG': 3, 'HGG': 5, 'total': 8, 'ratio': 0.375}
    """
    # Convert to numpy array
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    labels = np.asarray(labels)
    
    # Get unique classes and counts
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    # Build distribution dict
    distribution = {}
    if class_names is None:
        class_names = [f"Class_{i}" for i in unique_classes]
    
    for class_idx, count, name in zip(unique_classes, class_counts, class_names):
        distribution[name] = {
            'count': int(count),
            'percentage': float(count / total * 100),
            'weight': float(compute_class_weights(labels)[int(class_idx)])
        }
    
    distribution['total'] = total
    distribution['ratio'] = float(class_counts.min() / class_counts.max()) if len(class_counts) > 1 else 1.0
    
    return distribution


# Example usage patterns
if __name__ == "__main__":
    """
    Example usage and testing.
    """
    print("Stage 6: Class Balancing (Runtime-Only)")
    print("=" * 60)
    
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not installed.")
        print("Install with: pip install torch")
        exit(1)
    
    # Example: Simulate class imbalance (3 LGG, 5 HGG)
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 1])  # 3 LGG (0), 5 HGG (1)
    
    print(f"\nClass distribution:")
    print(f"  LGG (0): {(labels == 0).sum()} samples")
    print(f"  HGG (1): {(labels == 1).sum()} samples")
    
    # Compute class weights
    weights = compute_class_weights(labels, strategy="inverse_freq")
    print(f"\nComputed class weights (inverse_freq):")
    print(f"  LGG (0): {weights[0]:.4f}")
    print(f"  HGG (1): {weights[1]:.4f}")
    print(f"  (LGG has higher weight as it's the minority class)")
    
    # Create weighted sampler
    sampler = get_weighted_sampler(labels, strategy="inverse_freq", num_samples=100, seed=42)
    print(f"\nCreated WeightedRandomSampler:")
    print(f"  num_samples: {sampler.num_samples}")
    print(f"  replacement: {sampler.replacement}")
    
    # Get distribution stats
    stats = get_class_distribution(labels, class_names=["LGG", "HGG"])
    print(f"\nClass distribution statistics:")
    for class_name in ["LGG", "HGG"]:
        if class_name in stats:
            print(f"  {class_name}: {stats[class_name]['count']} samples "
                  f"({stats[class_name]['percentage']:.1f}%), "
                  f"weight={stats[class_name]['weight']:.4f}")
    print(f"  Imbalance ratio: {stats['ratio']:.3f}")
    
    print("\n" + "=" * 60)
    print("Stage 6 is runtime-only: No files are created on disk.")
    print("Balancing is applied in DataLoader during training only.")

