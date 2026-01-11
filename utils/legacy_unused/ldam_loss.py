"""
LDAM (Label-Distribution-Aware Margin) Loss with DRW (Deferred Re-Weighting)

This module implements LDAM loss for handling class imbalance in medical imaging,
combined with DRW (Deferred Re-Weighting) strategy that applies class weights
only after a certain number of epochs.

LDAM loss formula:
    L = -log(exp(s * (cos(θ_y) - m_y)) / (exp(s * (cos(θ_y) - m_y)) + Σ exp(s * cos(θ_j))))

Where:
    - θ_y is the angle between features and class y weight vector
    - m_y is the class-dependent margin (larger for minority classes)
    - s is a scaling factor

Reference:
    - Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss" (NeurIPS 2019)
    - DRW strategy defers re-weighting until model has learned basic representations

Author: Medical Imaging Pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss.
    
    For imbalanced 3D medical datasets, reduced margins and scaling improve stability:
    - max_m=0.3 (reduced from 0.5): Prevents excessive margin penalties that cause instability
    - s=20 (reduced from 30): Lower temperature scaling reduces gradient variance
    
    Args:
        num_classes: Number of classes
        max_m: Maximum margin value (default: 0.3 for stability in 3D medical imaging)
        s: Scaling factor (default: 20 for stability in 3D medical imaging)
        reduction: Loss reduction ('mean' or 'none')
    """
    
    def __init__(self, num_classes=2, max_m=0.3, s=20, reduction='mean'):
        super().__init__()
        self.num_classes = num_classes
        self.max_m = max_m
        self.s = s
        self.reduction = reduction
        
        # Margin values (will be set by compute_margins)
        self.register_buffer('margins', torch.zeros(num_classes))
    
    def compute_margins(self, class_counts: List[int]):
        """
        Compute class-dependent margins based on class frequencies.
        
        Formula: m_y = C / n_y^(1/4)
        Where C is chosen so that max(m_y) = max_m
        
        Args:
            class_counts: List of counts for each class [n_0, n_1, ...]
        """
        class_counts = torch.tensor(class_counts, dtype=torch.float32)
        
        # Compute margins: m_y proportional to 1 / n_y^(1/4)
        margins_raw = 1.0 / (class_counts ** 0.25)
        
        # Normalize so max margin = max_m
        if margins_raw.max() > 0:
            margins_normalized = margins_raw / margins_raw.max() * self.max_m
        else:
            margins_normalized = torch.zeros_like(margins_raw)
        
        self.margins = margins_normalized
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, class_weights: Optional[torch.Tensor] = None):
        """
        Compute LDAM loss.
        
        LDAM loss applies class-dependent margins to logits before computing cross-entropy.
        Formula: L = -log(exp(s * (logit_y - m_y)) / Σ exp(s * logit_j))
        
        Args:
            logits: Classification logits (batch_size, num_classes)
            targets: Class labels (batch_size,)
            class_weights: Optional per-sample weights for DRW (batch_size,)
        
        Returns:
            loss: LDAM loss value
        """
        batch_size = logits.size(0)
        
        # Get margins for target classes
        target_margins = self.margins[targets]  # (batch_size,)
        
        # Apply margin: subtract margin from target class logit
        # This makes the target class logit smaller (harder to predict correctly)
        logits_margin = logits.clone()
        for i in range(batch_size):
            logits_margin[i, targets[i]] -= target_margins[i]
        
        # Scale logits
        logits_margin = logits_margin * self.s
        
        # Compute cross-entropy loss with margin-adjusted logits
        loss = F.cross_entropy(logits_margin, targets, reduction='none', weight=None)
        
        # Apply class weights if provided (DRW)
        if class_weights is not None:
            loss = loss * class_weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


def compute_class_weights_drw(class_counts: List[int], drw_epoch: int, current_epoch: int, device: str = 'cuda'):
    """
    Compute class weights for DRW (Deferred Re-Weighting).
    
    Strategy:
    - Before drw_epoch: return None (no re-weighting)
    - After drw_epoch: return weights proportional to sqrt(n_max / n_j)
    
    Args:
        class_counts: List of counts for each class [n_0, n_1, ...]
        drw_epoch: Epoch at which to start applying weights
        current_epoch: Current training epoch
        device: Device for tensor ('cuda' or 'cpu')
    
    Returns:
        class_weights: Tensor of shape (num_classes,) on specified device, or None
    """
    if current_epoch < drw_epoch:
        return None
    
    class_counts = np.array(class_counts, dtype=np.float32)
    n_max = class_counts.max()
    
    # Compute weights: sqrt(n_max / n_j)
    weights = np.sqrt(n_max / (class_counts + 1e-8))  # Add epsilon to avoid division by zero
    
    # Create tensor directly on target device (GPU-first)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_loss_fn(
    num_classes: int,
    class_counts: List[int],
    max_m: float = 0.3,  # Reduced from 0.5 for stability in 3D medical imaging
    s: float = 20,  # Reduced from 30 for stability in 3D medical imaging
    drw_start_epoch: int = 15,
    device: str = 'cuda'
):
    """
    Build LDAM loss function with DRW support.
    
    Args:
        num_classes: Number of classes
        class_counts: List of counts for each class [n_0, n_1, ...]
        max_m: Maximum margin value for LDAM
        s: Scaling factor for LDAM
        drw_start_epoch: Epoch at which to start DRW
        device: Device for tensors
    
    Returns:
        loss_fn: Function that takes (logits, targets, epoch) and returns loss
    """
    # Create LDAM loss
    ldam_loss = LDAMLoss(num_classes=num_classes, max_m=max_m, s=s)
    ldam_loss.compute_margins(class_counts)
    ldam_loss = ldam_loss.to(device)
    
    def loss_fn(logits: torch.Tensor, targets: torch.Tensor, epoch: int):
        """
        Compute LDAM loss with DRW class weights (fully GPU-resident).
        
        All tensors remain on GPU throughout computation:
        - logits: (batch_size, num_classes) on GPU
        - targets: (batch_size,) on GPU
        - class_weights_drw: (num_classes,) on GPU (if DRW active)
        - sample_weights: (batch_size,) on GPU (if DRW active)
        
        Args:
            logits: Classification logits (batch_size, num_classes) on GPU
            targets: Class labels (batch_size,) on GPU
            epoch: Current training epoch
        
        Returns:
            loss: LDAM loss value (scalar tensor on GPU)
        """
        # Get device from logits (ensures we use the same device)
        device_actual = logits.device
        
        # Compute DRW class weights on the same device as logits/targets
        class_weights_drw = compute_class_weights_drw(
            class_counts, drw_start_epoch, epoch, device=device_actual
        )
        
        if class_weights_drw is not None:
            # Map class weights to per-sample weights (all on GPU)
            # class_weights_drw is already on device_actual (GPU)
            # targets is on device_actual (GPU)
            # Indexing GPU tensor with GPU indices is safe
            sample_weights = class_weights_drw[targets]
        else:
            sample_weights = None
        
        # Compute LDAM loss (all tensors on GPU)
        loss = ldam_loss(logits, targets, class_weights=sample_weights)
        
        return loss
    
    return loss_fn


# Example usage
if __name__ == "__main__":
    """
    Test LDAM loss with DRW.
    """
    print("Testing LDAM Loss with DRW")
    print("=" * 60)
    
    num_classes = 2
    class_counts = [75, 210]  # LGG=75, HGG=210 (imbalanced)
    
    print(f"Class counts: {class_counts}")
    print(f"Class distribution: LGG={class_counts[0]/(sum(class_counts))*100:.1f}%, HGG={class_counts[1]/(sum(class_counts))*100:.1f}%")
    
    # Build loss function
    loss_fn = build_loss_fn(
        num_classes=num_classes,
        class_counts=class_counts,
        max_m=0.5,
        s=30,
        drw_start_epoch=15,
        device='cpu'
    )
    
    # Test forward pass
    batch_size = 4
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    
    print(f"\nTesting epoch 10 (before DRW):")
    loss_before = loss_fn(logits, targets, epoch=10)
    print(f"Loss: {loss_before.item():.4f}")
    
    print(f"\nTesting epoch 20 (after DRW):")
    loss_after = loss_fn(logits, targets, epoch=20)
    print(f"Loss: {loss_after.item():.4f}")
    
    print("\n" + "=" * 60)
    print("LDAM Loss test passed!")

