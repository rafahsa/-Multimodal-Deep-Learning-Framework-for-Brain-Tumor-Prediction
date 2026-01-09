#!/usr/bin/env python3
"""
Dual-Stream MIL Training Script for BraTS 2018

This script trains a Dual-Stream Multiple Instance Learning (MIL) model for brain tumor
classification (HGG vs LGG) using 2D slice-based instances, CrossEntropyLoss, and
WeightedRandomSampler.

================================================================================
LOSS FUNCTION ANALYSIS AND DECISION
================================================================================

After comprehensive analysis of loss function options for MIL-based brain tumor
classification, the following decision has been made:

CHOSEN LOSS: CrossEntropyLoss + WeightedRandomSampler (data-level balancing)

RATIONALE:

1. MIL-Specific Considerations:
   - Bag-level supervision: Patient-level labels, instance-level (slice) features
   - Dual-stream aggregation already handles instance selection:
     * Stream 1: Critical instance identification (max-score selection)
     * Stream 2: Contextual attention aggregation (weighted combination)
   - Adding complex loss functions could destabilize the carefully designed
     dual-stream aggregation mechanism

2. Class Imbalance Analysis:
   - Moderate imbalance: HGG ≈ 210, LGG ≈ 75 (2.8:1 ratio)
   - WeightedRandomSampler handles this effectively at data level
   - No need for loss-level reweighting (which could conflict with aggregation)

3. Stability vs Sensitivity Trade-offs:
   - ResNet50-3D experience: LDAM + DRW caused severe instability
     * Loss values collapsing to near-zero or exploding
     * Validation AUC fluctuating 0.30-0.90 across epochs
     * Training curves showing erratic oscillations
   - CrossEntropyLoss + WeightedRandomSampler: Proven stable
     * Smooth convergence
     * Consistent validation metrics
     * Reproducible results

4. Risk of Over-Focusing on Critical Instance:
   - Dual-stream design mitigates this:
     * Stream 1 captures critical instance (strongest signal)
     * Stream 2 aggregates contextual support (prevents over-reliance)
     * Fusion combines both signals
   - CrossEntropyLoss at bag-level naturally balances both streams
   - Complex loss functions (Focal, LDAM) could amplify critical instance
     at the expense of contextual information

5. Architecture-Aware Decision:
   - MIL aggregation is already a form of attention/selection
   - Adding loss-level attention (Focal Loss) could create conflicting signals
   - Margin-based losses (LDAM) designed for instance-level, not bag-level
   - Standard CrossEntropyLoss is appropriate for bag-level classification

6. Generalization Across Folds:
   - Stable loss function → consistent training across folds
   - Complex losses → higher variance, harder to tune
   - CrossEntropyLoss enables fair comparison with ResNet50-3D and SwinUNETR-3D

REJECTED ALTERNATIVES:

❌ LDAM + DRW:
   - Proven unstable in ResNet50-3D training
   - Margin-based loss designed for instance-level, not bag-level
   - DRW reweighting conflicts with MIL aggregation
   - Risk of destabilizing dual-stream mechanism

❌ Focal Loss:
   - Hard-example mining could over-amplify noisy slices
   - Risk of over-focusing on critical instance (defeats purpose of Stream 2)
   - Additional hyperparameter (γ) increases complexity
   - Not specifically designed for MIL bag-level supervision

✅ CrossEntropyLoss + WeightedRandomSampler:
   - Proven stable in both ResNet50-3D and SwinUNETR-3D
   - Simple, well-understood, reproducible
   - Compatible with MIL bag-level supervision
   - Allows dual-stream aggregation to work naturally
   - Enables fair ensemble comparison

FINAL DECISION: CrossEntropyLoss + WeightedRandomSampler

This choice prioritizes:
- Training stability (critical for small medical datasets)
- Robust AUC (primary metric)
- Generalization across folds
- Ensemble compatibility (same loss strategy as other models)

================================================================================

IMPORTANT: Loss Tracking
- All loss values tracked are CrossEntropyLoss
- Class balancing handled by WeightedRandomSampler (data-level)
- Train and validation loss are logged at every epoch
- Full loss history is saved to metrics.json
- Dedicated loss_curve.png plot is generated

Author: Medical Imaging Pipeline
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
)


def convert_to_serializable(obj):
    """
    Recursively convert NumPy types to JSON-serializable Python types.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.dataset_mil import MILSliceDataset, get_all_labels
from utils.class_balancing import get_weighted_sampler
from utils.augmentations_2d import get_mil_slice_transforms, normalize_slice
from models.dual_stream_mil import create_dual_stream_mil


# Configure logging
def setup_logging(log_dir: Path, log_level: str = "INFO"):
    """Set up logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__), log_file


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience=10, min_delta=0.0, min_epochs=15, monitor_metric='auc', tie_breaker='f1'):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.monitor_metric = monitor_metric
        self.tie_breaker = tie_breaker
        self.best_score = -np.inf
        self.best_tie_score = -np.inf
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False
    
    def __call__(self, epoch, metrics: Dict):
        """Check if training should stop."""
        if epoch < self.min_epochs:
            return False
        
        primary_score = metrics.get(self.monitor_metric, -np.inf)
        tie_score = metrics.get(self.tie_breaker, -np.inf)
        
        # Check if improvement
        if primary_score > self.best_score + self.min_delta:
            self.best_score = primary_score
            self.best_tie_score = tie_score
            self.best_epoch = epoch
            self.counter = 0
            return False
        elif primary_score == self.best_score and tie_score > self.best_tie_score + self.min_delta:
            # Tie-breaker improvement
            self.best_tie_score = tie_score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict:
    """Compute classification metrics."""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    # AUC (if probabilities provided)
    if y_prob is not None:
        try:
            if y_prob.ndim > 1:
                y_prob_pos = y_prob[:, 1]  # Probability of positive class
            else:
                y_prob_pos = y_prob
            metrics['auc'] = float(roc_auc_score(y_true, y_prob_pos))
        except ValueError:
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0, target_names=['LGG', 'HGG'])
    metrics['classification_report'] = report
    
    return metrics


def save_plots(output_dir: Path, history: Dict, metrics: Dict, logger, epoch: Optional[int] = None):
    """Save training plots."""
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    suffix = f"_epoch_{epoch}" if epoch is not None else ""
    
    # Dedicated Loss Curve Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2, markersize=6)
    ax.plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2, markersize=6)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('CrossEntropy Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, len(history['train_loss'])])
    plt.tight_layout()
    plt.savefig(plots_dir / f'loss_curve{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved loss curve to {plots_dir / f'loss_curve{suffix}.png'}")
    
    # Learning curves (comprehensive)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', marker='o', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val', marker='s', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('CrossEntropy Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train', marker='o', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Val', marker='s', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC
    axes[1, 0].plot(history['val_auc'], label='Val AUC', marker='s', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC-ROC')
    axes[1, 0].set_title('Validation AUC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(history['lr'], label='LR', marker='o', linewidth=2, color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / f'learning_curves{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curve
    if 'val_probs' in metrics and metrics['val_probs'] is not None:
        y_true = np.array(metrics['val_labels'])
        y_prob = np.array(metrics['val_probs'])
        if y_prob.ndim > 1:
            y_prob_pos = y_prob[:, 1]
        else:
            y_prob_pos = y_prob
        
        fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
        auc = metrics.get('auc', 0.0)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, label=f'ROC (AUC = {auc:.4f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f'roc_curve{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Confusion Matrix
    if 'confusion_matrix' in metrics:
        cm = np.array(metrics['confusion_matrix'])
        fig, ax = plt.subplots(figsize=(8, 6))
        if HAS_SEABORN:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True)
        else:
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.figure.colorbar(im, ax=ax)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['LGG', 'HGG'])
        ax.set_yticklabels(['LGG', 'HGG'])
        plt.tight_layout()
        plt.savefig(plots_dir / f'confusion_matrix{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()


def update_ema_model(model, ema_model, decay):
    """Update exponential moving average model weights."""
    with torch.no_grad():
        # Handle DataParallel - get actual model
        actual_model = model.module if isinstance(model, DataParallel) else model
        actual_ema = ema_model.module if isinstance(ema_model, DataParallel) else ema_model
        
        # Update parameters directly
        for param, ema_param in zip(actual_model.parameters(), actual_ema.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
        
        # Update buffers (BatchNorm running stats, etc.)
        for buffer, ema_buffer in zip(actual_model.buffers(), actual_ema.buffers()):
            if buffer.dtype.is_floating_point:
                ema_buffer.data.mul_(decay).add_(buffer.data, alpha=1 - decay)
            else:
                ema_buffer.data.copy_(buffer.data)


def get_temperature(epoch: int, total_epochs: int, temp_start: float = 10.0, temp_end: float = 1.0, 
                    schedule: str = 'cosine') -> float:
    """
    Temperature annealing for curriculum learning in MIL.
    
    Starts with high temperature (soft selection, exploration) and anneals to
    low temperature (sharp selection, exploitation).
    
    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
        temp_start: Initial temperature (high = softer, more exploration)
        temp_end: Final temperature (low = sharper, more focused)
        schedule: 'linear', 'cosine', or 'exponential' (default: 'cosine' for faster annealing)
    
    Returns:
        Temperature for current epoch
    """
    if total_epochs <= 1:
        return temp_end
    
    # Convert to 0-indexed for calculations
    epoch_idx = epoch - 1
    progress = epoch_idx / (total_epochs - 1)
    
    if schedule == 'cosine':
        # Cosine annealing: faster decay early, slower later
        temperature = temp_end + (temp_start - temp_end) * 0.5 * (1 + np.cos(np.pi * progress))
    elif schedule == 'exponential':
        # Exponential decay: very fast early, very slow later
        decay_rate = np.log(temp_end / temp_start) / (total_epochs - 1)
        temperature = temp_start * np.exp(decay_rate * epoch_idx)
    else:  # linear
        # Linear decay: constant rate
        temperature = temp_start * (1 - progress) + temp_end * progress
    
    return max(temp_end, temperature)  # Never go below temp_end


def get_adaptive_label_smoothing(epoch: int, total_epochs: int, start: float = 0.2, end: float = 0.05) -> float:
    """
    Adaptive label smoothing: start high (prevents early overconfidence) and decay to lower value.
    
    High smoothing early prevents overconfidence, lower smoothing later allows sharper decisions.
    
    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
        start: Initial label smoothing (default: 0.2)
        end: Final label smoothing (default: 0.05)
    
    Returns:
        Label smoothing for current epoch
    """
    if total_epochs <= 1:
        return end
    
    epoch_idx = epoch - 1
    progress = epoch_idx / (total_epochs - 1)
    
    # Cosine decay: faster early, slower later
    smoothing = end + (start - end) * 0.5 * (1 + np.cos(np.pi * progress))
    return max(end, smoothing)


def get_adaptive_class_weight_scale(epoch: int, total_epochs: int, warmup_epochs: int = 10) -> float:
    """
    Adaptive class weight scaling: full weight early, decay after warmup.
    
    Class weights help early training but can cause instability later.
    Decay them after the model has learned basic patterns.
    
    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
        warmup_epochs: Epochs to use full class weights (default: 10)
    
    Returns:
        Scale factor for class weights (1.0 = full, 0.0 = disabled)
    """
    if epoch <= warmup_epochs:
        return 1.0
    
    # Linear decay from warmup_epochs to total_epochs
    decay_epochs = total_epochs - warmup_epochs
    if decay_epochs <= 0:
        return 1.0
    
    progress = (epoch - warmup_epochs) / decay_epochs
    # Decay to 0.3 (keep some class weighting but reduce it)
    return max(0.3, 1.0 - 0.7 * progress)


def get_adaptive_reg_weight(epoch: int, total_epochs: int, base_weight: float, 
                            decay_start: int = 15, min_weight: float = 0.005) -> float:
    """
    Adaptive regularization weight: full early, decay later.
    
    Regularization is important early but can be reduced as model stabilizes.
    
    Args:
        epoch: Current epoch (1-indexed)
        total_epochs: Total number of epochs
        base_weight: Base regularization weight
        decay_start: Epoch to start decaying (default: 15)
        min_weight: Minimum weight (default: 0.005)
    
    Returns:
        Regularization weight for current epoch
    """
    if epoch < decay_start:
        return base_weight
    
    # Linear decay from decay_start to total_epochs
    decay_epochs = total_epochs - decay_start
    if decay_epochs <= 0:
        return base_weight
    
    progress = (epoch - decay_start) / decay_epochs
    weight = base_weight * (1 - progress * (1 - min_weight / base_weight))
    return max(min_weight, weight)


def train_epoch(model, train_loader, loss_fn, optimizer, device, scaler, epoch, logger, 
                grad_clip=0.0, gradient_accumulation_steps=1, ema_model=None, ema_decay=0.0,
                temperature=1.0, reg_weight_entropy=0.0, reg_weight_confidence=0.0,
                class_weight_scale=1.0, label_smoothing=0.1):
    """
    Train for one epoch with temperature annealing and instance-level regularization.
    
    Anti-Overfitting Mechanisms:
    1. Temperature Annealing: Curriculum learning - start with exploration (high temp),
       gradually focus (low temp) to prevent early collapse to single slice.
    2. Attention Entropy Regularization: Encourages diverse attention across slices,
       preventing overfitting to a single "critical" slice.
    3. Selection Confidence Regularization: Encourages confident but not extreme selection,
       preventing collapse to hard selection.
    
    Args:
        temperature: Temperature for soft instance selection (for curriculum learning)
        reg_weight_entropy: Weight for attention entropy regularization (prevents attention collapse)
        reg_weight_confidence: Weight for selection confidence regularization (prevents extreme selection)
    
    Returns:
        epoch_loss: Average loss over training set
        epoch_acc: Training accuracy
        mil_stats: Dictionary with MIL diagnostics (attention entropy, selection stats)
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # MIL diagnostics tracking
    all_attention_entropies = []
    all_top1_attention_weights = []
    all_selection_entropies = []
    all_top1_selection_weights = []
    
    optimizer.zero_grad()  # Zero gradients at the start
    
    for batch_idx, (bags, labels, _) in enumerate(train_loader):
        # bags: (B, N, 4, H, W) where B=batch_size, N=bag_size (slices per patient)
        bags = bags.to(device)
        labels = labels.to(device)
        
        with autocast(enabled=scaler is not None):
            # Forward pass with temperature and interpretability info
            logits, interpretability = model(
                bags, 
                return_interpretability=True, 
                temperature=temperature
            )
            
            # Bag-level loss with adaptive label smoothing and class weights
            # Compute loss manually to support adaptive smoothing and class weight scaling
            log_probs = F.log_softmax(logits, dim=1)
            n_classes = logits.size(1)
            
            # Get class weights (if enabled) and apply scaling
            if hasattr(loss_fn, 'weight') and loss_fn.weight is not None:
                scaled_weights = loss_fn.weight * class_weight_scale
                # Index weights by labels
                weight_tensor = scaled_weights[labels]
            else:
                weight_tensor = None
            
            # Standard cross-entropy term
            ce_loss = F.nll_loss(log_probs, labels, weight=None, reduction='none')
            if weight_tensor is not None:
                ce_loss = ce_loss * weight_tensor
            
            # Label smoothing term: uniform distribution over all classes
            uniform_loss = -torch.sum(log_probs, dim=1) / n_classes
            if weight_tensor is not None:
                # For uniform term, use average weight
                avg_weight = scaled_weights.mean() if scaled_weights is not None else 1.0
                uniform_loss = uniform_loss * avg_weight
            
            # Combine with adaptive label smoothing
            bag_loss = torch.mean((1 - label_smoothing) * ce_loss + label_smoothing * uniform_loss)
            
            # Instance-level regularization (to prevent overfitting to specific slices)
            # Problem: Model can memorize patient-specific slice patterns
            # Solution: Regularize attention and selection to encourage diverse, generalizable learning
            reg_loss = 0.0
            
            if reg_weight_entropy > 0 or reg_weight_confidence > 0:
                selection_weights = interpretability['selection_weights']  # (B, N)
                attention_weights = interpretability['attention_weights']  # (B, N)
                instance_scores = interpretability['instance_scores']  # (B, N)
                
                # Compute entropies once and reuse for both logging and loss
                # Attention entropy: measure of diversity in attention distribution
                attention_entropy = -torch.sum(
                    attention_weights * torch.log(attention_weights + 1e-10), 
                    dim=1
                )  # (B,)
                
                # Selection entropy: measure of diversity in selection
                selection_entropy = -torch.sum(
                    selection_weights * torch.log(selection_weights + 1e-10),
                    dim=1
                )  # (B,)
                
                # Track MIL diagnostics for logging (detach to avoid gradient tracking)
                all_attention_entropies.extend(attention_entropy.detach().cpu().numpy())
                all_selection_entropies.extend(selection_entropy.detach().cpu().numpy())
                
                # Top-1 attention weight: indicator of attention collapse (1.0 = complete collapse)
                top1_attention = torch.max(attention_weights, dim=1)[0]  # (B,)
                all_top1_attention_weights.extend(top1_attention.detach().cpu().numpy())
                
                # Top-1 selection weight: indicator of selection collapse
                top1_selection = torch.max(selection_weights, dim=1)[0]  # (B,)
                all_top1_selection_weights.extend(top1_selection.detach().cpu().numpy())
                
                # Attention entropy loss: encourage diverse attention (prevent collapse to single slice)
                # Problem: Attention can collapse to one slice → overfitting to that slice
                # Solution: Maximize entropy (diversity) in attention distribution
                if reg_weight_entropy > 0:
                    # Use the already-computed entropy (no recomputation)
                    # Improved: Use squared penalty for stronger regularization when entropy is low
                    # This provides stronger gradients when attention is collapsing
                    target_entropy = np.log(float(attention_weights.shape[1]))  # Maximum entropy (uniform distribution)
                    entropy_deficit = target_entropy - attention_entropy  # How far from maximum
                    entropy_loss = torch.mean(entropy_deficit ** 2)  # Squared penalty (stronger when far from target)
                    reg_loss += reg_weight_entropy * entropy_loss
                
                # Selection confidence loss: encourage confident but not extreme selection
                # Problem: Selection can become too extreme (one slice gets all weight)
                # Solution: Encourage separation in scores (confidence) but not collapse
                # Improved: Add upper bound to prevent extreme selection
                if reg_weight_confidence > 0:
                    max_score = torch.max(instance_scores, dim=1)[0]  # (B,)
                    min_score = torch.min(instance_scores, dim=1)[0]  # (B,)
                    score_range = max_score - min_score
                    # Encourage separation but penalize extreme separation
                    # Target: moderate separation (not too small, not too large)
                    target_range = 2.0  # Reasonable target for logit separation
                    range_penalty = torch.mean((score_range - target_range) ** 2)
                    confidence_loss = range_penalty
                    reg_loss += reg_weight_confidence * confidence_loss
            
            # Total loss
            loss = bag_loss + reg_loss
            # Scale loss by gradient accumulation steps
            loss = loss / gradient_accumulation_steps
        
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation: only step every N batches
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            
            optimizer.zero_grad()
            
            # Update EMA model if enabled
            if ema_model is not None and ema_decay > 0:
                update_ema_model(model, ema_model, ema_decay)
        
        running_loss += loss.item() * gradient_accumulation_steps  # Scale back for logging
        
        # Predictions (argmax is invariant to softmax, so compute directly from logits)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        if batch_idx % 10 == 0:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item() * gradient_accumulation_steps:.6f}")
    
    # Handle remaining gradients if batch count is not divisible by accumulation steps
    if len(train_loader) % gradient_accumulation_steps != 0:
        if scaler is not None:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        optimizer.zero_grad()
    
    # Average loss over training set
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # Compute MIL diagnostics
    mil_stats = {}
    if all_attention_entropies:
        mil_stats['attention_entropy_mean'] = float(np.mean(all_attention_entropies))
        mil_stats['attention_entropy_std'] = float(np.std(all_attention_entropies))
        mil_stats['top1_attention_weight_mean'] = float(np.mean(all_top1_attention_weights))
        mil_stats['top1_attention_weight_std'] = float(np.std(all_top1_attention_weights))
        mil_stats['selection_entropy_mean'] = float(np.mean(all_selection_entropies))
        mil_stats['selection_entropy_std'] = float(np.std(all_selection_entropies))
        mil_stats['top1_selection_weight_mean'] = float(np.mean(all_top1_selection_weights))
        mil_stats['top1_selection_weight_std'] = float(np.std(all_top1_selection_weights))
        # Effective number of instances (diversity metric)
        # High entropy → more instances contribute → better generalization
        mil_stats['effective_attention_instances'] = float(np.mean([np.exp(e) for e in all_attention_entropies]))
        mil_stats['effective_selection_instances'] = float(np.mean([np.exp(e) for e in all_selection_entropies]))
    
    return epoch_loss, epoch_acc, mil_stats


def validate(model, val_loader, loss_fn, device, epoch, logger, temperature=1.0):
    """
    Validate model on validation set.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to run on
        epoch: Current epoch (for logging)
        logger: Logger instance
        temperature: Temperature for soft MIL selection (default: 1.0, use final temperature for evaluation)
    
    Returns:
        val_loss: Average loss over validation set
        metrics: Dictionary of validation metrics
        mil_stats: Dictionary with MIL diagnostics
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    # MIL diagnostics tracking
    all_attention_entropies = []
    all_top1_attention_weights = []
    all_selection_entropies = []
    all_top1_selection_weights = []
    
    with torch.no_grad():
        for bags, labels, _ in val_loader:
            bags = bags.to(device)
            labels = labels.to(device)
            
            with autocast():
                # Forward pass with temperature and interpretability (use final temperature for evaluation)
                logits, interpretability = model(
                    bags, 
                    return_interpretability=True,
                    temperature=temperature
                )
                loss = loss_fn(logits, labels)
                
                # Track MIL diagnostics
                attention_weights = interpretability['attention_weights']  # (B, N)
                selection_weights = interpretability['selection_weights']  # (B, N)
                
                # Attention entropy
                attention_entropy = -torch.sum(
                    attention_weights * torch.log(attention_weights + 1e-10),
                    dim=1
                )  # (B,)
                all_attention_entropies.extend(attention_entropy.detach().cpu().numpy())
                
                # Top-1 attention weight
                top1_attention = torch.max(attention_weights, dim=1)[0]  # (B,)
                all_top1_attention_weights.extend(top1_attention.detach().cpu().numpy())
                
                # Selection entropy
                selection_entropy = -torch.sum(
                    selection_weights * torch.log(selection_weights + 1e-10),
                    dim=1
                )  # (B,)
                all_selection_entropies.extend(selection_entropy.detach().cpu().numpy())
                
                # Top-1 selection weight
                top1_selection = torch.max(selection_weights, dim=1)[0]  # (B,)
                all_top1_selection_weights.extend(top1_selection.detach().cpu().numpy())
            
            running_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    val_loss = running_loss / len(val_loader)
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    
    # Store predictions for final evaluation
    metrics['val_probs'] = all_probs
    metrics['val_labels'] = all_labels
    metrics['val_preds'] = all_preds
    
    # Compute MIL diagnostics
    mil_stats = {}
    if all_attention_entropies:
        mil_stats['attention_entropy_mean'] = float(np.mean(all_attention_entropies))
        mil_stats['attention_entropy_std'] = float(np.std(all_attention_entropies))
        mil_stats['top1_attention_weight_mean'] = float(np.mean(all_top1_attention_weights))
        mil_stats['top1_attention_weight_std'] = float(np.std(all_top1_attention_weights))
        mil_stats['selection_entropy_mean'] = float(np.mean(all_selection_entropies))
        mil_stats['selection_entropy_std'] = float(np.std(all_selection_entropies))
        mil_stats['top1_selection_weight_mean'] = float(np.mean(all_top1_selection_weights))
        mil_stats['top1_selection_weight_std'] = float(np.std(all_top1_selection_weights))
        mil_stats['effective_attention_instances'] = float(np.mean([np.exp(e) for e in all_attention_entropies]))
        mil_stats['effective_selection_instances'] = float(np.mean([np.exp(e) for e in all_selection_entropies]))
    
    return val_loss, metrics, mil_stats


def main():
    parser = argparse.ArgumentParser(description='Train Dual-Stream MIL model for brain tumor classification')
    
    # Data args
    parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3, 4],
                       help='Fold number (0-4)')
    parser.add_argument('--data-root', type=str, default='data/processed/stage_4_resize/train',
                       help='Root directory for processed data')
    parser.add_argument('--splits-dir', type=str, default='splits',
                       help='Directory containing split CSV files')
    
    # Model args
    parser.add_argument('--bag-size', type=int, default=32,
                       help='Fixed number of slices per bag (default: 32, reduced from 64 to prevent memorization)')
    parser.add_argument('--sampling-strategy', type=str, default='random', choices=['random', 'sequential', 'entropy'],
                       help='Strategy for sampling slices (default: random, entropy pre-selection conflicts with learned selection)')
    parser.add_argument('--instance-encoder-backbone', type=str, default='resnet18', choices=['resnet18', 'efficientnet_b0'],
                       help='Backbone for instance encoder (default: resnet18)')
    parser.add_argument('--instance-encoder-input-size', type=int, default=224,
                       help='Input size for instance encoder (default: 224)')
    parser.add_argument('--attention-type', type=str, default='gated', choices=['gated', 'cosine'],
                       help='Attention type for contextual aggregation (default: gated)')
    parser.add_argument('--fusion-method', type=str, default='concat', choices=['concat', 'weighted', 'cross_attn'],
                       help='Fusion method for dual streams (default: concat)')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate in classification head (default: 0.5, increased from 0.4 for better regularization)')
    parser.add_argument('--use-hidden-layer', action='store_true', default=True,
                       help='Use hidden layer in classification head (default: True)')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=60,
                       help='Maximum number of epochs (default: 60)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (number of patients/bags, default: 4)')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate for instance encoder (default: 5e-5, reduced for better generalization)')
    parser.add_argument('--classifier-lr', type=float, default=1e-4,
                       help='Learning rate for classifier head (default: 1e-4, 2× encoder LR)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                       help='Weight decay (default: 5e-4, increased from 1e-4 for stronger L2 regularization)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=2,
                       help='Gradient accumulation steps (default: 2)')
    parser.add_argument('--ema-decay', type=float, default=0.995,
                       help='Exponential moving average decay (0.0-1.0, default: 0.995)')
    
    # Optimizer args
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam', 'sgd'],
                       help='Optimizer (default: adamw)')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'none'],
                       help='Learning rate scheduler (default: cosine)')
    parser.add_argument('--grad-clip', type=float, default=0.5,
                       help='Gradient clipping norm (0.0 = disabled, default: 0.5, more aggressive for stability)')
    
    # Early stopping args (tuned for overfitting prevention)
    parser.add_argument('--early-stopping', type=int, default=5,
                       help='Early stopping patience (default: 5, stop sooner to prevent overfitting)')
    parser.add_argument('--early-stopping-min-epochs', type=int, default=10,
                       help='Minimum epochs before early stopping can trigger (default: 10)')
    
    # Anti-overfitting regularization arguments (MIL-specific)
    # Note: --label-smoothing kept for backward compatibility but not used (replaced by adaptive schedules)
    parser.add_argument('--label-smoothing', type=float, default=None,
                       help='[DEPRECATED] Use --label-smoothing-start and --label-smoothing-end instead. Kept for backward compatibility.')
    parser.add_argument('--label-smoothing-start', type=float, default=0.2,
                       help='Initial label smoothing for adaptive schedule (default: 0.2, prevents early overconfidence)')
    parser.add_argument('--label-smoothing-end', type=float, default=0.05,
                       help='Final label smoothing for adaptive schedule (default: 0.05, allows sharper decisions later)')
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                       help='Use class weights in loss function (default: True, addresses class imbalance)')
    parser.add_argument('--class-weight-power', type=float, default=0.5,
                       help='Power for computing class weights: weight = (n_total / n_class)^power (default: 0.5, balanced)')
    parser.add_argument('--class-weight-warmup-epochs', type=int, default=10,
                       help='Epochs to use full class weights before decay (default: 10, reduces late-epoch instability)')
    parser.add_argument('--temperature-start', type=float, default=10.0,
                       help='Initial temperature for soft MIL selection (default: 10.0, high = exploration)')
    parser.add_argument('--temperature-end', type=float, default=1.0,
                       help='Final temperature for soft MIL selection (default: 1.0, low = exploitation)')
    parser.add_argument('--temperature-schedule', type=str, default='cosine', choices=['linear', 'cosine', 'exponential'],
                       help='Temperature annealing schedule (default: cosine, faster decay early to prevent late-epoch instability)')
    parser.add_argument('--reg-weight-entropy', type=float, default=0.01,
                       help='Base weight for attention entropy regularization (default: 0.01, decays adaptively)')
    parser.add_argument('--reg-weight-confidence', type=float, default=0.01,
                       help='Base weight for selection confidence regularization (default: 0.01, decays adaptively)')
    parser.add_argument('--reg-weight-decay-start', type=int, default=15,
                       help='Epoch to start decaying regularization weights (default: 15, allows fine-tuning later)')
    
    # Plotting and diagnostics
    parser.add_argument('--plot-every', type=int, default=1,
                       help='Save plots every N epochs (default: 1, set to 0 to disable per-epoch plots)')
    
    # Data loading args
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of data loader workers (default: auto)')
    
    # GPU args
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU ID to use (default: all available)')
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Use multiple GPUs with DataParallel')
    
    # Training features
    parser.add_argument('--amp', action='store_true', default=True,
                       help='Use mixed precision training (default: True)')
    parser.add_argument('--no-amp', dest='amp', action='store_false',
                       help='Disable mixed precision training')
    parser.add_argument('--tf32', action='store_true', default=True,
                       help='Enable TF32 (default: True)')
    
    # Misc args
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='results/DualStreamMIL-3D',
                       help='Output directory (default: results/DualStreamMIL-3D)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable TF32 if requested
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Setup device
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directories
    output_dir = Path(args.output_dir)
    run_dir = output_dir / 'runs' / f'fold_{args.fold}' / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints_dir = run_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_dir = run_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_dir = run_dir / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger, log_file = setup_logging(run_dir)
    logger.info("="*80)
    logger.info("Dual-Stream MIL Training Script")
    logger.info("="*80)
    logger.info(f"Fold: {args.fold}")
    logger.info(f"Output directory: {run_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info("="*80)
    
    # Save configuration
    config = vars(args)
    config['device'] = str(device)
    config['output_dir'] = str(run_dir)
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load datasets
    data_root = Path(args.data_root)
    splits_dir = Path(args.splits_dir)
    
    train_split_file = splits_dir / f'fold_{args.fold}_train.csv'
    val_split_file = splits_dir / f'fold_{args.fold}_val.csv'
    
    if not train_split_file.exists():
        raise FileNotFoundError(f"Training split file not found: {train_split_file}")
    if not val_split_file.exists():
        raise FileNotFoundError(f"Validation split file not found: {val_split_file}")
    
    # Get augmentation transforms
    train_aug = get_mil_slice_transforms(mode='train')
    val_aug = get_mil_slice_transforms(mode='val')
    
    # Create datasets
    train_dataset = MILSliceDataset(
        data_root=data_root,
        split_file=train_split_file,
        bag_size=args.bag_size,
        sampling_strategy=args.sampling_strategy,
        transform=train_aug,
        seed=args.seed
    )
    
    val_dataset = MILSliceDataset(
        data_root=data_root,
        split_file=val_split_file,
        bag_size=args.bag_size,
        sampling_strategy='sequential',  # Deterministic for validation
        transform=val_aug,
        seed=args.seed
    )
    
    logger.info(f"Training set: {len(train_dataset)} patients")
    logger.info(f"Validation set: {len(val_dataset)} patients")
    
    # Create weighted sampler for training
    train_labels = get_all_labels(train_dataset)
    train_sampler = get_weighted_sampler(train_labels, strategy='inverse_freq')
    
    # Create data loaders
    num_workers = args.num_workers if args.num_workers is not None else min(4, os.cpu_count() or 1)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create model
    logger.info("\n" + "="*60)
    logger.info("Creating Dual-Stream MIL Model")
    logger.info("="*60)
    model = create_dual_stream_mil(
        num_classes=2,
        instance_encoder_backbone=args.instance_encoder_backbone,
        instance_encoder_input_size=args.instance_encoder_input_size,
        attention_type=args.attention_type,
        fusion_method=args.fusion_method,
        dropout=args.dropout,
        use_hidden_layer=args.use_hidden_layer,
        logger=logger
    )
    model = model.to(device)
    
    # Multi-GPU support
    if args.multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = DataParallel(model)
    
    # Compute class weights for loss function (addresses class imbalance)
    class_weights = None
    if args.use_class_weights:
        # Get labels from training dataset
        all_train_labels = get_all_labels(train_dataset)
        unique, counts = np.unique(all_train_labels, return_counts=True)
        n_total = len(all_train_labels)
        n_classes = len(unique)
        
        # Compute weights: weight = (n_total / n_class)^power
        # power=0.5 gives balanced weights, power=1.0 gives inverse frequency
        weights = {}
        for cls, count in zip(unique, counts):
            weights[int(cls)] = (n_total / (n_classes * count)) ** args.class_weight_power
        
        # Convert to tensor, ordered by class index (0, 1)
        class_weights = torch.tensor([weights.get(0, 1.0), weights.get(1, 1.0)], dtype=torch.float32).to(device)
        logger.info(f"Class weights computed: LGG={class_weights[0]:.4f}, HGG={class_weights[1]:.4f}")
        logger.info(f"  Class distribution: LGG={counts[unique==0][0] if 0 in unique else 0}, HGG={counts[unique==1][0] if 1 in unique else 0}")
    else:
        logger.info("Class weights: Disabled (using uniform weights)")
    
    # Handle backward compatibility: if old --label-smoothing is used, set start/end to same value
    if args.label_smoothing is not None:
        logger.warning(f"--label-smoothing is deprecated. Using value {args.label_smoothing} for both start and end.")
        args.label_smoothing_start = args.label_smoothing
        args.label_smoothing_end = args.label_smoothing
    
    # Loss function: CrossEntropyLoss with adaptive label smoothing and class weights (anti-overfitting)
    # Problem: Model becomes overconfident (logits → ±∞) → poor generalization + class imbalance
    # Solution: Adaptive label smoothing (high early, low late) prevents early overconfidence and late instability
    # Note: We create a base loss function, but actual smoothing/weights are applied adaptively in train_epoch
    # This is because CrossEntropyLoss doesn't support dynamic label_smoothing
    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=0.0,  # Will be applied manually with adaptive value
        weight=class_weights  # Base weights, will be scaled adaptively
    )
    logger.info(f"Loss function: CrossEntropyLoss (adaptive label_smoothing: {args.label_smoothing_start} → {args.label_smoothing_end}, class_weights={'enabled' if class_weights is not None else 'disabled'})")
    logger.info(f"Class balancing: WeightedRandomSampler (data-level) + Adaptive class weights (loss-level, decay after {args.class_weight_warmup_epochs} epochs)")
    logger.info(f"Instance-level regularization: base entropy_weight={args.reg_weight_entropy}, base confidence_weight={args.reg_weight_confidence} (adaptive decay from epoch {args.reg_weight_decay_start})")
    logger.info(f"Temperature annealing: {args.temperature_start} → {args.temperature_end} (schedule: {args.temperature_schedule}, faster decay early)")
    logger.info("")
    logger.info("="*60)
    logger.info("ANTI-OVERFITTING MECHANISMS ENABLED:")
    logger.info("="*60)
    logger.info(f"  1. Reduced Bag Size ({args.bag_size}): Less memorization capacity (50% reduction from 64)")
    logger.info(f"  2. Adaptive Label Smoothing ({args.label_smoothing_start} → {args.label_smoothing_end}): Prevents early overconfidence, allows sharper decisions later")
    logger.info(f"  3. Adaptive Temperature Annealing ({args.temperature_start} → {args.temperature_end}, {args.temperature_schedule}): Faster decay early to prevent late-epoch instability")
    logger.info(f"  4. Adaptive Class Weights: Full weight for {args.class_weight_warmup_epochs} epochs, then decay to reduce late-epoch instability")
    logger.info(f"  5. Adaptive Attention Entropy Reg. (base={args.reg_weight_entropy}, decay from epoch {args.reg_weight_decay_start}): Encourages diverse attention early, reduces later")
    logger.info(f"  6. Adaptive Selection Confidence Reg. (base={args.reg_weight_confidence}, decay from epoch {args.reg_weight_decay_start}): Prevents extreme selection early, reduces later")
    logger.info(f"  7. Increased Dropout ({args.dropout}): Stronger feature regularization")
    logger.info(f"  8. Increased Weight Decay ({args.weight_decay}): Stronger L2 regularization")
    logger.info(f"  9. Reduced Learning Rates (encoder={args.lr}, classifier={args.classifier_lr}): Slower memorization")
    logger.info(f" 10. Aggressive Gradient Clipping ({args.grad_clip}): Prevents extreme updates")
    logger.info(f" 11. Early Stopping (patience={args.early_stopping}, min_epochs={args.early_stopping_min_epochs}): Stops before overfitting")
    logger.info("")
    logger.info("SAMPLING STRATEGY: Random (entropy pre-selection conflicts with learned selection)")
    logger.info("="*60)
    logger.info("")
    
    # Optimizer with differential learning rates
    encoder_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        else:
            encoder_params.append(param)
    
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            [
                {'params': encoder_params, 'lr': args.lr},
                {'params': classifier_params, 'lr': args.classifier_lr}
            ],
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            [
                {'params': encoder_params, 'lr': args.lr},
                {'params': classifier_params, 'lr': args.classifier_lr}
            ],
            weight_decay=args.weight_decay
        )
    else:  # sgd
        optimizer = torch.optim.SGD(
            [
                {'params': encoder_params, 'lr': args.lr},
                {'params': classifier_params, 'lr': args.classifier_lr}
            ],
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True
        )
    
    # Learning rate scheduler with warmup
    if args.scheduler == 'cosine':
        warmup_epochs = max(5, args.epochs // 10)  # 10% of epochs for warmup (min 5)
        
        def lr_lambda_encoder(epoch):
            """LR schedule for encoder"""
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                if args.epochs <= warmup_epochs:
                    return 1.0
                progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        def lr_lambda_classifier(epoch):
            """LR schedule for classifier"""
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                if args.epochs <= warmup_epochs:
                    return 1.0
                progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[lr_lambda_encoder, lr_lambda_classifier]
        )
        logger.info(f"Using cosine annealing with {warmup_epochs} epoch warmup (differential LRs)")
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None
    
    # Mixed precision
    scaler = GradScaler() if args.amp else None
    if args.amp:
        logger.info("Mixed precision training enabled")
    
    # Exponential Moving Average (EMA)
    ema_model = None
    if args.ema_decay > 0:
        logger.info(f"Exponential Moving Average enabled with decay={args.ema_decay}")
        ema_model = create_dual_stream_mil(
            num_classes=2,
            instance_encoder_backbone=args.instance_encoder_backbone,
            instance_encoder_input_size=args.instance_encoder_input_size,
            attention_type=args.attention_type,
            fusion_method=args.fusion_method,
            dropout=args.dropout,
            use_hidden_layer=args.use_hidden_layer,
            logger=None
        )
        if isinstance(model, DataParallel):
            main_model_state = model.module.state_dict()
        else:
            main_model_state = model.state_dict()
        ema_model.load_state_dict(main_model_state)
        ema_model = ema_model.to(device)
        ema_model.eval()
        if args.multi_gpu and torch.cuda.device_count() > 1:
            ema_model = DataParallel(ema_model)
    else:
        logger.info("Exponential Moving Average disabled")
    
    # Gradient accumulation
    if args.gradient_accumulation_steps > 1:
        logger.info(f"Gradient accumulation enabled: {args.gradient_accumulation_steps} steps")
        logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping,
        min_epochs=args.early_stopping_min_epochs,
        monitor_metric='auc',
        tie_breaker='f1'
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc': [],
        'lr': [],
        'train_mil_stats': [],  # MIL diagnostics per epoch
        'val_mil_stats': []    # MIL diagnostics per epoch
    }
    
    best_val_auc = -np.inf
    best_val_f1 = -np.inf
    best_epoch = 0
    
    # DRY-RUN VERIFICATION
    logger.info("\n" + "="*60)
    logger.info("DRY-RUN VERIFICATION: Checking data pipeline")
    logger.info("="*60)
    try:
        sample_batch = next(iter(train_loader))
        bags, labels, patient_ids = sample_batch
        
        batch_size = bags.shape[0]
        bag_size = bags.shape[1]
        expected_shape = (batch_size, bag_size, 4, 128, 128)
        
        logger.info(f"Sample batch - Bag shape: {bags.shape}")
        logger.info(f"Sample batch - Labels shape: {labels.shape}")
        logger.info(f"Sample batch - Expected bag shape: {expected_shape}")
        logger.info(f"Sample batch - Number of patient IDs: {len(patient_ids)}")
        
        assert bags.ndim == 5, f"Bag must be 5D (B, N, C, H, W), got {bags.ndim}D"
        assert bags.shape[0] == batch_size, f"Batch size mismatch"
        assert bags.shape[1] == bag_size, f"Bag size mismatch"
        assert bags.shape[2] == 4, f"Channel mismatch: expected 4 channels, got {bags.shape[2]}"
        assert labels.ndim == 1, f"Labels must be 1D (batch_size,)"
        
        logger.info("✓ Shape verification PASSED")
        logger.info("="*60 + "\n")
    except Exception as e:
        logger.error(f"DRY-RUN VERIFICATION FAILED: {e}")
        raise
    
    logger.info("Starting training...")
    logger.info(f"Total epochs: {args.epochs}")
    logger.info(f"Early stopping patience: {args.early_stopping}")
    logger.info(f"Batch size: {args.batch_size} patients (bags)")
    logger.info(f"Bag size: {args.bag_size} slices per patient")
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        # Compute adaptive schedules for current epoch
        # Temperature: faster annealing to prevent late-epoch instability
        current_temperature = get_temperature(
            epoch, args.epochs, args.temperature_start, args.temperature_end,
            schedule=args.temperature_schedule
        )
        
        # Label smoothing: decay from high (early) to low (late) for sharper decisions
        current_label_smoothing = get_adaptive_label_smoothing(
            epoch, args.epochs, args.label_smoothing_start, args.label_smoothing_end
        )
        
        # Class weights: full early, decay after warmup to reduce late-epoch instability
        class_weight_scale = get_adaptive_class_weight_scale(
            epoch, args.epochs, args.class_weight_warmup_epochs
        )
        
        # Regularization weights: full early, decay later as model stabilizes
        current_reg_entropy = get_adaptive_reg_weight(
            epoch, args.epochs, args.reg_weight_entropy, args.reg_weight_decay_start
        )
        current_reg_confidence = get_adaptive_reg_weight(
            epoch, args.epochs, args.reg_weight_confidence, args.reg_weight_decay_start
        )
        
        # Train for one epoch with adaptive schedules
        train_loss, train_acc, train_mil_stats = train_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler, epoch, logger,
            args.grad_clip, args.gradient_accumulation_steps, ema_model, args.ema_decay,
            temperature=current_temperature,
            reg_weight_entropy=current_reg_entropy,
            reg_weight_confidence=current_reg_confidence,
            class_weight_scale=class_weight_scale,
            label_smoothing=current_label_smoothing
        )
        
        # Validate with final temperature (use temperature_end for evaluation)
        val_loss, val_metrics, val_mil_stats = validate(
            model, val_loader, loss_fn, device, epoch, logger, 
            temperature=args.temperature_end
        )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = args.lr
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        history['lr'].append(current_lr)
        history['train_mil_stats'].append(train_mil_stats)
        history['val_mil_stats'].append(val_mil_stats)
        
        # Log metrics
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"EPOCH {epoch} METRICS:")
        logger.info("=" * 80)
        logger.info(f"  TEMPERATURE:      {current_temperature:.2f} (annealing: {args.temperature_start:.1f} → {args.temperature_end:.1f}, schedule: {args.temperature_schedule})")
        logger.info(f"  LABEL SMOOTHING:  {current_label_smoothing:.4f} (adaptive: {args.label_smoothing_start:.2f} → {args.label_smoothing_end:.2f})")
        logger.info(f"  CLASS WEIGHT SCALE: {class_weight_scale:.3f} (warmup: {args.class_weight_warmup_epochs} epochs)")
        logger.info(f"  REG WEIGHTS:       entropy={current_reg_entropy:.4f}, confidence={current_reg_confidence:.4f}")
        logger.info(f"  TRAIN:")
        logger.info(f"    Loss:            {train_loss:.6f}")
        logger.info(f"    Accuracy:        {train_acc:.4f}")
        logger.info(f"  VALIDATION:")
        logger.info(f"    Loss:            {val_loss:.6f}")
        logger.info(f"    Accuracy:        {val_metrics['accuracy']:.4f}")
        logger.info(f"    Precision:       {val_metrics['precision']:.4f}")
        logger.info(f"    Recall:          {val_metrics['recall']:.4f}")
        logger.info(f"    F1-Score:        {val_metrics['f1']:.4f}")
        logger.info(f"    AUC-ROC:         {val_metrics['auc']:.4f}")
        logger.info(f"  LEARNING RATE:    {current_lr:.6e}")
        
        # Log MIL diagnostics
        if train_mil_stats:
            logger.info(f"  MIL DIAGNOSTICS (Train):")
            logger.info(f"    Attention Entropy:     {train_mil_stats.get('attention_entropy_mean', 0):.4f} ± {train_mil_stats.get('attention_entropy_std', 0):.4f}")
            logger.info(f"    Top-1 Attention Weight: {train_mil_stats.get('top1_attention_weight_mean', 0):.4f} ± {train_mil_stats.get('top1_attention_weight_std', 0):.4f}")
            logger.info(f"    Effective Instances:   {train_mil_stats.get('effective_attention_instances', 0):.2f}")
        if val_mil_stats:
            logger.info(f"  MIL DIAGNOSTICS (Val):")
            logger.info(f"    Attention Entropy:     {val_mil_stats.get('attention_entropy_mean', 0):.4f} ± {val_mil_stats.get('attention_entropy_std', 0):.4f}")
            logger.info(f"    Top-1 Attention Weight: {val_mil_stats.get('top1_attention_weight_mean', 0):.4f} ± {val_mil_stats.get('top1_attention_weight_std', 0):.4f}")
            logger.info(f"    Effective Instances:   {val_mil_stats.get('effective_attention_instances', 0):.2f}")
            # Warning if attention collapse detected
            if val_mil_stats.get('top1_attention_weight_mean', 0) > 0.8:
                logger.warning(f"    ⚠️  ATTENTION COLLAPSE DETECTED: Top-1 weight > 0.8 (diversity: {val_mil_stats.get('effective_attention_instances', 0):.2f} instances)")
        
        logger.info("=" * 80)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'val_metrics': val_metrics,
            'history': history,
            'config': config,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'loss_type': 'CrossEntropyLoss'
        }
        
        torch.save(checkpoint, checkpoints_dir / 'last.pt')
        
        # Save best checkpoint (by val AUC)
        if val_metrics['auc'] > best_val_auc or (val_metrics['auc'] == best_val_auc and val_metrics['f1'] > best_val_f1):
            best_val_auc = val_metrics['auc']
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch
            
            checkpoint['best_val_auc'] = best_val_auc
            checkpoint['best_val_f1'] = best_val_f1
            
            best_checkpoint_path = checkpoints_dir / 'best.pt'
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(f"✓ Saved BEST regular checkpoint: {best_checkpoint_path}")
            logger.info(f"  Epoch: {epoch}, Val AUC: {best_val_auc:.4f}, Val F1: {best_val_f1:.4f}")
            
            # Save EMA checkpoint if enabled
            if ema_model is not None and args.ema_decay > 0:
                ema_checkpoint = checkpoint.copy()
                ema_checkpoint['model_state_dict'] = ema_model.module.state_dict() if isinstance(ema_model, DataParallel) else ema_model.state_dict()
                ema_checkpoint['is_ema'] = True
                best_ema_checkpoint_path = checkpoints_dir / 'best_ema.pt'
                torch.save(ema_checkpoint, best_ema_checkpoint_path)
                logger.info(f"✓ Saved BEST EMA checkpoint: {best_ema_checkpoint_path}")
                logger.info(f"  Epoch: {epoch}, Val AUC: {best_val_auc:.4f}, Val F1: {best_val_f1:.4f}")
        
        # Save plots based on plot_every setting
        if args.plot_every > 0 and (epoch % args.plot_every == 0 or epoch == best_epoch):
            try:
                save_plots(run_dir, history, val_metrics, logger, epoch=epoch)
            except Exception as e:
                logger.error(f"Failed to save plots at epoch {epoch}: {e}", exc_info=True)
        
        # Early stopping
        if early_stopping(epoch, val_metrics):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            logger.info(f"Best epoch: {early_stopping.best_epoch}, Best AUC: {early_stopping.best_score:.4f}")
            break
    
    # Final evaluation
    logger.info("\n" + "="*80)
    logger.info("Training completed!")
    logger.info("="*80)
    logger.info(f"Best epoch during training: {best_epoch}")
    logger.info(f"Best validation AUC during training: {best_val_auc:.4f}")
    logger.info("="*80)
    
    # Load best checkpoint for final evaluation
    best_checkpoint_path = checkpoints_dir / 'best.pt'
    best_ema_checkpoint_path = checkpoints_dir / 'best_ema.pt'
    
    use_ema = False
    if args.ema_decay > 0 and best_ema_checkpoint_path.exists():
        checkpoint_path = best_ema_checkpoint_path
        use_ema = True
        logger.info(f"Loading BEST EMA checkpoint: {checkpoint_path}")
    elif best_checkpoint_path.exists():
        checkpoint_path = best_checkpoint_path
        use_ema = False
        logger.info(f"Loading BEST regular checkpoint: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Best checkpoint not found")
    
    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    checkpoint_epoch = best_checkpoint.get('epoch')
    checkpoint_val_auc = best_checkpoint.get('val_metrics', {}).get('auc')
    checkpoint_is_ema = best_checkpoint.get('is_ema', False)
    
    logger.info(f"Checkpoint info:")
    logger.info(f"  Epoch: {checkpoint_epoch}")
    logger.info(f"  Val AUC: {checkpoint_val_auc}")
    logger.info(f"  Is EMA: {checkpoint_is_ema}")
    
    # Load weights
    if use_ema and ema_model is not None:
        eval_model = ema_model
        if isinstance(eval_model, DataParallel):
            eval_model.module.load_state_dict(best_checkpoint['model_state_dict'])
        else:
            eval_model.load_state_dict(best_checkpoint['model_state_dict'])
    else:
        eval_model = model
        if isinstance(eval_model, DataParallel):
            eval_model.module.load_state_dict(best_checkpoint['model_state_dict'])
        else:
            eval_model.load_state_dict(best_checkpoint['model_state_dict'])
    
    eval_model.eval()
    
    # Final evaluation
    logger.info("\n" + "="*80)
    logger.info("FINAL EVALUATION: Evaluating best checkpoint on validation set")
    logger.info(f"Checkpoint: {checkpoint_path.name}")
    logger.info(f"Model type: {'EMA' if use_ema else 'Regular'}")
    logger.info("="*80)
    _, final_metrics, final_mil_stats = validate(
        eval_model, val_loader, loss_fn, device, args.epochs, logger,
        temperature=args.temperature_end
    )
    
    # Log final MIL diagnostics
    if final_mil_stats:
        logger.info(f"\nFINAL MIL DIAGNOSTICS:")
        logger.info(f"  Attention Entropy:     {final_mil_stats.get('attention_entropy_mean', 0):.4f} ± {final_mil_stats.get('attention_entropy_std', 0):.4f}")
        logger.info(f"  Top-1 Attention Weight: {final_mil_stats.get('top1_attention_weight_mean', 0):.4f} ± {final_mil_stats.get('top1_attention_weight_std', 0):.4f}")
        logger.info(f"  Effective Instances:   {final_mil_stats.get('effective_attention_instances', 0):.2f}")
        if final_mil_stats.get('top1_attention_weight_mean', 0) > 0.8:
            logger.warning(f"  ⚠️  ATTENTION COLLAPSE: Top-1 weight > 0.8")
    
    # Compute best_epoch and best_val_auc from training_history
    if len(history['val_auc']) > 0:
        val_auc_array = np.array(history['val_auc'])
        best_epoch_idx = int(np.argmax(val_auc_array))
        computed_best_epoch = best_epoch_idx + 1
        computed_best_val_auc = float(val_auc_array[best_epoch_idx])
        
        if computed_best_val_auc != best_val_auc:
            logger.warning(f"Computed best_val_auc ({computed_best_val_auc:.6f}) differs from tracked ({best_val_auc:.6f})")
            best_epoch = computed_best_epoch
            best_val_auc = computed_best_val_auc
    
    # Add training history
    final_metrics['training_history'] = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc'],
        'val_precision': history['val_precision'],
        'val_recall': history['val_recall'],
        'val_f1': history['val_f1'],
        'val_auc': history['val_auc'],
        'lr': history['lr'],
        'train_mil_stats': history['train_mil_stats'],
        'val_mil_stats': history['val_mil_stats']
    }
    
    # Add final MIL diagnostics
    if final_mil_stats:
        final_metrics['final_mil_diagnostics'] = final_mil_stats
    
    # Add loss summary
    final_metrics['loss_summary'] = {
        'train_loss': {
            'final': float(history['train_loss'][-1]),
            'min': float(min(history['train_loss'])),
            'max': float(max(history['train_loss'])),
            'mean': float(np.mean(history['train_loss'])),
            'std': float(np.std(history['train_loss']))
        },
        'val_loss': {
            'final': float(history['val_loss'][-1]),
            'min': float(min(history['val_loss'])),
            'max': float(max(history['val_loss'])),
            'mean': float(np.mean(history['val_loss'])),
            'std': float(np.std(history['val_loss'])),
            'best_epoch': int(best_epoch),
            'best_value': float(history['val_loss'][best_epoch - 1]) if best_epoch > 0 and len(history['val_loss']) >= best_epoch else None
        }
    }
    
    # Add loss info
    final_metrics['loss_info'] = {
        'loss_type': 'CrossEntropyLoss',
        'class_balancing': 'WeightedRandomSampler (data-level)',
        'note': 'CrossEntropyLoss chosen for MIL stability and compatibility with dual-stream aggregation'
    }
    
    # Add checkpoint info
    final_metrics['checkpoint_info'] = {
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_name': checkpoint_path.name,
        'checkpoint_epoch': int(checkpoint_epoch) if isinstance(checkpoint_epoch, (int, float)) else None,
        'checkpoint_val_auc': float(checkpoint_val_auc) if isinstance(checkpoint_val_auc, (int, float)) else None,
        'is_ema': bool(use_ema),
        'best_epoch': int(best_epoch),
        'best_val_auc': float(best_val_auc)
    }
    
    # Update top-level metrics from best epoch
    if best_epoch > 0 and len(history['val_acc']) >= best_epoch:
        best_epoch_idx = best_epoch - 1
        final_metrics['accuracy'] = float(history['val_acc'][best_epoch_idx])
        if len(history['val_precision']) > best_epoch_idx:
            final_metrics['precision'] = float(history['val_precision'][best_epoch_idx])
        if len(history['val_recall']) > best_epoch_idx:
            final_metrics['recall'] = float(history['val_recall'][best_epoch_idx])
        if len(history['val_f1']) > best_epoch_idx:
            final_metrics['f1'] = float(history['val_f1'][best_epoch_idx])
        final_metrics['auc'] = float(best_val_auc)
        logger.info(f"Updated top-level metrics from best epoch {best_epoch} (val_auc: {best_val_auc:.6f})")
    
    # Save metrics
    final_metrics_serializable = convert_to_serializable(final_metrics)
    with open(metrics_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics_serializable, f, indent=2)
    
    logger.info(f"Saved metrics with full training history to {metrics_dir / 'metrics.json'}")
    
    # Save predictions
    np.save(predictions_dir / 'val_probs.npy', final_metrics['val_probs'])
    np.save(predictions_dir / 'val_labels.npy', final_metrics['val_labels'])
    np.save(predictions_dir / 'val_preds.npy', np.argmax(final_metrics['val_probs'], axis=1))
    
    # Save final plots
    save_plots(run_dir, history, final_metrics, logger, epoch=None)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL EVALUATION METRICS (Best Checkpoint):")
    logger.info("=" * 80)
    logger.info(f"  Checkpoint used:   {checkpoint_path.name}")
    logger.info(f"  Model type:        {'EMA' if use_ema else 'Regular'}")
    logger.info(f"  Checkpoint epoch:  {checkpoint_epoch}")
    logger.info(f"  Checkpoint AUC:    {checkpoint_val_auc}")
    logger.info("")
    logger.info("  Evaluation results:")
    logger.info(f"    Accuracy:        {final_metrics['accuracy']:.4f}")
    logger.info(f"    Precision:       {final_metrics['precision']:.4f}")
    logger.info(f"    Recall:          {final_metrics['recall']:.4f}")
    logger.info(f"    F1-Score:        {final_metrics['f1']:.4f}")
    logger.info(f"    AUC-ROC:         {final_metrics['auc']:.4f}")
    logger.info("")
    logger.info(f"Results saved to: {run_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

