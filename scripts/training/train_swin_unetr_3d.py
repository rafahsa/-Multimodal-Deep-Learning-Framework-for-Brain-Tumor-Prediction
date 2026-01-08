#!/usr/bin/env python3
"""
Swin UNETR-3D Training Script for BraTS 2018

This script trains a Swin UNETR encoder-based model for brain tumor classification (HGG vs LGG)
using CrossEntropyLoss, WeightedRandomSampler, and multi-GPU support.

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
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
)


def convert_to_serializable(obj):
    """
    Recursively convert NumPy types to JSON-serializable Python types.
    
    Handles: ndarray, numpy scalars (int, float, bool), dict, list, tuple
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

from utils.dataset_3d import Volume3DDataset
from utils.dataset_3d_multi_modal import MultiModalVolume3DDataset
# Removed LDAM loss import - using CrossEntropyLoss instead
# from utils.ldam_loss import build_loss_fn
from utils.class_balancing import get_weighted_sampler
from utils.augmentations_3d import get_resnet3d_transforms_3d
from models.swin_unetr_encoder import SwinUNETREncoderClassifier


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
    
    def __init__(self, patience=7, min_delta=0.0, min_epochs=10, monitor_metric='auc', tie_breaker='f1'):
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
    
    # Loss (also in learning curves for completeness)
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('CrossEntropy Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Val AUC
    axes[0, 1].plot(history['val_auc'], label='Val AUC', marker='o', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC-ROC')
    axes[0, 1].set_title('Validation AUC-ROC')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Val F1
    axes[1, 0].plot(history['val_f1'], label='Val F1', marker='o', color='orange', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Accuracy
    axes[1, 1].plot(history['train_acc'], label='Train Acc', marker='o', alpha=0.7, linewidth=2)
    axes[1, 1].plot(history['val_acc'], label='Val Acc', marker='s', alpha=0.7, linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Training and Validation Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(plots_dir / f'learning_curves{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(8, 6))
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['LGG', 'HGG'], yticklabels=['LGG', 'HGG'])
    else:
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['LGG', 'HGG'])
        ax.set_yticklabels(['LGG', 'HGG'])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontweight='bold')
        plt.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(plots_dir / f'confusion_matrix{suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC curve
    if 'val_probs' in metrics:
        y_true = metrics['val_labels']
        y_probs = metrics['val_probs'][:, 1] if metrics['val_probs'].ndim > 1 else metrics['val_probs']
        
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc = metrics['auc']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f'roc_curve{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, label='PR Curve', linewidth=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / f'pr_curve{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    if epoch is None:
        logger.info(f"Saved plots to {plots_dir}")


def train_epoch(model, train_loader, loss_fn, optimizer, device, scaler, epoch, logger, 
                grad_clip=0.0, gradient_accumulation_steps=1, ema_model=None, ema_decay=0.0):
    """
    Train for one epoch.
    
    Returns:
        epoch_loss: Average CrossEntropyLoss over training set
        epoch_acc: Training accuracy
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()  # Zero gradients at the start
    
    for batch_idx, (volumes, labels, _) in enumerate(train_loader):
        volumes = volumes.to(device)
        labels = labels.to(device)
        
        with autocast(enabled=scaler is not None):
            logits = model(volumes)
            # Compute CrossEntropyLoss
            loss = loss_fn(logits, labels)
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
        
        # Predictions
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
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
    
    return epoch_loss, epoch_acc


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
        # Skip integer buffers (like num_batches_tracked) - copy them directly
        for buffer, ema_buffer in zip(actual_model.buffers(), actual_ema.buffers()):
            if buffer.dtype.is_floating_point:
                # Apply EMA to floating point buffers (running_mean, running_var, etc.)
                ema_buffer.data.mul_(decay).add_(buffer.data, alpha=1 - decay)
            else:
                # Copy integer buffers directly (num_batches_tracked, etc.)
                ema_buffer.data.copy_(buffer.data)


def validate(model, val_loader, loss_fn, device, epoch, logger):
    """
    Validate model.
    
    Returns:
        epoch_loss: Average CrossEntropyLoss over validation set
        metrics: Dictionary containing classification metrics
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for volumes, labels, _ in val_loader:
            volumes = volumes.to(device)
            labels = labels.to(device)
            
            logits = model(volumes)
            # Compute CrossEntropyLoss
            loss = loss_fn(logits, labels)
            
            running_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Average loss over validation set
    epoch_loss = running_loss / len(val_loader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    metrics['val_probs'] = np.array(all_probs)
    metrics['val_labels'] = np.array(all_labels)
    metrics['val_loss'] = float(epoch_loss)  # Store loss
    
    return epoch_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train Swin UNETR-3D model")
    
    # Required args
    parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3, 4],
                       help='Fold number (0-4)')
    
    # Data args
    parser.add_argument('--modality', type=str, default=None, choices=['t1', 't1ce', 't2', 'flair', None],
                       help='Single modality to use (default: None = use all 4 modalities with early fusion)')
    parser.add_argument('--multi-modal', action='store_true', default=True,
                       help='Use multi-modality input (T1, T1ce, T2, FLAIR) with early fusion (default: True)')
    parser.add_argument('--data-root', type=str, default='data/processed/stage_4_resize/train',
                       help='Root directory for processed data')
    parser.add_argument('--splits-dir', type=str, default='splits',
                       help='Directory containing split CSV files')
    
    # Model args
    parser.add_argument('--dropout', type=float, default=0.4,
                       help='Dropout rate in classification head (default: 0.4)')
    parser.add_argument('--feature-size', type=int, default=48,
                       help='Base feature size for Swin UNETR (default: 48 for memory efficiency)')
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 2, 2],
                       help='Number of layers in each stage (default: [2, 2, 2, 2])')
    parser.add_argument('--num-heads', type=int, nargs='+', default=[3, 6, 12, 24],
                       help='Number of attention heads in each stage (default: [3, 6, 12, 24])')
    parser.add_argument('--use-checkpoint', action='store_true',
                       help='Enable gradient checkpointing for memory efficiency')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=60,
                       help='Maximum number of epochs (default: 60 for better convergence)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (default: 4, increased for stability)')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate for encoder (default: 5e-5)')
    parser.add_argument('--classifier-lr', type=float, default=1e-4,
                       help='Learning rate for classifier head (default: 1e-4)')
    parser.add_argument('--freeze-backbone-epochs', type=int, default=0,
                       help='Number of epochs to freeze backbone (0 = no freezing, default: 0)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=2,
                       help='Gradient accumulation steps for effective larger batch size (default: 2, effective batch=12)')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                       help='Label smoothing factor (0.0-0.1, default: 0.0)')
    parser.add_argument('--ema-decay', type=float, default=0.995,
                       help='Exponential moving average decay (0.0-1.0, default: 0.995 for better training dynamics)')
    
    # LDAM + DRW args (DEPRECATED - kept for backward compatibility but not used)
    # Loss function simplified to CrossEntropyLoss for stability
    parser.add_argument('--drw-start-epoch', type=int, default=25,
                       help='[DEPRECATED] Not used - LDAM+DRW removed for stability')
    parser.add_argument('--max-m', type=float, default=0.2,
                       help='[DEPRECATED] Not used - LDAM+DRW removed for stability')
    parser.add_argument('--s', type=float, default=15,
                       help='[DEPRECATED] Not used - LDAM+DRW removed for stability')
    
    # Optimizer/Scheduler args
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam', 'sgd'],
                       help='Optimizer (default: adamw)')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'none'],
                       help='Learning rate scheduler (default: cosine)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping (0.0 = disabled, default: 1.0 for stability)')
    
    # Early stopping args
    parser.add_argument('--early-stopping', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--early-stopping-min-epochs', type=int, default=15,
                       help='Minimum epochs before early stopping (default: 15)')
    
    # Data loading args
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of data loader workers (default: auto)')
    
    # GPU args
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU ID to use (default: all available)')
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Use multiple GPUs with DataParallel')
    
    # Performance args
    parser.add_argument('--amp', action='store_true', default=True,
                       help='Use mixed precision training (default: True)')
    parser.add_argument('--no-amp', dest='amp', action='store_false',
                       help='Disable mixed precision training')
    parser.add_argument('--tf32', action='store_true', default=True,
                       help='Enable TF32 (default: True)')
    
    # General args
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='results/SwinUNETR-3D',
                       help='Output directory (default: results/SwinUNETR-3D)')
    
    args = parser.parse_args()
    
    # Set device
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set performance optimizations
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(args.output_dir) / 'runs' / f'fold_{args.fold}' / f'run_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints_dir = run_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_dir = run_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    config_dir = run_dir / 'config'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_dir = run_dir / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger, log_file = setup_logging(run_dir / 'logs')
    logger.info(f"Starting Swin UNETR-3D training for fold {args.fold}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")
    
    # Save config
    config = vars(args).copy()
    config['device'] = str(device)
    config['torch_version'] = torch.__version__
    config['cuda_available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        config['cuda_version'] = torch.version.cuda
        config['gpu_name'] = torch.cuda.get_device_name(0)
        config['num_gpus'] = torch.cuda.device_count()
    config['class_mapping'] = {'LGG': 0, 'HGG': 1}
    
    # Convert config to JSON-serializable format
    config_serializable = convert_to_serializable(config)
    with open(config_dir / 'config.json', 'w') as f:
        json.dump(config_serializable, f, indent=2)
    logger.info(f"Saved config to {config_dir / 'config.json'}")
    
    # Load datasets
    splits_dir = project_root / args.splits_dir
    data_root = project_root / args.data_root
    
    train_csv = splits_dir / f'fold_{args.fold}_train.csv'
    val_csv = splits_dir / f'fold_{args.fold}_val.csv'
    
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Split files not found: {train_csv}, {val_csv}")
    
    logger.info(f"Loading datasets from {data_root}")
    logger.info(f"Train CSV: {train_csv}")
    logger.info(f"Val CSV: {val_csv}")
    
    # Get transforms
    # Determine if using multi-modality or single modality
    use_multimodal = args.multi_modal or args.modality is None
    num_channels = 4 if use_multimodal else 1
    
    train_transforms = get_resnet3d_transforms_3d(mode='train', num_channels=num_channels)
    val_transforms = get_resnet3d_transforms_3d(mode='val', num_channels=num_channels)
    
    if use_multimodal:
        logger.info("Using multi-modality input (T1, T1ce, T2, FLAIR) with early fusion")
        logger.info("Input shape: (4, D, H, W) - 4 channels stacked")
        
        # Create multi-modal datasets
        train_dataset = MultiModalVolume3DDataset(
            data_root=str(data_root),
            split_file=str(train_csv),
            modalities=['t1', 't1ce', 't2', 'flair'],
            transform=train_transforms
        )
        
        val_dataset = MultiModalVolume3DDataset(
            data_root=str(data_root),
            split_file=str(val_csv),
            modalities=['t1', 't1ce', 't2', 'flair'],
            transform=val_transforms
        )
        in_channels = 4
    else:
        if args.modality is None:
            raise ValueError("Must specify --modality when --multi-modal is False")
        logger.info(f"Using single modality: {args.modality}")
        logger.info("Input shape: (1, D, H, W) - single channel")
        
        # Create single-modality datasets
        train_dataset = Volume3DDataset(
            data_root=str(data_root),
            split_file=str(train_csv),
            modality=args.modality,
            transform=train_transforms
        )
        
        val_dataset = Volume3DDataset(
            data_root=str(data_root),
            split_file=str(val_csv),
            modality=args.modality,
            transform=val_transforms
        )
        in_channels = 1
    
    logger.info(f"Train dataset: {len(train_dataset)} patients")
    logger.info(f"Val dataset: {len(val_dataset)} patients")
    
    # Compute class counts for logging
    train_labels = train_dataset.get_all_labels()
    class_counts = [train_labels.count(0), train_labels.count(1)]  # [LGG, HGG]
    logger.info(f"Class counts: LGG={class_counts[0]}, HGG={class_counts[1]}")
    
    # Setup data loaders
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)
    
    # Class balancing: Use WeightedRandomSampler (data-level balancing)
    # Simplified approach: Only WeightedRandomSampler, no loss-level balancing
    train_sampler = get_weighted_sampler(train_labels, strategy='inverse_freq', seed=args.seed)
    logger.info("Using WeightedRandomSampler for training (class balancing enabled)")
    logger.info("Note: Simplified to data-level balancing only for improved stability")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Use sampler
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False
    )
    
    logger.info(f"Data loaders created (num_workers={num_workers})")
    
    # Create model
    logger.info("Creating Swin UNETR Encoder Classifier...")
    logger.info("Model architecture: Swin UNETR Encoder (MONAI)")
    
    # Create Swin UNETR encoder classifier
    logger.info(f"Creating Swin UNETR with {in_channels} input channels")
    logger.info(f"Feature size: {args.feature_size}, Depths: {args.depths}, Heads: {args.num_heads}")
    model = SwinUNETREncoderClassifier(
        img_size=(128, 128, 128),
        in_channels=in_channels,
        num_classes=2,
        feature_size=args.feature_size,
        depths=args.depths,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_checkpoint=args.use_checkpoint,
        logger=logger
    )
    model = model.to(device)
    
    # Multi-GPU support
    if args.multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created: {total_params/1e6:.2f}M total params, {trainable_params/1e6:.2f}M trainable")
    
    # Build loss function: CrossEntropyLoss (simplified from LDAM+DRW for stability)
    logger.info("Building CrossEntropyLoss (simplified for stability)")
    logger.info("Using WeightedRandomSampler for class balancing (data-level)")
    logger.info("Note: Removed LDAM+DRW to eliminate loss-level conflicts and improve stability")
    
    # Optional: Compute simple class weights for loss (not DRW, just static weights)
    # This is optional - WeightedRandomSampler already balances at data level
    use_class_weights_in_loss = False  # Set to True if you want loss-level weights too
    if use_class_weights_in_loss:
        n_total = sum(class_counts)
        n_minority = min(class_counts)
        n_majority = max(class_counts)
        # Simple inverse frequency weights
        class_weights = torch.tensor([
            n_total / (2 * n_minority),  # Weight for minority class (LGG)
            n_total / (2 * n_majority)   # Weight for majority class (HGG)
        ], dtype=torch.float32, device=device)
        logger.info(f"Using class weights in loss: {class_weights.cpu().numpy()}")
    else:
        class_weights = None
        logger.info("Not using class weights in loss (WeightedRandomSampler handles balancing)")
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    # Differential learning rates for encoder vs classifier
    # Encoder: 5e-5 (lower, more conservative)
    # Classifier: 1e-4 (higher, can adapt faster)
    classifier_lr = args.classifier_lr
    
    # Separate parameter groups
    backbone_params = list(model.get_backbone_params())
    classifier_params = list(model.get_classifier_params())
    
    logger.info(f"Encoder parameters: {sum(p.numel() for p in backbone_params)/1e6:.2f}M")
    logger.info(f"Classifier parameters: {sum(p.numel() for p in classifier_params)/1e6:.2f}M")
    logger.info(f"Encoder LR: {args.lr:.2e}, Classifier LR: {classifier_lr:.2e}")
    
    # Freeze encoder if requested
    if args.freeze_backbone_epochs > 0:
        logger.info(f"Freezing encoder for first {args.freeze_backbone_epochs} epochs")
        for param in backbone_params:
            param.requires_grad = False
    
    # Optimizer with differential learning rates
    if args.optimizer == 'adamw':
        # AdamW with different LRs for encoder and classifier
        optimizer = torch.optim.AdamW(
            [
                {'params': backbone_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
                {'params': classifier_params, 'lr': classifier_lr, 'weight_decay': args.weight_decay}
            ],
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif args.optimizer == 'adam':
        # Adam with different LRs for encoder and classifier
        optimizer = torch.optim.Adam(
            [
                {'params': backbone_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
                {'params': classifier_params, 'lr': classifier_lr, 'weight_decay': args.weight_decay}
            ],
            betas=(0.9, 0.999),
            eps=1e-8
        )
    else:
        # SGD with Nesterov momentum
        optimizer = torch.optim.SGD(
            [
                {'params': backbone_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
                {'params': classifier_params, 'lr': classifier_lr, 'weight_decay': args.weight_decay}
            ],
            momentum=0.9,
            nesterov=True
        )
    
    # Learning rate scheduler with warmup
    # For differential LRs, we need to apply scheduler to each parameter group separately
    if args.scheduler == 'cosine':
        # Cosine annealing with warmup for better initial convergence
        warmup_epochs = max(5, args.epochs // 10)  # 10% of epochs for warmup (min 5)
        
        def lr_lambda_backbone(epoch):
            """LR schedule for encoder - more conservative"""
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                # Guard against division by zero when epochs <= warmup_epochs
                if args.epochs <= warmup_epochs:
                    return 1.0  # Constant LR, skip cosine decay
                progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
                # More conservative decay for encoder
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        def lr_lambda_classifier(epoch):
            """LR schedule for classifier - can decay faster"""
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                # Guard against division by zero when epochs <= warmup_epochs
                if args.epochs <= warmup_epochs:
                    return 1.0  # Constant LR, skip cosine decay
                progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        # Create scheduler with separate lambdas for each parameter group
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[lr_lambda_backbone, lr_lambda_classifier]
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
    
    # Exponential Moving Average (EMA) for model weights
    ema_model = None
    if args.ema_decay > 0:
        logger.info(f"Exponential Moving Average enabled with decay={args.ema_decay}")
        # Create EMA model with same architecture as main model
        # Use same in_channels to support multi-modal mode
        logger.info(f"Creating EMA model with {in_channels} input channels (matching main model)")
        ema_model = SwinUNETREncoderClassifier(
            img_size=(128, 128, 128),
            in_channels=in_channels,
            num_classes=2,
            feature_size=args.feature_size,
            depths=args.depths,
            num_heads=args.num_heads,
            dropout=args.dropout,
            use_checkpoint=args.use_checkpoint,
            logger=logger
        )
        # Get state dict from main model (handle DataParallel wrapping)
        if isinstance(model, DataParallel):
            main_model_state = model.module.state_dict()
        else:
            main_model_state = model.state_dict()
        # Load state dict from main model (after weight adaptation)
        ema_model.load_state_dict(main_model_state)
        ema_model = ema_model.to(device)
        ema_model.eval()  # EMA model always in eval mode
        if args.multi_gpu and torch.cuda.device_count() > 1:
            ema_model = DataParallel(ema_model)
    else:
        logger.info("Exponential Moving Average disabled")
    
    # Gradient accumulation
    if args.gradient_accumulation_steps > 1:
        logger.info(f"Gradient accumulation enabled: {args.gradient_accumulation_steps} steps")
        logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    else:
        logger.info("Gradient accumulation disabled")
    
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
        'lr': []
    }
    
    best_val_auc = -np.inf
    best_val_f1 = -np.inf
    
    # DRY-RUN VERIFICATION: Load one batch and verify shapes before training
    logger.info("\n" + "="*60)
    logger.info("DRY-RUN VERIFICATION: Checking data pipeline")
    logger.info("="*60)
    try:
        # Get one batch from training loader
        sample_batch = next(iter(train_loader))
        volumes, labels, patient_ids = sample_batch
        
        # Verify shapes
        batch_size = volumes.shape[0]
        expected_shape = (batch_size, in_channels, 128, 128, 128)
        
        logger.info(f"Sample batch - Volume shape: {volumes.shape}")
        logger.info(f"Sample batch - Labels shape: {labels.shape}")
        logger.info(f"Sample batch - Expected volume shape: {expected_shape}")
        logger.info(f"Sample batch - Number of patient IDs: {len(patient_ids)}")
        
        # Authoritative shape check
        assert volumes.ndim == 5, f"Volume must be 5D (B, C, D, H, W), got {volumes.ndim}D with shape {volumes.shape}"
        assert volumes.shape[0] == batch_size, f"Batch size mismatch: expected {batch_size}, got {volumes.shape[0]}"
        assert volumes.shape[1] == in_channels, f"Channel mismatch: expected {in_channels} channels, got {volumes.shape[1]}. This indicates a dataset/transform issue."
        assert volumes.shape[2:] == (128, 128, 128), f"Spatial dimensions mismatch: expected (128, 128, 128), got {volumes.shape[2:]}"
        assert labels.ndim == 1, f"Labels must be 1D (batch_size,), got {labels.ndim}D with shape {labels.shape}"
        assert labels.shape[0] == batch_size, f"Label batch size mismatch: expected {batch_size}, got {labels.shape[0]}"
        
        logger.info("✓ Shape verification PASSED")
        logger.info(f"✓ Volume shape: {volumes.shape} (correct)")
        logger.info(f"✓ Labels shape: {labels.shape} (correct)")
        logger.info(f"✓ Data pipeline is ready for training")
        logger.info("="*60 + "\n")
    except Exception as e:
        logger.error(f"\n{'='*60}")
        logger.error("DRY-RUN VERIFICATION FAILED")
        logger.error(f"{'='*60}")
        logger.error(f"Error: {e}")
        logger.error(f"Volume shape (if available): {volumes.shape if 'volumes' in locals() else 'N/A'}")
        logger.error(f"Expected shape: (batch_size, {in_channels}, 128, 128, 128)")
        logger.error("\nThis indicates a structural issue in the data pipeline.")
        logger.error("Please check:")
        logger.error("  1. Dataset __getitem__ returns correct shape (4, D, H, W)")
        logger.error("  2. Transforms preserve channel dimension")
        logger.error("  3. DataLoader collates correctly")
        logger.error(f"{'='*60}\n")
        raise
    
    logger.info("Starting training...")
    logger.info(f"Total epochs: {args.epochs}")
    logger.info(f"Early stopping patience: {args.early_stopping}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler, epoch, logger, 
            args.grad_clip, args.gradient_accumulation_steps, ema_model, args.ema_decay
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, loss_fn, device, epoch, logger)
        
        # Unfreeze encoder if freeze period ended
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            logger.info(f"Unfreezing encoder at epoch {epoch}")
            for param in backbone_params:
                param.requires_grad = True
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            # Get LR from backbone group (or first group)
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
        
        # Log metrics with clear formatting
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"EPOCH {epoch} METRICS:")
        logger.info("=" * 80)
        logger.info(f"  TRAIN:")
        logger.info(f"    Loss:            {train_loss:.6f}")
        logger.info(f"    Accuracy:        {train_acc:.4f}")
        logger.info("")
        logger.info(f"  VALIDATION:")
        logger.info(f"    Loss:            {val_loss:.6f}")
        logger.info(f"    Accuracy:        {val_metrics['accuracy']:.4f}")
        logger.info(f"    Precision:       {val_metrics['precision']:.4f}")
        logger.info(f"    Recall:          {val_metrics['recall']:.4f}")
        logger.info(f"    F1-Score:         {val_metrics['f1']:.4f}")
        logger.info(f"    AUC-ROC:         {val_metrics['auc']:.4f}")
        logger.info("")
        logger.info(f"  TRAINING:")
        logger.info(f"    Learning Rate:   {current_lr:.6f}")
        logger.info("=" * 80)
        logger.info("")
        
        # Save plots after each epoch
        save_plots(run_dir, history, val_metrics, logger, epoch=epoch)
        
        # Save checkpoint (include loss history)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'val_metrics': val_metrics,
            'history': history,
            'config': config,
            'train_loss': train_loss,  # Current epoch train loss
            'val_loss': val_loss,      # Current epoch val loss
            'loss_type': 'CrossEntropyLoss'  # Loss type information
        }
        
        # Save last checkpoint
        torch.save(checkpoint, checkpoints_dir / 'last.pt')
        
        # Save best checkpoint (by val AUC, tie-break by F1)
        # CRITICAL: This saves the model state at the epoch with highest validation AUC
        if val_metrics['auc'] > best_val_auc or (val_metrics['auc'] == best_val_auc and val_metrics['f1'] > best_val_f1):
            best_val_auc = val_metrics['auc']
            best_val_f1 = val_metrics['f1']
            
            # Update checkpoint with current best metrics before saving
            checkpoint['val_metrics'] = val_metrics  # Ensure checkpoint has latest best metrics
            checkpoint['best_val_auc'] = best_val_auc  # Store best AUC explicitly
            checkpoint['best_val_f1'] = best_val_f1  # Store best F1 explicitly
            
            # Save regular model checkpoint (best by val AUC)
            checkpoint['model_state_dict'] = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
            best_checkpoint_path = checkpoints_dir / 'best.pt'
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(f"✓ Saved BEST regular checkpoint: {best_checkpoint_path}")
            logger.info(f"  Epoch: {epoch}, Val AUC: {best_val_auc:.4f}, Val F1: {best_val_f1:.4f}")
            
            # Also save EMA model if enabled (best EMA by val AUC)
            if ema_model is not None and args.ema_decay > 0:
                ema_checkpoint = checkpoint.copy()
                ema_checkpoint['model_state_dict'] = ema_model.module.state_dict() if isinstance(ema_model, DataParallel) else ema_model.state_dict()
                ema_checkpoint['is_ema'] = True
                best_ema_checkpoint_path = checkpoints_dir / 'best_ema.pt'
                torch.save(ema_checkpoint, best_ema_checkpoint_path)
                logger.info(f"✓ Saved BEST EMA checkpoint: {best_ema_checkpoint_path}")
                logger.info(f"  Epoch: {epoch}, Val AUC: {best_val_auc:.4f}, Val F1: {best_val_f1:.4f}")
            else:
                logger.info(f"  (EMA disabled or not available)")
        
        # Early stopping
        if early_stopping(epoch, val_metrics):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            logger.info(f"Best epoch: {early_stopping.best_epoch}, Best AUC: {early_stopping.best_score:.4f}")
            break
    
    # Final evaluation
    logger.info("\n" + "="*80)
    logger.info("Training completed!")
    logger.info("="*80)
    logger.info(f"Best epoch during training: {early_stopping.best_epoch}")
    logger.info(f"Best validation AUC during training: {early_stopping.best_score:.4f}")
    logger.info("="*80)
    
    # CRITICAL: Load the BEST checkpoint explicitly (by val AUC)
    # Policy: If EMA is enabled and best_ema.pt exists, use EMA best checkpoint
    # Otherwise, use regular best.pt checkpoint
    # NEVER use last.pt for final evaluation
    
    best_checkpoint_path = checkpoints_dir / 'best.pt'
    best_ema_checkpoint_path = checkpoints_dir / 'best_ema.pt'
    
    # Determine which checkpoint to load
    use_ema = False
    if args.ema_decay > 0 and best_ema_checkpoint_path.exists():
        # EMA is enabled and EMA best checkpoint exists - use it
        use_ema = True
        checkpoint_path = best_ema_checkpoint_path
        logger.info(f"Loading BEST EMA checkpoint: {checkpoint_path}")
        logger.info("Policy: Using EMA best checkpoint for final evaluation")
    elif best_checkpoint_path.exists():
        # Use regular best checkpoint
        use_ema = False
        checkpoint_path = best_checkpoint_path
        logger.info(f"Loading BEST regular checkpoint: {checkpoint_path}")
        logger.info("Policy: Using non-EMA best checkpoint for final evaluation")
    else:
        # Fallback: best checkpoint doesn't exist (shouldn't happen, but handle gracefully)
        logger.error(f"ERROR: Best checkpoint not found at {best_checkpoint_path}")
        logger.error("Cannot perform final evaluation without best checkpoint")
        raise FileNotFoundError(f"Best checkpoint not found: {best_checkpoint_path}")
    
    # Load the checkpoint
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Verify checkpoint contains expected keys
    if 'model_state_dict' not in best_checkpoint:
        raise ValueError(f"Checkpoint missing 'model_state_dict': {checkpoint_path}")
    if 'val_metrics' not in best_checkpoint:
        raise ValueError(f"Checkpoint missing 'val_metrics': {checkpoint_path}")
    
    # Log checkpoint information
    checkpoint_epoch = best_checkpoint.get('epoch', 'unknown')
    checkpoint_val_auc = best_checkpoint.get('val_metrics', {}).get('auc', 'unknown')
    checkpoint_is_ema = best_checkpoint.get('is_ema', False)
    
    logger.info(f"Checkpoint info:")
    logger.info(f"  Epoch: {checkpoint_epoch}")
    logger.info(f"  Val AUC: {checkpoint_val_auc}")
    logger.info(f"  Is EMA: {checkpoint_is_ema}")
    logger.info(f"  Expected best epoch: {early_stopping.best_epoch}")
    logger.info(f"  Expected best AUC: {early_stopping.best_score:.4f}")
    
    # Verify checkpoint matches best epoch (warning if mismatch)
    if checkpoint_epoch != early_stopping.best_epoch:
        logger.warning(f"WARNING: Checkpoint epoch ({checkpoint_epoch}) != best epoch ({early_stopping.best_epoch})")
        logger.warning("This may indicate a checkpoint saving issue")
    
    # Load weights into appropriate model
    if use_ema and ema_model is not None:
        # Load EMA model weights
        logger.info("Loading weights into EMA model...")
        eval_model = ema_model
        if isinstance(eval_model, DataParallel):
            eval_model.module.load_state_dict(best_checkpoint['model_state_dict'])
        else:
            eval_model.load_state_dict(best_checkpoint['model_state_dict'])
        logger.info("✓ EMA model weights loaded")
    else:
        # Load regular model weights
        logger.info("Loading weights into regular model...")
        eval_model = model
        if isinstance(eval_model, DataParallel):
            eval_model.module.load_state_dict(best_checkpoint['model_state_dict'])
        else:
            eval_model.load_state_dict(best_checkpoint['model_state_dict'])
        logger.info("✓ Regular model weights loaded")
    
    # CRITICAL: Ensure model is in eval mode
    eval_model.eval()
    logger.info("✓ Model set to eval() mode")
    
    # Perform final evaluation
    logger.info("")
    logger.info("="*80)
    logger.info("FINAL EVALUATION: Evaluating best checkpoint on validation set")
    logger.info(f"Checkpoint: {checkpoint_path.name}")
    logger.info(f"Model type: {'EMA' if use_ema else 'Regular'}")
    logger.info("="*80)
    _, final_metrics = validate(eval_model, val_loader, loss_fn, device, args.epochs, logger)
    
    # Add training history and loss information to final metrics
    final_metrics['training_history'] = {
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc'],
        'val_precision': history['val_precision'],
        'val_recall': history['val_recall'],
        'val_f1': history['val_f1'],
        'val_auc': history['val_auc'],
        'lr': history['lr']
    }
    
    # Add loss summary statistics
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
            'best_epoch': int(early_stopping.best_epoch),
            'best_value': float(history['val_loss'][early_stopping.best_epoch - 1]) if early_stopping.best_epoch > 0 else None
        }
    }
    
    # Add loss type information
    final_metrics['loss_info'] = {
        'loss_type': 'CrossEntropyLoss',
        'class_balancing': 'WeightedRandomSampler (data-level)',
        'note': 'Simplified loss function for improved training stability'
    }
    
    # Add checkpoint information used for final evaluation
    final_metrics['checkpoint_info'] = {
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_name': checkpoint_path.name,
        'checkpoint_epoch': int(checkpoint_epoch) if isinstance(checkpoint_epoch, (int, float)) else None,
        'checkpoint_val_auc': float(checkpoint_val_auc) if isinstance(checkpoint_val_auc, (int, float)) else None,
        'is_ema': bool(use_ema),
        'best_epoch': int(early_stopping.best_epoch),
        'best_val_auc': float(early_stopping.best_score)
    }
    
    # Save final metrics with full history
    # Convert NumPy types to JSON-serializable Python types
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
    
    # Verify evaluation AUC matches checkpoint AUC (should be very close, allow small numerical differences)
    if isinstance(checkpoint_val_auc, (int, float)) and isinstance(final_metrics['auc'], (int, float)):
        auc_diff = abs(final_metrics['auc'] - checkpoint_val_auc)
        if auc_diff > 0.01:  # More than 1% difference is suspicious
            logger.warning(f"  WARNING: Evaluation AUC ({final_metrics['auc']:.4f}) differs from checkpoint AUC ({checkpoint_val_auc:.4f})")
            logger.warning(f"  Difference: {auc_diff:.4f} - This may indicate an evaluation issue")
        else:
            logger.info(f"  ✓ Evaluation AUC matches checkpoint AUC (diff: {auc_diff:.4f})")
    logger.info("")
    logger.info("LOSS SUMMARY (CrossEntropyLoss):")
    logger.info(f"  Train Loss:")
    logger.info(f"    Final:           {final_metrics['loss_summary']['train_loss']['final']:.6f}")
    logger.info(f"    Min:             {final_metrics['loss_summary']['train_loss']['min']:.6f}")
    logger.info(f"    Max:             {final_metrics['loss_summary']['train_loss']['max']:.6f}")
    logger.info(f"    Mean:            {final_metrics['loss_summary']['train_loss']['mean']:.6f}")
    logger.info(f"    Std:             {final_metrics['loss_summary']['train_loss']['std']:.6f}")
    logger.info(f"  Val Loss:")
    logger.info(f"    Final:           {final_metrics['loss_summary']['val_loss']['final']:.6f}")
    logger.info(f"    Best (Epoch {final_metrics['loss_summary']['val_loss']['best_epoch']}): {final_metrics['loss_summary']['val_loss']['best_value']:.6f}" if final_metrics['loss_summary']['val_loss']['best_value'] is not None else f"    Best: N/A")
    logger.info(f"    Min:             {final_metrics['loss_summary']['val_loss']['min']:.6f}")
    logger.info(f"    Max:             {final_metrics['loss_summary']['val_loss']['max']:.6f}")
    logger.info(f"    Mean:            {final_metrics['loss_summary']['val_loss']['mean']:.6f}")
    logger.info(f"    Std:             {final_metrics['loss_summary']['val_loss']['std']:.6f}")
    logger.info("")
    logger.info(f"Results saved to: {run_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

