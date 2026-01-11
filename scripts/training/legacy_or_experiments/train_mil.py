#!/usr/bin/env python3
"""
Dual-Stream MIL Training Script for BraTS2018

This script trains a Dual-Stream Multiple Instance Learning model for brain tumor
classification (HGG vs LGG) with entropy-based slice selection, LDAM loss, and DRW.

IMPORTANT: Entropy-based slice selection is ALWAYS enabled (not optional).

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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
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

from utils.mil_dataset import MILDataset
from utils.ldam_loss import build_loss_fn
from utils.class_balancing import get_weighted_sampler
from models.dual_stream_mil.model import create_dual_stream_mil


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


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, pos_label: int = 1, metric: str = "f1") -> Tuple[float, Dict]:
    """
    Find optimal threshold for binary classification.
    
    Args:
        y_true: True labels (0 or 1)
        y_prob: Predicted probabilities for positive class (1D array)
        pos_label: Positive class label (default: 1 for HGG)
        metric: Optimization metric ('f1' for F1 score)
    
    Returns:
        (best_threshold, threshold_sweep_dict)
        - best_threshold: Optimal threshold value
        - threshold_sweep_dict: Dictionary mapping thresholds to metrics
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    threshold_sweep = []
    best_f1 = -1.0
    best_recall = -1.0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred_thresh = (y_prob >= threshold).astype(int)
        
        prec = precision_score(y_true, y_pred_thresh, average='binary', pos_label=pos_label, zero_division=0)
        rec = recall_score(y_true, y_pred_thresh, average='binary', pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_true, y_pred_thresh, average='binary', pos_label=pos_label, zero_division=0)
        acc = accuracy_score(y_true, y_pred_thresh)
        
        threshold_sweep.append({
            'threshold': float(threshold),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'accuracy': float(acc)
        })
        
        # Select best: primary = max F1, tie-breaker = max recall
        if f1 > best_f1 or (f1 == best_f1 and rec > best_recall):
            best_f1 = f1
            best_recall = rec
            best_threshold = threshold
    
    return float(best_threshold), threshold_sweep


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, pos_label: int = 1, threshold: float = 0.5) -> Dict:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels (0=LGG, 1=HGG)
        y_pred: Predictions at given threshold
        y_probs: Predicted probabilities (shape [n_samples, 2] or [n_samples] for positive class)
        pos_label: Positive class label (1=HGG)
        threshold: Decision threshold used for y_pred
    
    Returns:
        Dictionary of metrics
    """
    # Ensure y_probs is 2D, extract positive class probabilities
    if y_probs.ndim == 1:
        y_prob_pos = y_probs
    else:
        y_prob_pos = y_probs[:, pos_label]
    
    metrics = {}
    metrics['threshold'] = threshold
    
    # Accuracy
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    
    # Binary metrics with pos_label=1 (HGG is positive class)
    metrics['precision'] = float(precision_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0))
    
    # Per-class metrics (for detailed analysis)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['precision_per_class'] = precision_per_class.tolist()
    metrics['recall_per_class'] = recall_per_class.tolist()
    metrics['f1_per_class'] = f1_per_class.tolist()
    
    # AUC-ROC (uses probabilities, not predictions)
    try:
        metrics['auc'] = float(roc_auc_score(y_true, y_prob_pos))
    except ValueError:
        metrics['auc'] = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0, target_names=['LGG', 'HGG'])
    metrics['classification_report'] = report
    
    return metrics


def save_plots(output_dir: Path, history: Dict, metrics: Dict, logger):
    """Save training plots."""
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Learning curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Val AUC
    axes[0, 1].plot(history['val_auc'], label='Val AUC', marker='o', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC-ROC')
    axes[0, 1].set_title('Validation AUC-ROC')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # Val F1
    axes[1, 0].plot(history['val_f1'], label='Val F1', marker='o', color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # Accuracy
    axes[1, 1].plot(history['train_acc'], label='Train Acc', marker='o', alpha=0.7)
    axes[1, 1].plot(history['val_acc'], label='Val Acc', marker='s', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Training and Validation Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved learning curves to {plots_dir / 'learning_curves.png'}")
    
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
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontweight='bold')
        plt.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {plots_dir / 'confusion_matrix.png'}")
    
    # ROC curve (if we have probabilities)
    if 'val_probs' in metrics:
        from sklearn.metrics import roc_curve
        y_true = metrics['val_labels']
        y_probs = metrics['val_probs'][:, 1]  # Probability of positive class
        
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
        plt.savefig(plots_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved ROC curve to {plots_dir / 'roc_curve.png'}")


def main():
    parser = argparse.ArgumentParser(description="Train Dual-Stream MIL model")
    
    # Required args
    parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3, 4],
                       help='Fold number (0-4)')
    
    # Data args
    parser.add_argument('--modality', type=str, default='flair', choices=['t1', 't1ce', 't2', 'flair'],
                       help='Modality to use (default: flair)')
    parser.add_argument('--top-k', type=int, default=16,
                       help='Number of top-k slices for entropy selection (default: 16)')
    parser.add_argument('--data-root', type=str, default='data/processed/stage_4_resize/train',
                       help='Root directory for processed data')
    parser.add_argument('--entropy-dir', type=str, default='data/entropy',
                       help='Directory containing entropy JSON files')
    parser.add_argument('--splits-dir', type=str, default='splits',
                       help='Directory containing split CSV files')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=30,
                       help='Maximum number of epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate (default: 0.5)')
    
    # LDAM + DRW args
    parser.add_argument('--drw-start-epoch', type=int, default=15,
                       help='Epoch to start DRW re-weighting (default: 15)')
    parser.add_argument('--max-m', type=float, default=0.5,
                       help='Maximum margin for LDAM (default: 0.5)')
    parser.add_argument('--s', type=float, default=30,
                       help='Scaling factor for LDAM (default: 30)')
    
    # Optimizer/Scheduler args
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'none'],
                       help='Learning rate scheduler (default: cosine)')
    parser.add_argument('--grad-clip', type=float, default=0.0,
                       help='Gradient clipping (0.0 = disabled, default: 0.0)')
    
    # Early stopping args
    parser.add_argument('--early-stopping-patience', type=int, default=7,
                       help='Early stopping patience (default: 7)')
    parser.add_argument('--early-stopping-min-epochs', type=int, default=10,
                       help='Minimum epochs before early stopping (default: 10)')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.0,
                       help='Minimum delta for early stopping (default: 0.0)')
    
    # Data loading args
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of data loader workers (default: auto)')
    parser.add_argument('--use-balanced-sampler', action='store_true', default=True,
                       help='Use WeightedRandomSampler for class balancing (default: True)')
    parser.add_argument('--no-balanced-sampler', dest='use_balanced_sampler', action='store_false',
                       help='Disable WeightedRandomSampler')
    
    # Performance args
    parser.add_argument('--amp', action='store_true', default=True,
                       help='Use mixed precision training (default: True)')
    parser.add_argument('--no-amp', dest='amp', action='store_false',
                       help='Disable mixed precision training')
    parser.add_argument('--tf32', action='store_true', default=True,
                       help='Enable TF32 (default: True)')
    parser.add_argument('--no-tf32', dest='tf32', action='store_false',
                       help='Disable TF32')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile (experimental)')
    
    # General args
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='results/MIL',
                       help='Output directory (default: results/MIL)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto/cuda/cpu, default: auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
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
    logger.info(f"Starting MIL training for fold {args.fold}")
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
    config['class_mapping'] = {'LGG': 0, 'HGG': 1}
    config['entropy_settings'] = {'top_k': args.top_k, 'axis': 'axial'}
    
    with open(config_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_dir / 'config.json'}")
    
    # Load datasets
    project_root = Path(__file__).parent.parent.parent
    splits_dir = project_root / args.splits_dir
    data_root = project_root / args.data_root
    entropy_dir = project_root / args.entropy_dir
    
    train_csv = splits_dir / f'fold_{args.fold}_train.csv'
    val_csv = splits_dir / f'fold_{args.fold}_val.csv'
    
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Split files not found: {train_csv}, {val_csv}")
    
    logger.info(f"Loading datasets from {data_root}")
    logger.info(f"Train CSV: {train_csv}")
    logger.info(f"Val CSV: {val_csv}")
    logger.info(f"Entropy directory: {entropy_dir}")
    logger.info(f"IMPORTANT: Entropy-based slice selection is ALWAYS ENABLED (top-k={args.top_k})")
    
    # Create datasets (entropy ALWAYS enabled)
    train_dataset = MILDataset(
        split_csv=str(train_csv),
        data_root=str(data_root),
        modality=args.modality,
        use_entropy=True,  # ALWAYS enabled
        entropy_dir=str(entropy_dir),
        transform=None  # Augmentation handled separately if needed
    )
    
    val_dataset = MILDataset(
        split_csv=str(val_csv),
        data_root=str(data_root),
        modality=args.modality,
        use_entropy=True,  # ALWAYS enabled
        entropy_dir=str(entropy_dir),
        transform=None
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} patients")
    logger.info(f"Val dataset: {len(val_dataset)} patients")
    
    # Compute class counts for LDAM
    train_labels = [train_dataset.get_class_label(i) for i in range(len(train_dataset))]
    class_counts = [train_labels.count(0), train_labels.count(1)]  # [LGG, HGG]
    logger.info(f"Class counts: LGG={class_counts[0]}, HGG={class_counts[1]}")
    
    # Setup data loaders
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)
    
    # Class balancing: Use WeightedRandomSampler by default for imbalanced data
    # This helps prevent model collapse to majority class (HGG)
    train_sampler = None
    if args.use_balanced_sampler:
        train_sampler = get_weighted_sampler(train_labels, strategy='inverse_freq', seed=args.seed)
        logger.info("Using WeightedRandomSampler for training (class balancing enabled)")
    else:
        logger.info("WeightedRandomSampler disabled (may lead to class imbalance bias)")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
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
    logger.info("Creating Dual-Stream MIL model...")
    model = create_dual_stream_mil(num_classes=2, pretrained_encoder=False, dropout=args.dropout)
    model = model.to(device)
    
    if args.compile and hasattr(torch, 'compile'):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created: {total_params/1e6:.2f}M total params, {trainable_params/1e6:.2f}M trainable")
    
    # Setup loss function (LDAM + DRW)
    loss_fn = build_loss_fn(
        num_classes=2,
        class_counts=class_counts,
        max_m=args.max_m,
        s=args.s,
        drw_start_epoch=args.drw_start_epoch,
        device=str(device)
    )
    logger.info(f"LDAM loss configured (max_m={args.max_m}, s={args.s}, drw_start_epoch={args.drw_start_epoch})")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Setup scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//3, gamma=0.1)
    else:
        scheduler = None
    
    # Setup mixed precision
    scaler = GradScaler() if args.amp else None
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
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
        'val_auc': [],
        'val_f1': [],
        'lr': []
    }
    
    best_val_auc = -np.inf
    best_val_f1 = -np.inf
    best_epoch = 0
    
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (bags, labels) in enumerate(train_loader):
            bags = bags.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if args.amp:
                with autocast():
                    logits, _ = model(bags)  # Model returns (logits, attention)
                    loss = loss_fn(logits, labels, epoch)
                
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(bags)  # Model returns (logits, attention)
                loss = loss_fn(logits, labels, epoch)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels_list = []
        val_probs = []
        
        with torch.no_grad():
            for bags, labels in val_loader:
                bags = bags.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                if args.amp:
                    with autocast():
                        logits, _ = model(bags)  # Model returns (logits, attention)
                        loss = loss_fn(logits, labels, epoch)
                else:
                    logits, _ = model(bags)  # Model returns (logits, attention)
                    loss = loss_fn(logits, labels, epoch)
                
                val_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels_list)
        val_probs = np.array(val_probs)
        
        # Compute metrics (binary, pos_label=1 for HGG)
        val_metrics = compute_metrics(val_labels, val_preds, val_probs, pos_label=1, threshold=0.5)
        val_acc = val_metrics['accuracy']
        val_auc = val_metrics['auc']
        val_f1 = val_metrics['f1']
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = args.lr
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        # Log epoch results
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.6f}"
        )
        
        # Check for best model
        is_best = False
        if val_auc > best_val_auc or (val_auc == best_val_auc and val_f1 > best_val_f1):
            best_val_auc = val_auc
            best_val_f1 = val_f1
            best_epoch = epoch
            is_best = True
        
        # Save checkpoints
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'val_auc': val_auc,
            'val_f1': val_f1,
            'best_val_auc': best_val_auc,
            'best_epoch': best_epoch,
            'config': config
        }
        
        # Save last checkpoint
        torch.save(checkpoint, checkpoints_dir / 'last.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoints_dir / 'best.pt')
            logger.info(f"Saved best model (Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f})")
        
        # Early stopping check
        if early_stopping(epoch, {'auc': val_auc, 'f1': val_f1}):
            logger.info(f"Early stopping triggered at epoch {epoch+1} (best epoch: {best_epoch+1})")
            break
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time/60:.2f} minutes")
    logger.info(f"Best epoch: {best_epoch+1} (Val AUC: {best_val_auc:.4f}, Val F1: {best_val_f1:.4f})")
    
    # Final evaluation and saving (ALWAYS executed, whether early stopping or max epochs)
    logger.info("Loading best model for final evaluation...")
    
    # Load best checkpoint with weights_only=False for compatibility
    try:
        best_checkpoint = torch.load(checkpoints_dir / 'best.pt', map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        best_checkpoint = torch.load(checkpoints_dir / 'best.pt', map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Final validation evaluation
    model.eval()
    final_val_preds = []
    final_val_labels = []
    final_val_probs = []
    
    with torch.no_grad():
        for bags, labels in val_loader:
            bags = bags.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if args.amp:
                with autocast():
                    logits, _ = model(bags)  # Model returns (logits, attention)
            else:
                logits, _ = model(bags)  # Model returns (logits, attention)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            final_val_preds.extend(preds.cpu().numpy())
            final_val_labels.extend(labels.cpu().numpy())
            final_val_probs.extend(probs.cpu().numpy())
    
    final_val_preds = np.array(final_val_preds)
    final_val_labels = np.array(final_val_labels)
    final_val_probs = np.array(final_val_probs)
    
    # Extract probabilities for positive class (HGG)
    final_val_prob_pos = final_val_probs[:, 1]
    
    # Compute metrics at threshold=0.5 (for reference)
    logger.info("Computing metrics using binary pos_label=1 (HGG)")
    preds_thr05 = (final_val_prob_pos >= 0.5).astype(int)
    metrics_thr05 = compute_metrics(final_val_labels, preds_thr05, final_val_probs, pos_label=1, threshold=0.5)
    logger.info(f"Threshold@0.5: Acc={metrics_thr05['accuracy']:.4f}, Prec={metrics_thr05['precision']:.4f}, Rec={metrics_thr05['recall']:.4f}, F1={metrics_thr05['f1']:.4f}, AUC={metrics_thr05['auc']:.4f}")
    
    # Find optimal threshold (F1, tie-breaker: recall)
    logger.info("Finding optimal threshold (F1 maximization, tie-breaker: recall)...")
    best_threshold, threshold_sweep = find_optimal_threshold(final_val_labels, final_val_prob_pos, pos_label=1, metric="f1")
    logger.info(f"Best threshold (F1, tie recall): {best_threshold:.4f}")
    
    # Compute predictions at optimal threshold
    final_val_preds_optimal = (final_val_prob_pos >= best_threshold).astype(int)
    
    # Compute metrics at optimal threshold
    final_metrics = compute_metrics(final_val_labels, final_val_preds_optimal, final_val_probs, pos_label=1, threshold=best_threshold)
    logger.info(f"Optimized metrics: Acc={final_metrics['accuracy']:.4f}, Prec={final_metrics['precision']:.4f}, Rec={final_metrics['recall']:.4f}, F1={final_metrics['f1']:.4f}, AUC={final_metrics['auc']:.4f}")
    
    # Add reference metrics at 0.5
    final_metrics['metrics_at_threshold_0.5'] = metrics_thr05
    final_metrics['val_probs'] = final_val_probs.tolist()
    final_metrics['val_labels'] = final_val_labels.tolist()
    
    # Save metrics
    with open(metrics_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_dir / 'metrics.json'}")
    
    # Save threshold analysis
    threshold_analysis = {
        'best_threshold': best_threshold,
        'optimization_metric': 'f1',
        'tie_breaker': 'recall',
        'threshold_sweep': threshold_sweep,
        'metrics_at_best_threshold': {
            'threshold': best_threshold,
            'accuracy': final_metrics['accuracy'],
            'precision': final_metrics['precision'],
            'recall': final_metrics['recall'],
            'f1': final_metrics['f1'],
            'auc': final_metrics['auc']
        },
        'metrics_at_threshold_0.5': {
            'threshold': 0.5,
            'accuracy': metrics_thr05['accuracy'],
            'precision': metrics_thr05['precision'],
            'recall': metrics_thr05['recall'],
            'f1': metrics_thr05['f1'],
            'auc': metrics_thr05['auc']
        }
    }
    with open(metrics_dir / 'threshold_analysis.json', 'w') as f:
        json.dump(threshold_analysis, f, indent=2)
    logger.info(f"Saved threshold analysis to {metrics_dir / 'threshold_analysis.json'}")
    
    # Save predictions (using optimal threshold)
    np.save(predictions_dir / 'val_probs.npy', final_val_probs)
    np.save(predictions_dir / 'val_preds.npy', final_val_preds_optimal)
    np.save(predictions_dir / 'val_labels.npy', final_val_labels)
    logger.info(f"Saved predictions to {predictions_dir} (predictions use optimal threshold {best_threshold:.4f})")
    
    # Save plots
    plot_metrics = final_metrics.copy()
    plot_metrics['val_probs'] = final_val_probs
    plot_metrics['val_labels'] = final_val_labels
    save_plots(run_dir, history, plot_metrics, logger)
    
    # Update latest run pointer
    latest_dir = Path(args.output_dir) / 'latest' / f'fold_{args.fold}'
    latest_dir.mkdir(parents=True, exist_ok=True)
    with open(latest_dir / 'LATEST_RUN.txt', 'w') as f:
        f.write(str(run_dir))
    logger.info(f"Updated latest run pointer: {latest_dir / 'LATEST_RUN.txt'}")
    
    # Print final summary
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    logger.info(f"Best Epoch: {best_epoch+1}")
    logger.info(f"Accuracy: {final_metrics['accuracy']:.4f}")
    logger.info(f"Precision: {final_metrics['precision']:.4f}")
    logger.info(f"Recall: {final_metrics['recall']:.4f}")
    logger.info(f"F1 Score: {final_metrics['f1']:.4f}")
    logger.info(f"AUC-ROC: {final_metrics['auc']:.4f}")
    logger.info(f"Run directory: {run_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

