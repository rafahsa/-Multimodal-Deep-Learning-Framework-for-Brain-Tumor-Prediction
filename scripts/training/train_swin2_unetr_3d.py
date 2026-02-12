#!/usr/bin/env python3
"""
Swin-2 UNETR-3D Training Script (Pilot Experiment)

This is a COMPLETELY SEPARATE experiment from Swin-1.
Swin-2 modifications:
- patch_size = 2 (memory constraint: patch_size=1 creates ~2M tokens, OOM in 3D attention)
- window_size = 4 (smaller local attention, preserves research intent for subtle patterns)
- feature_size = 24 (reduced from 48 for memory efficiency, must be divisible by 12)
- depths = [2, 2, 2, 1] (reduced from [2, 2, 2, 2] for memory efficiency)
- Focal Loss (alpha=0.25, gamma=2.0) for hard example focus
- Hard example mining (oversample Swin-1 FN cases)

ARCHITECTURAL MEMORY FIX:
- patch_size=1 is infeasible in 3D Swin: creates ~2M tokens, attention softmax OOM
- window_size=4 preserves research intent: smaller windows focus on local subtle patterns
  (complements Swin-1's global attention with local detail, still targets FN cases)

Author: Medical Imaging Pipeline
Isolation: This script does NOT modify or depend on Swin-1 code
"""

import sys
from pathlib import Path

# CRITICAL: Set project root explicitly BEFORE any imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Now safe to import project modules
from utils.dataset_3d_multi_modal import MultiModalVolume3DDataset
from utils.class_balancing import get_weighted_sampler
from utils.augmentations_3d import get_resnet3d_transforms_3d
from models.swin_unetr_encoder import SwinUNETREncoderClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance and hard examples.
    
    Formula:
        loss = -alpha * (1-p)^gamma * log(p)  for positive class
        loss = -(1-alpha) * p^gamma * log(1-p)  for negative class
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        p_pos = probs[:, 1]
        p_neg = probs[:, 0]
        
        pos_mask = (targets == 1).float()
        neg_mask = (targets == 0).float()
        
        focal_pos = -self.alpha * pos_mask * torch.pow(1 - p_pos + 1e-10, self.gamma) * torch.log(p_pos + 1e-10)
        focal_neg = -(1 - self.alpha) * neg_mask * torch.pow(p_neg + 1e-10, self.gamma) * torch.log(1 - p_neg + 1e-10)
        
        loss = focal_pos + focal_neg
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class EarlyStopping:
    """Early stopping handler."""
    def __init__(self, patience: int = 10, min_epochs: int = 15, mode: str = 'max'):
        self.patience = patience
        self.min_epochs = min_epochs
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if (self.mode == 'max' and score > self.best_score) or (self.mode == 'min' and score < self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience and epoch >= self.min_epochs


def setup_logging(log_dir: Path, log_level: str = "INFO"):
    """Set up logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__), log_file


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
    """Compute classification metrics."""
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba))
    }


def train_epoch(model, train_loader, loss_fn, optimizer, device, scaler, epoch, logger, 
                grad_clip=0.0, gradient_accumulation_steps=1):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    for batch_idx, (volumes, labels, _) in enumerate(train_loader):
        volumes = volumes.to(device)
        labels = labels.to(device)
        
        with autocast(enabled=scaler is not None):
            logits = model(volumes)
            loss = loss_fn(logits, labels)
            loss = loss / gradient_accumulation_steps
        
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
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
        
        running_loss += loss.item() * gradient_accumulation_steps
        
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, loss_fn, device, epoch, logger, temperature=1.0):
    """Validate model with optional temperature scaling."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_patient_ids = []
    
    with torch.no_grad():
        for volumes, labels, patient_ids in val_loader:
            volumes = volumes.to(device)
            labels = labels.to(device)
            
            logits = model(volumes)
            loss = loss_fn(logits, labels)
            
            running_loss += loss.item()
            
            # Apply temperature scaling to logits
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_patient_ids.extend(patient_ids)
    
    epoch_loss = running_loss / len(val_loader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    metrics['val_probs'] = np.array(all_probs)
    metrics['val_labels'] = np.array(all_labels)
    metrics['val_patient_ids'] = all_patient_ids
    metrics['val_loss'] = float(epoch_loss)
    
    return epoch_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train Swin-2 UNETR-3D model (Pilot Experiment)")
    
    # Required args
    parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3, 4],
                       help='Fold number (0-4)')
    
    # Data args
    parser.add_argument('--data-root', type=str, default='data/processed/stage_4_resize/train',
                       help='Root directory for processed data')
    parser.add_argument('--splits-dir', type=str, default='splits',
                       help='Directory containing split CSV files')
    
    # Model args
    parser.add_argument('--dropout', type=float, default=0.4,
                       help='Dropout rate in classification head (default: 0.4)')
    parser.add_argument('--feature-size', type=int, default=24,
                       help='Base feature size for Swin UNETR (default: 24 for Swin-2, reduced from 48 for memory, must be divisible by 12)')
    parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 2, 1],
                       help='Number of layers in each stage (default: [2, 2, 2, 1] for Swin-2, reduced for memory)')
    parser.add_argument('--num-heads', type=int, nargs='+', default=[3, 6, 12, 24],
                       help='Number of attention heads in each stage (default: [3, 6, 12, 24])')
    parser.add_argument('--window-size', type=int, default=4,
                       help='Window size for Swin Transformer (default: 4 for Swin-2, smaller local attention)')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=60,
                       help='Maximum number of epochs (default: 60)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate for encoder (default: 5e-5)')
    parser.add_argument('--classifier-lr', type=float, default=1e-4,
                       help='Learning rate for classifier head (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay (default: 1e-4)')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=2,
                       help='Gradient accumulation steps (default: 2)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping (0.0 = disabled, default: 1.0)')
    
    # Focal Loss args
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                       help='Focal Loss alpha parameter (default: 0.25)')
    parser.add_argument('--focal-gamma', type=float, default=1.0,
                       help='Focal Loss gamma parameter (default: 1.0, reduced from 2.0 to prevent collapse)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature scaling for logits during evaluation (default: 1.0)')
    
    # Hard example mining args
    parser.add_argument('--hard-mining', action='store_true',
                       help='Enable hard example mining (oversample Swin-1 FN cases)')
    parser.add_argument('--hard-mining-multiplier', type=int, default=2,
                       help='Hard example oversampling multiplier (default: 2x)')
    parser.add_argument('--oof-predictions-file', type=str, 
                       default='ensemble/oof_predictions/merged_oof_predictions.csv',
                       help='Path to Swin-1 OOF predictions for hard mining')
    
    # Early stopping
    parser.add_argument('--early-stopping', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--early-stopping-min-epochs', type=int, default=15,
                       help='Minimum epochs before early stopping (default: 15)')
    
    # General args
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='results/SwinUNETR-3D-Swin2',
                       help='Output directory (default: results/SwinUNETR-3D-Swin2)')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of data loader workers (default: auto)')
    parser.add_argument('--amp', action='store_true', default=True,
                       help='Use mixed precision training (default: True)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(args.output_dir) / f'fold_{args.fold}' / f'run_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints_dir = run_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_dir = run_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_dir = run_dir / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger, log_file = setup_logging(run_dir / 'logs')
    logger.info("="*80)
    logger.info("SWIN-2 UNETR-3D TRAINING (PILOT EXPERIMENT)")
    logger.info("="*80)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Fold: {args.fold}")
    logger.info(f"Swin-2 Configuration (Memory-Optimized):")
    logger.info(f"  - patch_size: 2 (MANDATORY: patch_size=1 creates ~2M tokens, OOM in 3D attention)")
    logger.info(f"  - window_size: {args.window_size} (smaller local attention, preserves research intent)")
    logger.info(f"  - feature_size: {args.feature_size} (reduced from 48 for memory efficiency, must be divisible by 12)")
    logger.info(f"  - depths: {args.depths} (reduced from [2,2,2,2] for memory efficiency)")
    logger.info(f"  - Focal Loss: alpha={args.focal_alpha}, gamma={args.focal_gamma}")
    logger.info(f"  - Hard mining: {args.hard_mining}")
    if args.hard_mining:
        logger.info(f"  - Hard mining multiplier: {args.hard_mining_multiplier}x")
    logger.info("="*80)
    logger.info("ARCHITECTURAL MEMORY FIX:")
    logger.info("  - patch_size=1 infeasible: ~2M tokens → attention softmax OOM")
    logger.info("  - window_size=4 preserves intent: local attention for subtle patterns (FN cases)")
    logger.info("  - Research goal unchanged: FN reduction + complementarity via local detail")
    logger.info("="*80)
    
    # Load datasets
    splits_dir = PROJECT_ROOT / args.splits_dir
    data_root = PROJECT_ROOT / args.data_root
    
    train_csv = splits_dir / f'fold_{args.fold}_train.csv'
    val_csv = splits_dir / f'fold_{args.fold}_val.csv'
    
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Split files not found: {train_csv}, {val_csv}")
    
    logger.info(f"Loading datasets from {data_root}")
    
    train_transforms = get_resnet3d_transforms_3d(mode='train', num_channels=4)
    val_transforms = get_resnet3d_transforms_3d(mode='val', num_channels=4)
    
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
    
    logger.info(f"Train dataset: {len(train_dataset)} patients")
    logger.info(f"Val dataset: {len(val_dataset)} patients")
    
    # Get labels and patient IDs
    train_labels = train_dataset.get_all_labels()
    train_patient_ids = [train_dataset.samples[i][2] for i in range(len(train_dataset))]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    logger.info(f"Class counts: LGG={class_counts[0]}, HGG={class_counts[1]}")
    
    # Setup data loaders
    num_workers = args.num_workers if args.num_workers is not None else min(8, os.cpu_count() or 1)
    
    # Hard example mining
    if args.hard_mining:
        logger.info("="*80)
        logger.info("HARD EXAMPLE MINING ENABLED")
        logger.info("="*80)
        
        oof_file = PROJECT_ROOT / args.oof_predictions_file
        if not oof_file.exists():
            logger.warning(f"OOF predictions file not found: {oof_file}, disabling hard mining")
            args.hard_mining = False
        else:
            logger.info(f"Loading Swin-1 OOF predictions from: {oof_file}")
            oof_df = pd.read_csv(oof_file)
            
            # Identify Swin-1 FN cases: label==1 and hgg_prob_swin < 0.5
            swin1_fn = oof_df[(oof_df['label'] == 1) & (oof_df['hgg_prob_swin'] < 0.5)].copy()
            logger.info(f"Total Swin-1 FN cases in OOF: {len(swin1_fn)}")
            
            # Filter to training split only (exclude validation fold)
            swin1_fn_train = swin1_fn[swin1_fn['patient_id'].isin(train_patient_ids)].copy()
            logger.info(f"Swin-1 FN cases in training split: {len(swin1_fn_train)}")
            
            if len(swin1_fn_train) == 0:
                logger.warning("No Swin-1 FN cases in training split, disabling hard mining")
                args.hard_mining = False
            else:
                hard_patient_ids = set(swin1_fn_train['patient_id'].tolist())
                logger.info(f"Hard example patient IDs: {sorted(hard_patient_ids)}")
                
                # Create weights: hard cases get multiplier, others get 1.0
                # CAP: Hard cases should not exceed 30% of each batch
                weights = []
                for patient_id in train_patient_ids:
                    if patient_id in hard_patient_ids:
                        weights.append(float(args.hard_mining_multiplier))
                    else:
                        weights.append(1.0)
                
                weights = np.array(weights, dtype=np.float32)
                
                # Cap hard-mining: ensure hard cases don't exceed 30% of batch
                # Calculate current hard case proportion
                hard_weight_sum = weights[weights > 1.0].sum()
                total_weight_sum = weights.sum()
                hard_proportion = hard_weight_sum / total_weight_sum if total_weight_sum > 0 else 0.0
                
                if hard_proportion > 0.30:
                    # Scale down hard case weights to cap at 30%
                    scale_factor = 0.30 / hard_proportion
                    weights[weights > 1.0] = 1.0 + (weights[weights > 1.0] - 1.0) * scale_factor
                    logger.info(f"Capped hard-mining: {hard_proportion:.2%} → {0.30:.2%} of batch")
                
                weights = weights / weights.sum() * len(weights)  # Normalize
                
                train_sampler = WeightedRandomSampler(
                    weights=weights,
                    num_samples=len(weights),
                    replacement=True
                )
                logger.info(f"Hard example mining: {len(hard_patient_ids)} cases oversampled {args.hard_mining_multiplier}x")
                logger.info("="*80)
    
    if not args.hard_mining:
        # Use normal class-balanced sampler
        train_sampler = get_weighted_sampler(train_labels, strategy='inverse_freq', seed=args.seed)
        logger.info("Using WeightedRandomSampler for class balancing")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
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
    
    # Create model (Swin-2 configuration: memory-optimized architecture)
    logger.info("Creating Swin-2 UNETR Encoder Classifier...")
    logger.info("Swin-2 Configuration (Memory-Optimized):")
    logger.info(f"  patch_size=2 (MANDATORY: patch_size=1 → ~2M tokens → OOM)")
    logger.info(f"  window_size={args.window_size} (smaller local attention for subtle patterns)")
    logger.info(f"  feature_size={args.feature_size} (reduced for memory)")
    logger.info(f"  depths={args.depths} (reduced for memory)")
    
    # ARCHITECTURAL MEMORY FIX EXPLANATION:
    # - patch_size=1 is infeasible: With img_size=(128,128,128), patch_size=1 creates
    #   (128/1)^3 = 2,097,152 tokens. 3D window attention's softmax operation explodes
    #   memory even with batch_size=1, causing OOM in the first Swin block.
    # - patch_size=2 is mandatory: Creates (128/2)^3 = 262,144 tokens (8× reduction),
    #   making attention feasible while still providing fine detail.
    # - window_size=4 preserves research intent: Smaller windows (4 vs 7) focus on
    #   local subtle patterns that Swin-1's global attention (window_size=7) might miss.
    #   This local detail focus still targets FN cases (small/diffuse tumors) while
    #   being memory-efficient. Research goal (FN reduction + complementarity) unchanged.
    
    model = SwinUNETREncoderClassifier(
        img_size=(128, 128, 128),
        in_channels=4,
        num_classes=2,
        patch_size=2,  # MANDATORY: patch_size=1 creates ~2M tokens → OOM in 3D attention
        window_size=args.window_size,  # Swin-2: window_size=4 (smaller local attention)
        feature_size=args.feature_size,  # Swin-2: 24 (reduced from 48 for memory, must be divisible by 12)
        depths=args.depths,  # Swin-2: [2,2,2,1] (reduced from [2,2,2,2] for memory)
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_checkpoint=False,
        use_hidden_layer=True,  # FIX: Use 2-layer MLP + dropout to prevent collapse
        logger=logger
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model created: {total_params/1e6:.2f}M parameters")
    
    # Loss function: Focal Loss (Swin-2)
    logger.info(f"Using Focal Loss: alpha={args.focal_alpha}, gamma={args.focal_gamma}")
    loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, reduction='mean')
    
    # Optimizer with differential learning rates
    backbone_params = list(model.get_backbone_params())
    classifier_params = list(model.get_classifier_params())
    
    optimizer = torch.optim.AdamW(
        [
            {'params': backbone_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': classifier_params, 'lr': args.classifier_lr, 'weight_decay': args.weight_decay}
        ],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7
    )
    
    # Mixed precision
    scaler = GradScaler() if args.amp else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping,
        min_epochs=args.early_stopping_min_epochs,
        mode='max'
    )
    
    # Training loop
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_auc': []
    }
    best_val_auc = 0.0
    best_epoch = 0
    
    logger.info("Starting training...")
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, loss_fn, optimizer, device, scaler, epoch, logger,
            args.grad_clip, args.gradient_accumulation_steps
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, loss_fn, device, epoch, logger, temperature=args.temperature)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        logger.info(f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}")
        logger.info(f"Val Loss: {val_loss:.6f}, Val Acc: {val_metrics['accuracy']:.4f}")
        logger.info(f"Val Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        logger.info(f"Val F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        logger.info(f"LR: {current_lr:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_metrics': val_metrics,
            'history': history,
            'config': vars(args)
        }
        
        torch.save(checkpoint, checkpoints_dir / 'last.pt')
        
        # Save best checkpoint
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            torch.save(checkpoint, checkpoints_dir / 'best.pt')
            logger.info(f"✓ Saved best checkpoint (AUC: {best_val_auc:.4f})")
        
        # Early stopping
        if early_stopping(val_metrics['auc'], epoch):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Load best model and evaluate
    logger.info(f"\nLoading best checkpoint from epoch {best_epoch} (AUC: {best_val_auc:.4f})")
    best_checkpoint = torch.load(checkpoints_dir / 'best.pt', map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Final evaluation (with temperature scaling)
    final_loss, final_metrics = validate(model, val_loader, loss_fn, device, 0, logger, temperature=args.temperature)
    
    # Save predictions CSV
    val_patient_ids = final_metrics['val_patient_ids']
    hgg_probs = final_metrics['val_probs'][:, 1]
    labels = final_metrics['val_labels']
    
    predictions_df = pd.DataFrame({
        'patient_id': val_patient_ids,
        'fold': args.fold,
        'swin2_prob': hgg_probs,
        'label': labels
    })
    
    csv_path = predictions_dir / 'swin2_predictions.csv'
    predictions_df.to_csv(csv_path, index=False)
    logger.info(f"✓ Saved predictions CSV to: {csv_path}")
    
    # Save metrics
    final_metrics_serializable = {
        'fold': args.fold,
        'best_epoch': best_epoch,
        'best_val_auc': best_val_auc,
        'final_metrics': {
            'accuracy': float(final_metrics['accuracy']),
            'precision': float(final_metrics['precision']),
            'recall': float(final_metrics['recall']),
            'f1': float(final_metrics['f1']),
            'auc': float(final_metrics['auc'])
        },
        'history': history,
        'config': vars(args)
    }
    
    with open(metrics_dir / 'metrics.json', 'w') as f:
        json.dump(final_metrics_serializable, f, indent=2)
    logger.info(f"✓ Saved metrics to: {metrics_dir / 'metrics.json'}")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Best epoch: {best_epoch}, Best AUC: {best_val_auc:.4f}")
    logger.info(f"Final metrics: Acc={final_metrics['accuracy']:.4f}, "
                f"Prec={final_metrics['precision']:.4f}, "
                f"Rec={final_metrics['recall']:.4f}, "
                f"F1={final_metrics['f1']:.4f}, "
                f"AUC={final_metrics['auc']:.4f}")
    logger.info(f"Predictions saved to: {csv_path}")


if __name__ == '__main__':
    main()

