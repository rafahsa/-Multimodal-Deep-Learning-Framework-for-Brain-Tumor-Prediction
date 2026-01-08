#!/usr/bin/env python3
"""
Multi-Modality Dual-Stream MIL Training Script for BraTS2018

This script trains a Multi-Modality Dual-Stream MIL model (FLAIR + T1ce)
for brain tumor classification (HGG vs LGG) with entropy-based slice selection,
LDAM loss, and DRW.

IMPORTANT: Entropy-based slice selection is ALWAYS enabled for MIL.
Each modality uses the same entropy-based slice selection (same top-k slices).

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
    roc_auc_score, confusion_matrix, classification_report, roc_curve
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

from utils.mil_dataset_multi_modal import MultiModalityMILDataset
from utils.ldam_loss import build_loss_fn
from utils.class_balancing import get_weighted_sampler
from models.dual_stream_mil.model_multi_modal import MultiModalityDualStreamMIL

# Import shared utilities from single-modality script
sys.path.insert(0, str(project_root / 'scripts' / 'training'))
from train_mil import (
    setup_logging, set_seed, EarlyStopping, 
    find_optimal_threshold, compute_metrics, save_plots
)


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Modality Dual-Stream MIL model")
    
    # Required args
    parser.add_argument('--fold', type=int, required=True, choices=[0, 1, 2, 3, 4],
                       help='Fold number (0-4)')
    
    # Data args
    parser.add_argument('--modalities', type=str, nargs='+', default=['flair', 't1ce'],
                       choices=['t1', 't1ce', 't2', 'flair'],
                       help='Modalities to use (default: flair t1ce)')
    parser.add_argument('--top-k', type=int, default=16,
                       help='Number of top-k slices for entropy selection (default: 16)')
    parser.add_argument('--data-root', type=str, default='data/processed/stage_4_resize/train',
                       help='Root directory for processed data')
    parser.add_argument('--entropy-dir', type=str, default='data/entropy',
                       help='Directory containing entropy JSON files')
    parser.add_argument('--splits-dir', type=str, default='splits',
                       help='Directory containing split CSV files')
    
    # Training args
    parser.add_argument('--epochs', type=int, default=50,
                       help='Maximum number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size (default: 2)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
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
                       help='Use torch.compile (PyTorch 2.0+)')
    
    # Other args
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default='results/MIL',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use (default: auto)')
    
    args = parser.parse_args()
    
    # Setup directories
    output_base_dir = project_root / args.output_dir
    run_dir = output_base_dir / 'runs' / f'fold_{args.fold}' / datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints_dir = run_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = run_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = run_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = run_dir / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    config_dir = run_dir / 'config'
    config_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = run_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(logs_dir, args.fold, run_dir.name)
    logger.info(f"Starting Multi-Modality MIL training for fold {args.fold}")
    logger.info(f"Modalities: {args.modalities}")
    logger.info(f"Run directory: {run_dir}")
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")
    
    # Save config
    config = vars(args).copy()
    config['device'] = str(device)
    config['torch_version'] = torch.__version__
    config['cuda_available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        config['cuda_version'] = torch.version.cuda
        config['gpu_name'] = torch.cuda.get_device_name(0)
    config['class_mapping'] = {'LGG': 0, 'HGG': 1}
    config['pos_label'] = 1  # HGG is positive class for binary metrics
    config['multi_modal'] = True
    config['entropy_settings'] = {'top_k': args.top_k, 'axis': 'axial'}
    
    with open(config_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_dir / 'config.json'}")
    
    # Set seed
    set_seed(args.seed)
    logger.info(f"Random seed set to {args.seed}")
    
    # Performance settings
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for CUDA matrix multiplications and cuDNN")
    
    # Data loading
    data_root = project_root / args.data_root
    splits_dir = project_root / args.splits_dir
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
    
    # Create multi-modality datasets
    train_dataset = MultiModalityMILDataset(
        split_csv=str(train_csv),
        data_root=str(data_root),
        modalities=args.modalities,
        top_k=args.top_k,
        entropy_dir=str(entropy_dir),
        transform=None
    )
    
    val_dataset = MultiModalityMILDataset(
        split_csv=str(val_csv),
        data_root=str(data_root),
        modalities=args.modalities,
        top_k=args.top_k,
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
    
    # Class balancing: Use WeightedRandomSampler by default
    train_sampler = None
    if args.use_balanced_sampler:
        train_sampler = get_weighted_sampler(train_labels, strategy='inverse_freq', seed=args.seed)
        logger.info("Using WeightedRandomSampler for training (class balancing enabled)")
    else:
        logger.info("WeightedRandomSampler disabled (may lead to class imbalance bias)")
    
    def collate_fn(batch):
        """Custom collate function for multi-modality bags"""
        bags_dicts, labels = zip(*batch)
        
        # Stack bags for each modality
        stacked_bags = {}
        for mod in args.modalities:
            modality_bags = [bags_dict[mod] for bags_dict in bags_dicts]
            stacked_bags[mod] = torch.stack(modality_bags, dim=0)  # (batch_size, num_slices, 1, H, W)
        
        labels = torch.stack(labels, dim=0)  # (batch_size,)
        return stacked_bags, labels
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    logger.info(f"Data loaders created (num_workers={num_workers})")
    
    # Model
    logger.info("Creating Multi-Modality Dual-Stream MIL model...")
    model = MultiModalityDualStreamMIL(
        num_classes=2,
        pretrained_encoder=True,  # Use ImageNet pretrained
        dropout=args.dropout
    ).to(device)
    
    if args.compile and hasattr(torch, 'compile'):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created: {total_params/1e6:.2f}M total params, {trainable_params/1e6:.2f}M trainable")
    
    # Loss function
    loss_fn = build_loss_fn(
        num_classes=2,
        class_counts=class_counts,
        max_m=args.max_m,
        s=args.s,
        drw_start_epoch=args.drw_start_epoch,
        device=str(device)
    )
    logger.info(f"LDAM loss configured (max_m={args.max_m}, s={args.s}, drw_start_epoch={args.drw_start_epoch})")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs//3, gamma=0.1)
    else:
        scheduler = None
    
    # Mixed precision
    scaler = GradScaler() if args.amp else None
    if args.amp:
        logger.info("Using Automatic Mixed Precision (AMP)")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        min_epochs=args.early_stopping_min_epochs,
        monitor_metric='auc',
        tie_breaker='f1'
    )
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()
    best_val_auc = -np.inf
    best_val_f1 = -np.inf
    best_epoch = -1
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [], 'val_f1': [],
        'lr': []
    }
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (bags_dict, labels) in enumerate(train_loader):
            # Move bags to device (dict of tensors)
            bags_dict_device = {mod: bags_dict[mod].to(device, non_blocking=True) for mod in args.modalities}
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if args.amp:
                with autocast():
                    logits, _ = model(bags_dict_device['flair'], bags_dict_device['t1ce'])
                    loss = loss_fn(logits, labels, epoch)
                
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(bags_dict_device['flair'], bags_dict_device['t1ce'])
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
            for bags_dict, labels in val_loader:
                bags_dict_device = {mod: bags_dict[mod].to(device, non_blocking=True) for mod in args.modalities}
                labels = labels.to(device, non_blocking=True)
                
                if args.amp:
                    with autocast():
                        logits, _ = model(bags_dict_device['flair'], bags_dict_device['t1ce'])
                        loss = loss_fn(logits, labels, epoch)
                else:
                    logits, _ = model(bags_dict_device['flair'], bags_dict_device['t1ce'])
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
        
        torch.save(checkpoint, checkpoints_dir / 'last.pt')
        if is_best:
            torch.save(checkpoint, checkpoints_dir / 'best.pt')
            logger.info(f"Saved best model (Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f})")
        
        # Early stopping
        if early_stopping(epoch, {'auc': val_auc, 'f1': val_f1}):
            logger.info(f"Early stopping triggered at epoch {epoch+1} (best epoch: {best_epoch+1})")
            break
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time/60:.2f} minutes")
    logger.info(f"Best epoch: {best_epoch+1} (Val AUC: {best_val_auc:.4f}, Val F1: {best_val_f1:.4f})")
    
    # Final evaluation
    logger.info("Loading best model for final evaluation...")
    try:
        best_checkpoint = torch.load(checkpoints_dir / 'best.pt', map_location=device, weights_only=False)
    except TypeError:
        best_checkpoint = torch.load(checkpoints_dir / 'best.pt', map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Final validation evaluation
    model.eval()
    final_val_preds = []
    final_val_labels = []
    final_val_probs = []
    
    with torch.no_grad():
        for bags_dict, labels in val_loader:
            bags_dict_device = {mod: bags_dict[mod].to(device, non_blocking=True) for mod in args.modalities}
            labels = labels.to(device, non_blocking=True)
            
            if args.amp:
                with autocast():
                    logits, _ = model(bags_dict_device['flair'], bags_dict_device['t1ce'])
            else:
                logits, _ = model(bags_dict_device['flair'], bags_dict_device['t1ce'])
            
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
    
    # Find optimal threshold
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
    
    # Save predictions
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
    latest_dir = output_base_dir / 'latest' / f'fold_{args.fold}'
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

