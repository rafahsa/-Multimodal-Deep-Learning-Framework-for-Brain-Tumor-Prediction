#!/usr/bin/env python3
"""
Post-hoc fix script for SwinUNETR-3D metrics.json files.

Fixes checkpoint_info and top-level metrics to match ResNet50-3D format.
Computes best_epoch and best_val_auc from training_history.val_auc.

Usage:
    python scripts/utils/fix_swinunetr_metrics.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np


def fix_metrics_json(metrics_path: Path) -> bool:
    """
    Fix a single metrics.json file.
    
    Args:
        metrics_path: Path to metrics.json file
        
    Returns:
        True if fixed successfully, False otherwise
    """
    print(f"\nProcessing: {metrics_path}")
    
    # Load metrics.json
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"  ✗ Error loading metrics.json: {e}")
        return False
    
    # Check if training_history exists
    if 'training_history' not in metrics:
        print(f"  ✗ Missing 'training_history' key")
        return False
    
    training_history = metrics['training_history']
    
    # Check if val_auc exists in training_history
    if 'val_auc' not in training_history:
        print(f"  ✗ Missing 'val_auc' in training_history")
        return False
    
    val_auc_list = training_history['val_auc']
    if not val_auc_list or len(val_auc_list) == 0:
        print(f"  ✗ Empty val_auc list")
        return False
    
    # Find best epoch (argmax of val_auc)
    # Note: epochs are 1-indexed in training (epoch 1, 2, ...), but lists are 0-indexed
    # So best_epoch = index + 1
    val_auc_array = np.array(val_auc_list)
    best_epoch_idx = int(np.argmax(val_auc_array))
    best_epoch = best_epoch_idx + 1  # Convert to 1-indexed epoch
    best_val_auc = float(val_auc_array[best_epoch_idx])
    
    print(f"  Found best epoch: {best_epoch} (index {best_epoch_idx}) with val_auc: {best_val_auc:.6f}")
    
    # Update checkpoint_info
    if 'checkpoint_info' not in metrics:
        print(f"  ✗ Missing 'checkpoint_info' key")
        return False
    
    checkpoint_info = metrics['checkpoint_info']
    
    # Update best_epoch and best_val_auc in checkpoint_info
    old_best_epoch = checkpoint_info.get('best_epoch', 'N/A')
    old_best_val_auc = checkpoint_info.get('best_val_auc', 'N/A')
    
    checkpoint_info['best_epoch'] = best_epoch
    checkpoint_info['best_val_auc'] = best_val_auc
    
    # Keep other fields as-is
    # checkpoint_epoch, checkpoint_val_auc, is_ema, checkpoint_path should remain unchanged
    
    print(f"  Updated checkpoint_info:")
    print(f"    best_epoch: {old_best_epoch} → {best_epoch}")
    print(f"    best_val_auc: {old_best_val_auc} → {best_val_auc:.6f}")
    
    # Update top-level metrics using best epoch
    # Get metrics from training_history at best_epoch_idx
    if 'val_acc' in training_history and len(training_history['val_acc']) > best_epoch_idx:
        metrics['accuracy'] = float(training_history['val_acc'][best_epoch_idx])
    else:
        print(f"  ⚠ Warning: Could not update accuracy (missing or incomplete val_acc)")
    
    if 'val_precision' in training_history and len(training_history['val_precision']) > best_epoch_idx:
        metrics['precision'] = float(training_history['val_precision'][best_epoch_idx])
    else:
        print(f"  ⚠ Warning: Could not update precision (missing or incomplete val_precision)")
    
    if 'val_recall' in training_history and len(training_history['val_recall']) > best_epoch_idx:
        metrics['recall'] = float(training_history['val_recall'][best_epoch_idx])
    else:
        print(f"  ⚠ Warning: Could not update recall (missing or incomplete val_recall)")
    
    if 'val_f1' in training_history and len(training_history['val_f1']) > best_epoch_idx:
        metrics['f1'] = float(training_history['val_f1'][best_epoch_idx])
    else:
        print(f"  ⚠ Warning: Could not update f1 (missing or incomplete val_f1)")
    
    # AUC is already updated from best_val_auc
    metrics['auc'] = best_val_auc
    
    print(f"  Updated top-level metrics from epoch {best_epoch}:")
    print(f"    accuracy: {metrics.get('accuracy', 'N/A')}")
    print(f"    precision: {metrics.get('precision', 'N/A')}")
    print(f"    recall: {metrics.get('recall', 'N/A')}")
    print(f"    f1: {metrics.get('f1', 'N/A')}")
    print(f"    auc: {metrics['auc']:.6f}")
    
    # Update loss_summary.val_loss.best_epoch if it exists
    if 'loss_summary' in metrics and 'val_loss' in metrics['loss_summary']:
        metrics['loss_summary']['val_loss']['best_epoch'] = best_epoch
        # Update best_value to be val_loss at best_epoch (if val_loss exists)
        if 'val_loss' in training_history and len(training_history['val_loss']) > best_epoch_idx:
            metrics['loss_summary']['val_loss']['best_value'] = float(training_history['val_loss'][best_epoch_idx])
    
    # Save updated metrics.json
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  ✓ Saved updated metrics.json")
        return True
    except Exception as e:
        print(f"  ✗ Error saving metrics.json: {e}")
        return False


def main():
    """Fix all SwinUNETR-3D metrics.json files."""
    # Find all metrics.json files in SwinUNETR-3D results
    results_dir = Path(__file__).parent.parent.parent / 'results' / 'SwinUNETR-3D'
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Find all metrics.json files
    metrics_files = list(results_dir.rglob('metrics/metrics.json'))
    
    if not metrics_files:
        print(f"No metrics.json files found in {results_dir}")
        sys.exit(1)
    
    print(f"Found {len(metrics_files)} metrics.json files to fix")
    print("=" * 80)
    
    # Fix each file
    success_count = 0
    failed_files = []
    
    for metrics_path in sorted(metrics_files):
        if fix_metrics_json(metrics_path):
            success_count += 1
        else:
            failed_files.append(metrics_path)
    
    # Summary
    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Successfully fixed: {success_count}/{len(metrics_files)}")
    
    if failed_files:
        print(f"  Failed files:")
        for f in failed_files:
            print(f"    - {f}")
        sys.exit(1)
    else:
        print(f"  ✓ All files fixed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

