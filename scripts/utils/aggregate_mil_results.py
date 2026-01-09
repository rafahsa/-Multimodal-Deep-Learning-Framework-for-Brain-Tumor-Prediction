#!/usr/bin/env python3
"""Aggregate Dual-Stream MIL cross-validation results."""

import json
import pandas as pd
from pathlib import Path
import numpy as np
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

results_dir = Path('results/DualStreamMIL-3D/runs')

fold_results = []

for fold in range(5):
    fold_dir = results_dir / f'fold_{fold}'
    if not fold_dir.exists():
        print(f"Warning: Fold {fold} directory not found: {fold_dir}")
        continue
    
    # Find latest run
    runs = sorted(fold_dir.glob('run_*'))
    if not runs:
        print(f"Warning: No runs found for fold {fold}")
        continue
    
    latest_run = runs[-1]
    metrics_file = latest_run / 'metrics' / 'metrics.json'
    
    if not metrics_file.exists():
        print(f"Warning: Metrics file not found: {metrics_file}")
        continue
    
    try:
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        # Get confusion matrix for class balance analysis
        cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
        lgg_correct = cm[0][0] if len(cm) > 0 and len(cm[0]) > 0 else 0
        lgg_total = sum(cm[0]) if len(cm) > 0 else 0
        hgg_correct = cm[1][1] if len(cm) > 1 and len(cm[1]) > 1 else 0
        hgg_total = sum(cm[1]) if len(cm) > 1 else 0
        
        fold_results.append({
            'fold': fold,
            'best_epoch': metrics.get('checkpoint_info', {}).get('best_epoch', -1),
            'best_val_auc': metrics.get('checkpoint_info', {}).get('best_val_auc', 0),
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1': metrics.get('f1', 0),
            'auc': metrics.get('auc', 0),
            'lgg_correct': lgg_correct,
            'lgg_total': lgg_total,
            'lgg_recall': lgg_correct / lgg_total if lgg_total > 0 else 0,
            'hgg_correct': hgg_correct,
            'hgg_total': hgg_total,
            'hgg_recall': hgg_correct / hgg_total if hgg_total > 0 else 0,
            'is_ema': metrics.get('checkpoint_info', {}).get('is_ema', False),
            'run_dir': str(latest_run)
        })
    except Exception as e:
        print(f"Error processing fold {fold}: {e}")
        continue

if len(fold_results) == 0:
    print("No results found.")
    sys.exit(1)

df = pd.DataFrame(fold_results)

print("\n" + "="*100)
print("Dual-Stream MIL Cross-Validation Results")
print("="*100)
print(df.to_string(index=False))
print("\n" + "="*100)
print("Summary Statistics:")
print("="*100)
print(f"AUC:  {df['auc'].mean():.4f} ± {df['auc'].std():.4f}")
print(f"F1:   {df['f1'].mean():.4f} ± {df['f1'].std():.4f}")
print(f"Acc:  {df['accuracy'].mean():.4f} ± {df['accuracy'].std():.4f}")
print(f"LGG Recall: {df['lgg_recall'].mean():.4f} ± {df['lgg_recall'].std():.4f}")
print(f"HGG Recall: {df['hgg_recall'].mean():.4f} ± {df['hgg_recall'].std():.4f}")
print("="*100)

# Class balance analysis
print("\nClass Balance Analysis:")
print("="*100)
print(f"LGG: {df['lgg_recall'].mean():.2%} ± {df['lgg_recall'].std():.2%} recall")
print(f"HGG: {df['hgg_recall'].mean():.2%} ± {df['hgg_recall'].std():.2%} recall")
imbalance = abs(df['lgg_recall'].mean() - df['hgg_recall'].mean())
if imbalance > 0.2:
    print(f"⚠️  WARNING: Significant class imbalance detected ({imbalance:.2%} difference)")
else:
    print(f"✓ Class balance acceptable ({imbalance:.2%} difference)")
print("="*100)

# Save to CSV
output_csv = results_dir / 'cv_summary.csv'
df.to_csv(output_csv, index=False)
print(f"\nResults saved to: {output_csv}")

