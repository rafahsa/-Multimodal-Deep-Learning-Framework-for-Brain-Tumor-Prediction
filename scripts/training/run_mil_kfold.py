#!/usr/bin/env python3
"""
K-Fold Cross-Validation Orchestrator for Dual-Stream MIL Training

This script runs MIL training across all k-folds sequentially and aggregates results.

Usage:
    python scripts/training/run_mil_kfold.py --folds 0,1,2,3,4 --epochs 30
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd


def parse_folds(folds_str: str) -> List[int]:
    """Parse folds string into list of integers."""
    return [int(f.strip()) for f in folds_str.split(',')]


def run_fold(fold: int, args: argparse.Namespace, project_root: Path):
    """Run training for a single fold."""
    print(f"\n{'='*70}")
    print(f"Training Fold {fold}")
    print(f"{'='*70}\n")
    
    # Build command
    cmd = [
        sys.executable,
        str(project_root / 'scripts' / 'training' / 'train_mil.py'),
        '--fold', str(fold),
        '--modality', args.modality,
        '--top-k', str(args.top_k),
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.lr),
        '--drw-start-epoch', str(args.drw_start_epoch),
        '--max-m', str(args.max_m),
        '--s', str(args.s),
        '--seed', str(args.seed),
        '--early-stopping-patience', str(args.early_stopping_patience),
        '--early-stopping-min-epochs', str(args.early_stopping_min_epochs),
        '--output-dir', args.output_dir,
    ]
    
    if args.amp:
        cmd.append('--amp')
    if args.tf32:
        cmd.append('--tf32')
    if args.compile:
        cmd.append('--compile')
    if args.use_balanced_sampler:
        cmd.append('--use-balanced-sampler')
    if args.grad_clip > 0:
        cmd.extend(['--grad-clip', str(args.grad_clip)])
    if args.num_workers is not None:
        cmd.extend(['--num-workers', str(args.num_workers)])
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True, cwd=str(project_root))
        print(f"\nFold {fold} training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Fold {fold} training failed with exit code {e.returncode}")
        return False


def load_fold_metrics(output_dir: Path, fold: int) -> Dict:
    """Load metrics for a fold from the latest run."""
    latest_dir = output_dir / 'latest' / f'fold_{fold}'
    latest_file = latest_dir / 'LATEST_RUN.txt'
    
    if not latest_file.exists():
        raise FileNotFoundError(f"Latest run file not found for fold {fold}: {latest_file}")
    
    # Read latest run directory
    with open(latest_file, 'r') as f:
        run_dir = Path(f.read().strip())
    
    metrics_file = run_dir / 'metrics' / 'metrics.json'
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    return metrics, run_dir


def aggregate_results(output_dir: Path, folds: List[int]):
    """Aggregate results across folds and create summary."""
    print(f"\n{'='*70}")
    print("Aggregating Results Across Folds")
    print(f"{'='*70}\n")
    
    summary_dir = output_dir / 'kfold_summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    fold_metrics_list = []
    
    for fold in folds:
        try:
            metrics, run_dir = load_fold_metrics(output_dir, fold)
            metrics['fold'] = fold
            metrics['run_dir'] = str(run_dir)
            all_metrics.append(metrics)
            fold_metrics_list.append({
                'fold': fold,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'auc': metrics['auc'],
            })
            print(f"Fold {fold}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")
        except Exception as e:
            print(f"ERROR loading metrics for fold {fold}: {e}")
            return
    
    if len(all_metrics) == 0:
        print("ERROR: No metrics loaded")
        return
    
    # Create DataFrame
    df = pd.DataFrame(fold_metrics_list)
    
    # Compute mean and std
    metrics_to_aggregate = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    mean_std = {}
    
    for metric in metrics_to_aggregate:
        values = df[metric].values
        mean_std[f'{metric}_mean'] = float(np.mean(values))
        mean_std[f'{metric}_std'] = float(np.std(values))
    
    # Save summary JSON (detailed)
    summary_json = {
        'timestamp': datetime.now().isoformat(),
        'folds': folds,
        'per_fold_metrics': all_metrics,
        'aggregated': mean_std
    }
    
    with open(summary_dir / 'summary.json', 'w') as f:
        json.dump(summary_json, f, indent=2)
    
    # Save summary CSV (table)
    df.to_csv(summary_dir / 'summary.csv', index=False)
    
    # Save mean±std JSON
    with open(summary_dir / 'mean_std.json', 'w') as f:
        json.dump(mean_std, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*70}")
    print("Aggregated Results (Mean ± Std)")
    print(f"{'='*70}")
    print(f"Accuracy:  {mean_std['accuracy_mean']:.4f} ± {mean_std['accuracy_std']:.4f}")
    print(f"Precision: {mean_std['precision_mean']:.4f} ± {mean_std['precision_std']:.4f}")
    print(f"Recall:    {mean_std['recall_mean']:.4f} ± {mean_std['recall_std']:.4f}")
    print(f"F1 Score:  {mean_std['f1_mean']:.4f} ± {mean_std['f1_std']:.4f}")
    print(f"AUC-ROC:   {mean_std['auc_mean']:.4f} ± {mean_std['auc_std']:.4f}")
    print(f"{'='*70}")
    
    print(f"\nSummary files saved to: {summary_dir}")
    print(f"  - summary.json (detailed per-fold metrics)")
    print(f"  - summary.csv (table format)")
    print(f"  - mean_std.json (aggregated statistics)")


def main():
    parser = argparse.ArgumentParser(description="Run MIL training across k-folds")
    
    # K-fold args
    parser.add_argument('--folds', type=str, default='0,1,2,3,4',
                       help='Comma-separated list of folds (default: 0,1,2,3,4)')
    
    # Pass-through args
    parser.add_argument('--modality', type=str, default='flair', choices=['t1', 't1ce', 't2', 'flair'],
                       help='Modality (default: flair)')
    parser.add_argument('--top-k', type=int, default=16,
                       help='Top-k slices (default: 16)')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Max epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--drw-start-epoch', type=int, default=15,
                       help='DRW start epoch (default: 15)')
    parser.add_argument('--max-m', type=float, default=0.5,
                       help='LDAM max margin (default: 0.5)')
    parser.add_argument('--s', type=float, default=30,
                       help='LDAM scaling factor (default: 30)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--early-stopping-patience', type=int, default=7,
                       help='Early stopping patience (default: 7)')
    parser.add_argument('--early-stopping-min-epochs', type=int, default=10,
                       help='Early stopping min epochs (default: 10)')
    parser.add_argument('--amp', action='store_true', default=True,
                       help='Use AMP (default: True)')
    parser.add_argument('--no-amp', dest='amp', action='store_false',
                       help='Disable AMP')
    parser.add_argument('--tf32', action='store_true', default=True,
                       help='Use TF32 (default: True)')
    parser.add_argument('--no-tf32', dest='tf32', action='store_false',
                       help='Disable TF32')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile')
    parser.add_argument('--use-balanced-sampler', action='store_true',
                       help='Use balanced sampler')
    parser.add_argument('--grad-clip', type=float, default=0.0,
                       help='Gradient clipping (default: 0.0)')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of workers (default: auto)')
    parser.add_argument('--output-dir', type=str, default='results/MIL',
                       help='Output directory (default: results/MIL)')
    
    args = parser.parse_args()
    
    # Parse folds
    folds = parse_folds(args.folds)
    print(f"Running training for folds: {folds}")
    print(f"Output directory: {args.output_dir}")
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    # Run each fold sequentially
    failed_folds = []
    for fold in folds:
        success = run_fold(fold, args, project_root)
        if not success:
            failed_folds.append(fold)
    
    if failed_folds:
        print(f"\nERROR: Failed folds: {failed_folds}")
        sys.exit(1)
    
    # Aggregate results
    output_dir = project_root / args.output_dir
    aggregate_results(output_dir, folds)
    
    print("\nK-fold cross-validation completed successfully!")


if __name__ == '__main__':
    main()

