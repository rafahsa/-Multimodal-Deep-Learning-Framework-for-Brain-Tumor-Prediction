#!/usr/bin/env python3
"""
Generate Visualizations for Enhanced Meta-Features Results

All visualizations based STRICTLY on nested CV outer-test results only.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from typing import Dict, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
RESULTS_DIR = Path('ensemble/results/nested_cv_meta_features')
VIS_DIR = RESULTS_DIR / 'visualizations'
VIS_DIR.mkdir(parents=True, exist_ok=True)

# Style settings
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Colors
COLOR_ENHANCED = '#2E86AB'  # Blue
COLOR_BASELINE = '#A23B72'  # Purple


def load_results() -> Tuple[Dict, Dict]:
    """Load enhanced and baseline results."""
    # Load enhanced results
    result_files = list(RESULTS_DIR.glob('meta_features_results_*.json'))
    if not result_files:
        raise FileNotFoundError("No enhanced results found!")
    
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading enhanced results: {latest_file}")
    
    with open(latest_file) as f:
        enhanced = json.load(f)
    
    # Load baseline
    baseline_file = Path('ensemble/results/nested_cv_meta_learning/nested_cv_results_20260208_233521.json')
    baseline = None
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline_data = json.load(f)
            if 'LogisticRegression' in baseline_data:
                baseline = baseline_data['LogisticRegression']
    
    return enhanced, baseline


def plot_fn_fp_tradeoff(enhanced: Dict, baseline: Dict = None):
    """Plot 1: FN-FP Trade-off (per outer fold)."""
    logger.info("Generating FN-FP Trade-off plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Enhanced results
    fold_results = enhanced['fold_results']
    fps = [r['fp'] for r in fold_results]
    fns = [r['fn'] for r in fold_results]
    
    ax.scatter(fps, fns, 
              label='Enhanced Meta-Features',
              color=COLOR_ENHANCED,
              s=150, alpha=0.7, edgecolors='black', linewidth=1.5,
              marker='o')
    
    # Mean ± std
    mean_fp = enhanced['fp_mean']
    mean_fn = enhanced['fn_mean']
    std_fp = enhanced['fp_std']
    std_fn = enhanced['fn_std']
    
    ax.errorbar(mean_fp, mean_fn,
               xerr=std_fp, yerr=std_fn,
               fmt='x', color=COLOR_ENHANCED,
               markersize=15, markeredgewidth=3, capsize=5, capthick=2,
               label='Enhanced (mean ± std)')
    
    # Baseline if available
    if baseline:
        baseline_fps = [r['fp'] for r in baseline['fold_results']]
        baseline_fns = [r['fn'] for r in baseline['fold_results']]
        
        ax.scatter(baseline_fps, baseline_fns,
                  label='Baseline (Simple Features)',
                  color=COLOR_BASELINE,
                  s=150, alpha=0.7, edgecolors='black', linewidth=1.5,
                  marker='s')
        
        ax.errorbar(baseline['fp_mean'], baseline['fn_mean'],
                   xerr=baseline['fp_std'], yerr=baseline['fn_std'],
                   fmt='x', color=COLOR_BASELINE,
                   markersize=15, markeredgewidth=3, capsize=5, capthick=2,
                   label='Baseline (mean ± std)')
    
    ax.set_xlabel('False Positives (FP)', fontsize=13, fontweight='bold')
    ax.set_ylabel('False Negatives (FN)', fontsize=13, fontweight='bold')
    ax.set_title('FN-FP Trade-off: Enhanced Meta-Features\n(Nested CV - Outer-Test Only)', 
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    fig.text(0.5, 0.02,
            'Results reflect true generalization performance under patient-level nested CV.',
            ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'fn_fp_tradeoff_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: fn_fp_tradeoff_enhanced.png")


def plot_cost_distribution(enhanced: Dict, baseline: Dict = None):
    """Plot 2: Cost Distribution Across Folds."""
    logger.info("Generating Cost Distribution plot...")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    data_for_boxplot = []
    labels = []
    
    # Enhanced
    fold_results = enhanced['fold_results']
    costs = [r['cost'] for r in fold_results]
    data_for_boxplot.append(costs)
    labels.append('Enhanced')
    
    # Baseline
    if baseline:
        baseline_costs = [r['cost'] for r in baseline['fold_results']]
        data_for_boxplot.append(baseline_costs)
        labels.append('Baseline')
    
    bp = ax.boxplot(data_for_boxplot, labels=labels, patch_artist=True,
                   widths=0.6, showmeans=True, meanline=True)
    
    # Color boxes
    colors = [COLOR_ENHANCED, COLOR_BASELINE] if baseline else [COLOR_ENHANCED]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for mean_line in bp['means']:
        mean_line.set_color('red')
        mean_line.set_linewidth(2)
        mean_line.set_linestyle('--')
    
    ax.set_ylabel('Cost (2×FN + FP)', fontsize=13, fontweight='bold')
    ax.set_title('Cost Distribution Across Outer Folds\n(Enhanced Meta-Features)', 
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.text(0.5, 0.02,
            'Results reflect true generalization performance under patient-level nested CV.',
            ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'cost_distribution_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: cost_distribution_enhanced.png")


def plot_recall_vs_precision(enhanced: Dict, baseline: Dict = None):
    """Plot 3: Recall vs Precision."""
    logger.info("Generating Recall vs Precision plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Enhanced
    fold_results = enhanced['fold_results']
    recalls = [r['recall'] for r in fold_results]
    precisions = [r['precision'] for r in fold_results]
    
    ax.scatter(recalls, precisions,
              label='Enhanced Meta-Features',
              color=COLOR_ENHANCED,
              s=150, alpha=0.7, edgecolors='black', linewidth=1.5,
              marker='o')
    
    ax.errorbar(enhanced['recall_mean'], enhanced['precision_mean'],
               xerr=enhanced['recall_std'], yerr=enhanced['precision_std'],
               fmt='x', color=COLOR_ENHANCED,
               markersize=15, markeredgewidth=3, capsize=5, capthick=2,
               label='Enhanced (mean ± std)')
    
    # Baseline
    if baseline:
        baseline_recalls = [r['recall'] for r in baseline['fold_results']]
        baseline_precisions = [r['precision'] for r in baseline['fold_results']]
        
        ax.scatter(baseline_recalls, baseline_precisions,
                  label='Baseline (Simple Features)',
                  color=COLOR_BASELINE,
                  s=150, alpha=0.7, edgecolors='black', linewidth=1.5,
                  marker='s')
        
        ax.errorbar(baseline['recall_mean'], baseline['precision_mean'],
                   xerr=baseline['recall_std'], yerr=baseline['precision_std'],
                   fmt='x', color=COLOR_BASELINE,
                   markersize=15, markeredgewidth=3, capsize=5, capthick=2,
                   label='Baseline (mean ± std)')
    
    ax.set_xlabel('Recall (Sensitivity)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax.set_title('Recall vs Precision: Enhanced Meta-Features\n(Nested CV - Outer-Test Only)', 
                 fontsize=15, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.75, 1.0])
    ax.set_ylim([0.7, 0.95])
    
    fig.text(0.5, 0.02,
            'Results reflect true generalization performance under patient-level nested CV.',
            ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'recall_vs_precision_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: recall_vs_precision_enhanced.png")


def plot_per_fold_fn(enhanced: Dict, baseline: Dict = None):
    """Plot 4: Per-Fold FN Bar Chart."""
    logger.info("Generating Per-Fold FN plot...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    fold_results = enhanced['fold_results']
    folds = [r['fold'] for r in fold_results]
    fns = [r['fn'] for r in fold_results]
    
    width = 0.35
    x = np.arange(len(folds))
    
    # Enhanced
    ax.bar(x - width/2 if baseline else x, fns,
          width, label='Enhanced Meta-Features',
          color=COLOR_ENHANCED, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Baseline
    if baseline:
        baseline_fns = [r['fn'] for r in baseline['fold_results']]
        ax.bar(x + width/2, baseline_fns,
              width, label='Baseline (Simple Features)',
              color=COLOR_BASELINE, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Highlight worst-case
    max_fn = max(fns)
    worst_fold_idx = fns.index(max_fn)
    ax.axvline(x=worst_fold_idx - width/2 if baseline else worst_fold_idx,
              color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(worst_fold_idx - width/2 if baseline else worst_fold_idx,
           max_fn + 0.3, 'Worst FN', rotation=90,
           ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
    
    ax.set_xlabel('Outer Fold', fontsize=13, fontweight='bold')
    ax.set_ylabel('False Negatives (FN)', fontsize=13, fontweight='bold')
    ax.set_title('FN per Outer Fold: Enhanced Meta-Features\n(Medical Safety Transparency)', 
                 fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    fig.text(0.5, 0.02,
            'Results reflect true generalization performance under patient-level nested CV.',
            ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'per_fold_fn_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: per_fold_fn_enhanced.png")


def plot_confusion_matrix_summary(enhanced: Dict):
    """Plot 5: Confusion Matrix Summary (aggregated)."""
    logger.info("Generating Confusion Matrix Summary plot...")
    
    # Aggregate across folds
    total_tn = sum(r['tn'] for r in enhanced['fold_results'])
    total_fp = sum(r['fp'] for r in enhanced['fold_results'])
    total_fn = sum(r['fn'] for r in enhanced['fold_results'])
    total_tp = sum(r['tp'] for r in enhanced['fold_results'])
    
    cm = np.array([[total_tn, total_fp],
                   [total_fn, total_tp]])
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['LGG (Negative)', 'HGG (Positive)'],
                yticklabels=['LGG (Negative)', 'HGG (Positive)'],
                ax=ax, annot_kws={'size': 16, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title('Confusion Matrix: Enhanced Meta-Features\n(Aggregated across 5 Outer Folds)', 
                 fontsize=15, fontweight='bold')
    
    # Add annotations
    ax.text(0.5, -0.15, f'FN = {total_fn}', transform=ax.transAxes,
            ha='center', fontsize=12, color='red', fontweight='bold')
    ax.text(1.5, -0.15, f'FP = {total_fp}', transform=ax.transAxes,
            ha='center', fontsize=12, color='orange', fontweight='bold')
    
    fig.text(0.5, 0.02,
            'Results reflect true generalization performance under patient-level nested CV.',
            ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(VIS_DIR / 'confusion_matrix_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: confusion_matrix_enhanced.png")


def main():
    """Main function."""
    logger.info("="*80)
    logger.info("GENERATING ENHANCED META-FEATURES VISUALIZATIONS")
    logger.info("="*80)
    
    enhanced, baseline = load_results()
    
    logger.info("\nGenerating visualizations...")
    plot_fn_fp_tradeoff(enhanced, baseline)
    plot_cost_distribution(enhanced, baseline)
    plot_recall_vs_precision(enhanced, baseline)
    plot_per_fold_fn(enhanced, baseline)
    plot_confusion_matrix_summary(enhanced)
    
    logger.info("\n" + "="*80)
    logger.info("✓ ALL VISUALIZATIONS GENERATED")
    logger.info("="*80)
    logger.info(f"Output directory: {VIS_DIR}")
    logger.info(f"Generated {len(list(VIS_DIR.glob('*.png')))} plots")


if __name__ == '__main__':
    main()

