#!/usr/bin/env python3
"""
Generate Verification Visualizations for Nested CV Threshold Audit

This script generates publication-ready visualizations comparing:
- Fold-specific thresholds (official nested CV)
- Fixed threshold 0.22 (reconstructed)

All numerical values are from the completed audit, not recomputed.

Author: Medical Imaging Pipeline
Date: 2026-02-12
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'visualizations' / 'nested_cv_verification'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Figure settings
DPI = 300
FIG_SIZE = (10, 8)
FONT_SIZE = 12
TITLE_SIZE = 14
LABEL_SIZE = 11

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Data from audit (fold-specific thresholds)
FOLD_SPECIFIC_CM = {
    'tn': 36,
    'fp': 39,
    'fn': 14,
    'tp': 196
}

FOLD_SPECIFIC_METRICS = {
    'recall': 0.9333,
    'precision': 0.8362,
    'f1': 0.8812,
    'fn_mean': 2.8,
    'fp_mean': 7.8
}

# Data from audit (fixed threshold 0.22)
FIXED_THRESHOLD_CM = {
    'tn': 1,
    'fp': 74,
    'fn': 0,
    'tp': 210
}

FIXED_THRESHOLD_METRICS = {
    'recall': 1.0000,
    'precision': 0.7395,
    'f1': 0.8502,
    'fn_mean': 0.0,
    'fp_mean': 14.8
}


def figure_1_confusion_matrix_fold_specific():
    """Generate Figure 1: Confusion Matrix (Fold-Specific Thresholds)."""
    logger.info("\nGenerating Figure 1: Confusion Matrix (Fold-Specific Thresholds)...")
    
    cm = np.array([
        [FOLD_SPECIFIC_CM['tn'], FOLD_SPECIFIC_CM['fp']],
        [FOLD_SPECIFIC_CM['fn'], FOLD_SPECIFIC_CM['tp']]
    ])
    
    fig, ax = plt.subplots(figsize=(8, 7), dpi=DPI)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['LGG', 'HGG'], yticklabels=['LGG', 'HGG'],
                cbar_kws={'label': 'Count'}, linewidths=1, linecolor='gray',
                annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Nested CV Confusion Matrix (Fold-Specific Thresholds)\n'
                 'Thresholds: 0.31, 0.35, 0.34, 0.37, 0.34', 
                 fontsize=TITLE_SIZE, fontweight='bold')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'figure_1_confusion_matrix_fold_specific.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file}")


def figure_2_confusion_matrix_fixed_threshold():
    """Generate Figure 2: Confusion Matrix (Fixed Threshold = 0.22)."""
    logger.info("\nGenerating Figure 2: Confusion Matrix (Fixed Threshold = 0.22)...")
    
    cm = np.array([
        [FIXED_THRESHOLD_CM['tn'], FIXED_THRESHOLD_CM['fp']],
        [FIXED_THRESHOLD_CM['fn'], FIXED_THRESHOLD_CM['tp']]
    ])
    
    fig, ax = plt.subplots(figsize=(8, 7), dpi=DPI)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax,
                xticklabels=['LGG', 'HGG'], yticklabels=['LGG', 'HGG'],
                cbar_kws={'label': 'Count'}, linewidths=1, linecolor='gray',
                annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Confusion Matrix (Fixed Threshold = 0.22)', 
                 fontsize=TITLE_SIZE, fontweight='bold')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'figure_2_confusion_matrix_fixed_threshold.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file}")


def figure_3_metric_comparison():
    """Generate Figure 3: Metric Comparison Bar Plot."""
    logger.info("\nGenerating Figure 3: Metric Comparison Bar Plot...")
    
    metrics = ['Recall', 'Precision', 'F1-Score']
    fold_specific_values = [
        FOLD_SPECIFIC_METRICS['recall'],
        FOLD_SPECIFIC_METRICS['precision'],
        FOLD_SPECIFIC_METRICS['f1']
    ]
    fixed_threshold_values = [
        FIXED_THRESHOLD_METRICS['recall'],
        FIXED_THRESHOLD_METRICS['precision'],
        FIXED_THRESHOLD_METRICS['f1']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    bars1 = ax.bar(x - width/2, fold_specific_values, width, 
                   label='Fold-Specific Thresholds', color='#2E86AB', 
                   alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, fixed_threshold_values, width,
                   label='Fixed Threshold (0.22)', color='#F18F01',
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Metric', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Score', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Effect of Threshold Selection on Classification Metrics', 
                 fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.4f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'figure_3_metric_comparison.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file}")


def figure_4_fn_fp_tradeoff():
    """Generate Figure 4: FN/FP Trade-off Plot."""
    logger.info("\nGenerating Figure 4: FN/FP Trade-off Plot...")
    
    error_types = ['False Negatives', 'False Positives']
    fold_specific_values = [
        FOLD_SPECIFIC_METRICS['fn_mean'],
        FOLD_SPECIFIC_METRICS['fp_mean']
    ]
    fixed_threshold_values = [
        FIXED_THRESHOLD_METRICS['fn_mean'],
        FIXED_THRESHOLD_METRICS['fp_mean']
    ]
    
    x = np.arange(len(error_types))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    bars1 = ax.bar(x - width/2, fold_specific_values, width,
                   label='Fold-Specific Thresholds', color='#2E86AB',
                   alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, fixed_threshold_values, width,
                   label='Fixed Threshold (0.22)', color='#F18F01',
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Error Type', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Mean Count per Fold', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('False Negative / False Positive Trade-off', 
                 fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(error_types, fontsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{height:.1f}', ha='center', va='bottom',
                   fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'figure_4_fn_fp_tradeoff.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file}")


def figure_5_side_by_side_confusion_matrices():
    """Generate Figure 5: Side-by-Side Confusion Matrices (Bonus)."""
    logger.info("\nGenerating Figure 5: Side-by-Side Confusion Matrices...")
    
    cm_fold_specific = np.array([
        [FOLD_SPECIFIC_CM['tn'], FOLD_SPECIFIC_CM['fp']],
        [FOLD_SPECIFIC_CM['fn'], FOLD_SPECIFIC_CM['tp']]
    ])
    
    cm_fixed = np.array([
        [FIXED_THRESHOLD_CM['tn'], FIXED_THRESHOLD_CM['fp']],
        [FIXED_THRESHOLD_CM['fn'], FIXED_THRESHOLD_CM['tp']]
    ])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=DPI)
    
    # Left: Fold-specific
    sns.heatmap(cm_fold_specific, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['LGG', 'HGG'], yticklabels=['LGG', 'HGG'],
                cbar_kws={'label': 'Count'}, linewidths=1, linecolor='gray',
                annot_kws={'size': 14, 'weight': 'bold'})
    axes[0].set_xlabel('Predicted Label', fontsize=LABEL_SIZE, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=LABEL_SIZE, fontweight='bold')
    axes[0].set_title('Fold-Specific Thresholds\n(0.31, 0.35, 0.34, 0.37, 0.34)', 
                      fontsize=TITLE_SIZE, fontweight='bold')
    
    # Right: Fixed threshold
    sns.heatmap(cm_fixed, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=['LGG', 'HGG'], yticklabels=['LGG', 'HGG'],
                cbar_kws={'label': 'Count'}, linewidths=1, linecolor='gray',
                annot_kws={'size': 14, 'weight': 'bold'})
    axes[1].set_xlabel('Predicted Label', fontsize=LABEL_SIZE, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=LABEL_SIZE, fontweight='bold')
    axes[1].set_title('Fixed Threshold = 0.22', 
                      fontsize=TITLE_SIZE, fontweight='bold')
    
    plt.suptitle('Confusion Matrix Comparison: Threshold Selection Impact', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'figure_5_side_by_side_confusion_matrices.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file}")


def main():
    """Main function to generate all verification visualizations."""
    logger.info("="*80)
    logger.info("GENERATING NESTED CV THRESHOLD VERIFICATION VISUALIZATIONS")
    logger.info("="*80)
    logger.info("Using audit results (no recomputation)")
    logger.info("="*80)
    
    # Generate all figures
    figure_1_confusion_matrix_fold_specific()
    figure_2_confusion_matrix_fixed_threshold()
    figure_3_metric_comparison()
    figure_4_fn_fp_tradeoff()
    figure_5_side_by_side_confusion_matrices()
    
    logger.info("\n" + "="*80)
    logger.info("VERIFICATION VISUALIZATIONS GENERATED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"\nAll figures saved to: {OUTPUT_DIR}")
    logger.info(f"Total figures generated: 5")
    logger.info("\nFigures:")
    logger.info("  1. Confusion Matrix (Fold-Specific Thresholds)")
    logger.info("  2. Confusion Matrix (Fixed Threshold = 0.22)")
    logger.info("  3. Metric Comparison Bar Plot")
    logger.info("  4. FN/FP Trade-off Plot")
    logger.info("  5. Side-by-Side Confusion Matrices (Bonus)")


if __name__ == "__main__":
    main()

