#!/usr/bin/env python3
"""
Generate Final Ensemble Evaluation Visualizations (Figures 42-48)

This script generates publication-ready visualizations for the FINAL ensemble model
using verified artifacts from meta_learner_roi_mil.

Author: Medical Imaging Pipeline
Date: 2026-02-13
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'visualizations' / 'FINAL13.2.2026'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# FINAL verified artifacts
FINAL_PREDICTIONS_PATH = PROJECT_ROOT / 'ensemble' / 'results' / 'meta_learner_roi_mil' / 'predictions.csv'
FINAL_METRICS_PATH = PROJECT_ROOT / 'ensemble' / 'results' / 'meta_learner_roi_mil' / 'meta_learner_metrics.json'

# Figure settings
DPI = 300
FIG_SIZE = (10, 8)
FONT_SIZE = 12
TITLE_SIZE = 14
LABEL_SIZE = 11

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_final_artifacts():
    """Load final verified ensemble artifacts."""
    logger.info("="*80)
    logger.info("LOADING FINAL ENSEMBLE ARTIFACTS")
    logger.info("="*80)
    
    # Load predictions
    if not FINAL_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Predictions file not found: {FINAL_PREDICTIONS_PATH}")
    
    df = pd.read_csv(FINAL_PREDICTIONS_PATH)
    logger.info(f"✓ Loaded predictions: {len(df)} samples")
    logger.info(f"  Columns: {list(df.columns)}")
    
    # Load metrics
    if not FINAL_METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics file not found: {FINAL_METRICS_PATH}")
    
    with open(FINAL_METRICS_PATH, 'r') as f:
        metrics = json.load(f)
    logger.info(f"✓ Loaded metrics from: {FINAL_METRICS_PATH}")
    
    # Verify coefficients
    coefficients = metrics.get('model_coefficients', {})
    logger.info("\nFinal Coefficients (from metrics file):")
    for feat_name, coef_value in coefficients.items():
        logger.info(f"  {feat_name}: {coef_value:.6f}")
    logger.info(f"  Intercept: {metrics.get('model_intercept', 'N/A'):.6f}")
    
    return df, metrics


def figure_42_roc_curve(df, metrics):
    """Generate Figure 42: ROC Curve of the Ensemble Model."""
    logger.info("\nGenerating Figure 42: ROC Curve...")
    
    y_true = df['true_label'].values
    y_prob = df['predicted_probability'].values
    auc = metrics['auc_roc']
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    ax.plot(fpr, tpr, linewidth=2.5, label=f'ROC Curve (AUC = {auc:.4f})', color='#2E86AB')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier', alpha=0.7)
    
    ax.set_xlabel('False Positive Rate', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('ROC Curve of the Ensemble Model', fontsize=TITLE_SIZE, fontweight='bold')
    ax.legend(loc='lower right', fontsize=FONT_SIZE)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'figure_42_roc.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file}")


def figure_43_precision_recall_curve(df, metrics):
    """Generate Figure 43: Precision-Recall Curve."""
    logger.info("\nGenerating Figure 43: Precision-Recall Curve...")
    
    y_true = df['true_label'].values
    y_prob = df['predicted_probability'].values
    ap = average_precision_score(y_true, y_prob)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    ax.plot(recall, precision, linewidth=2.5, label=f'PR Curve (AP = {ap:.4f})', color='#A23B72')
    
    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=1.5, 
               label=f'Baseline (AP = {baseline:.4f})', alpha=0.7)
    
    ax.set_xlabel('Recall', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Precision-Recall Curve of the Ensemble Model', 
                fontsize=TITLE_SIZE, fontweight='bold')
    ax.legend(loc='lower left', fontsize=FONT_SIZE)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'figure_43_precision_recall.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file}")


def figure_44_confusion_matrix(metrics):
    """Generate Figure 44: Confusion Matrix (Default Decision Threshold)."""
    logger.info("\nGenerating Figure 44: Confusion Matrix...")
    
    cm = np.array(metrics['confusion_matrix'])
    threshold = metrics.get('threshold', 0.5)
    
    fig, ax = plt.subplots(figsize=(8, 7), dpi=DPI)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['LGG', 'HGG'], yticklabels=['LGG', 'HGG'],
                cbar_kws={'label': 'Count'}, linewidths=1, linecolor='gray',
                annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title(f'Confusion Matrix of the Ensemble Model\n(Threshold = {threshold})', 
                fontsize=TITLE_SIZE, fontweight='bold')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'figure_44_confusion_matrix.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file}")


def figure_45_feature_importance(metrics):
    """Generate Figure 45: Feature Importance of the Ensemble Meta-Learner."""
    logger.info("\nGenerating Figure 45: Feature Importance...")
    
    feature_importance = metrics['feature_importance']
    
    # Map feature names to model names (using CORRECT mapping)
    model_mapping = {
        'hgg_prob_resnet': 'ResNet50-3D',
        'hgg_prob_swin': 'SwinUNETR-3D',
        'mil_prob': 'DualStreamMIL-3D',
        'hgg_prob_mil': 'DualStreamMIL-3D'
    }
    
    # Extract and map
    model_names = []
    importance_values = []
    for feat_name, importance in feature_importance.items():
        model_name = model_mapping.get(feat_name, feat_name)
        model_names.append(model_name)
        importance_values.append(abs(importance))  # Use absolute value
    
    # Sort by importance
    sorted_idx = np.argsort(importance_values)[::-1]
    model_names = [model_names[i] for i in sorted_idx]
    importance_values = [importance_values[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    bars = ax.barh(range(len(model_names)), importance_values, color='#F18F01', alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=FONT_SIZE)
    ax.set_xlabel('Absolute Coefficient Magnitude', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Feature Importance of the Ensemble Meta-Learner', 
                fontsize=TITLE_SIZE, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, importance_values)):
        ax.text(val + max(importance_values) * 0.01, i, f'{val:.3f}',
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'figure_45_feature_importance.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file}")
    logger.info(f"  Coefficients used: {dict(zip(model_names, importance_values))}")


def figure_46_per_class_metrics(metrics):
    """Generate Figure 46: Per-Class Performance Comparison."""
    logger.info("\nGenerating Figure 46: Per-Class Performance...")
    
    report = metrics['classification_report']
    
    classes = ['LGG', 'HGG']
    precision = [report['0']['precision'], report['1']['precision']]
    recall = [report['0']['recall'], report['1']['recall']]
    f1 = [report['0']['f1-score'], report['1']['f1-score']]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2E86AB', 
                   alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#A23B72', 
                   alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#F18F01', 
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Class', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Score', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Per-Class Performance of the Ensemble Model', 
                fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'figure_46_per_class_metrics.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file}")


def figure_47_probability_distribution(df):
    """Generate Figure 47: Distribution of Ensemble-Predicted Probabilities."""
    logger.info("\nGenerating Figure 47: Probability Distribution...")
    
    lgg_probs = df[df['true_label'] == 0]['predicted_probability'].values
    hgg_probs = df[df['true_label'] == 1]['predicted_probability'].values
    
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    ax.hist(lgg_probs, bins=30, alpha=0.7, label='LGG (True Label)', 
            color='#2E86AB', edgecolor='black', linewidth=0.5, density=True)
    ax.hist(hgg_probs, bins=30, alpha=0.7, label='HGG (True Label)', 
            color='#A23B72', edgecolor='black', linewidth=0.5, density=True)
    
    # Add KDE overlay
    try:
        from scipy import stats
        if len(lgg_probs) > 1:
            kde_lgg = stats.gaussian_kde(lgg_probs)
            x_lgg = np.linspace(lgg_probs.min(), lgg_probs.max(), 200)
            ax.plot(x_lgg, kde_lgg(x_lgg), color='#2E86AB', linewidth=2, linestyle='--', alpha=0.8)
        
        if len(hgg_probs) > 1:
            kde_hgg = stats.gaussian_kde(hgg_probs)
            x_hgg = np.linspace(hgg_probs.min(), hgg_probs.max(), 200)
            ax.plot(x_hgg, kde_hgg(x_hgg), color='#A23B72', linewidth=2, linestyle='--', alpha=0.8)
    except ImportError:
        pass  # Skip KDE if scipy not available
    
    ax.set_xlabel('Predicted Probability (HGG)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Density', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Distribution of Ensemble-Predicted Probabilities', 
                fontsize=TITLE_SIZE, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'figure_47_probability_distribution.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file}")


def figure_48_summary_metrics(metrics):
    """Generate Figure 48: Summary Visualization of Ensemble Performance Metrics."""
    logger.info("\nGenerating Figure 48: Summary Metrics...")
    
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    
    metric_names = ['Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    metric_values = [
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        metrics['auc_roc']
    ]
    
    bars = ax.bar(metric_names, metric_values, alpha=0.8, color='#2E86AB',
                  edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Score', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_title('Summary of Ensemble Performance Metrics', 
                fontsize=TITLE_SIZE, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.02,
               f'{val:.4f}', ha='center', va='bottom', 
               fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / 'figure_48_summary_metrics.png'
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✓ Saved: {output_file}")


def main():
    """Main function to generate all figures."""
    logger.info("="*80)
    logger.info("GENERATING FINAL ENSEMBLE EVALUATION FIGURES (42-48)")
    logger.info("="*80)
    logger.info("Using FINAL verified artifacts from meta_learner_roi_mil")
    logger.info("="*80)
    
    # Load final artifacts
    df, metrics = load_final_artifacts()
    
    # Generate all figures
    logger.info("\n" + "="*80)
    logger.info("GENERATING FIGURES")
    logger.info("="*80)
    
    figure_42_roc_curve(df, metrics)
    figure_43_precision_recall_curve(df, metrics)
    figure_44_confusion_matrix(metrics)
    figure_45_feature_importance(metrics)
    figure_46_per_class_metrics(metrics)
    figure_47_probability_distribution(df)
    figure_48_summary_metrics(metrics)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("FIGURES GENERATED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"\nAll figures saved to: {OUTPUT_DIR}")
    logger.info(f"Total figures generated: 7")
    logger.info("\nFigures:")
    logger.info("  42. ROC Curve of the Ensemble Model")
    logger.info("  43. Precision-Recall Curve of the Ensemble Model")
    logger.info("  44. Confusion Matrix of the Ensemble Model")
    logger.info("  45. Feature Importance of the Ensemble Meta-Learner")
    logger.info("  46. Per-Class Performance of the Ensemble Model")
    logger.info("  47. Distribution of Ensemble-Predicted Probabilities")
    logger.info("  48. Summary of Ensemble Performance Metrics")
    
    logger.info("\n" + "="*80)
    logger.info("FINAL VERIFICATION")
    logger.info("="*80)
    logger.info(f"\nMetrics Used:")
    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    logger.info(f"  TN: {cm[0][0]}, FP: {cm[0][1]}")
    logger.info(f"  FN: {cm[1][0]}, TP: {cm[1][1]}")
    logger.info(f"\nCoefficients (Figure 45):")
    for feat_name, coef_value in metrics['feature_importance'].items():
        logger.info(f"  {feat_name}: {coef_value:.6f}")


if __name__ == "__main__":
    main()
