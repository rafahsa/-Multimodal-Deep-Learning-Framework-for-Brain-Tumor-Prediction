"""
Generate Visualizations for Logistic Regression Meta-Learner

This script creates comprehensive visualizations for the trained meta-learner,
including ROC curve, Precision-Recall curve, confusion matrix, feature importance,
and performance metrics summary.

Usage:
    python scripts/ensemble/generate_visualizations.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report
)
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
MERGED_OOF_FILE = Path('ensemble/oof_predictions/merged_oof_predictions.csv')
MODEL_FILE = Path('ensemble/models/meta_learner_logistic_regression.joblib')
METRICS_FILE = Path('ensemble/results/meta_learner_metrics.json')
VIS_DIR = Path('ensemble/visualizations')

# Style settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12


def load_data_and_model():
    """Load merged OOF predictions and trained model."""
    logger.info("Loading data and model...")
    
    # Load data
    df = pd.read_csv(MERGED_OOF_FILE)
    logger.info(f"Loaded {len(df)} samples")
    
    # Load model
    model = joblib.load(MODEL_FILE)
    logger.info("Loaded trained meta-learner model")
    
    # Load metrics
    with open(METRICS_FILE, 'r') as f:
        metrics = json.load(f)
    logger.info("Loaded performance metrics")
    
    # Prepare features and target
    feature_cols = ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']
    X = df[feature_cols].values
    y = df['label'].values
    
    # Get predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of HGG (class 1)
    
    return df, model, metrics, X, y, y_pred, y_pred_proba


def plot_roc_curve(y_true, y_pred_proba, metrics, save_path: Path):
    """Plot ROC curve with AUC value."""
    logger.info("Generating ROC curve...")
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = metrics['auc_roc']
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier (AUC = 0.5000)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    plt.title('ROC Curve: Logistic Regression Meta-Learner', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved ROC curve to: {save_path}")


def plot_precision_recall_curve(y_true, y_pred_proba, metrics, save_path: Path):
    """Plot Precision-Recall curve with Average Precision."""
    logger.info("Generating Precision-Recall curve...")
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Calculate baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)  # Proportion of positive class
    
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'Precision-Recall curve (AP = {pr_auc:.4f})')
    plt.axhline(y=baseline, color='r', linestyle='--', lw=2,
                label=f'Baseline (AP = {baseline:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=14)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=14)
    plt.title('Precision-Recall Curve: Logistic Regression Meta-Learner', 
              fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved Precision-Recall curve to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, metrics, save_path: Path):
    """Plot confusion matrix with annotations."""
    logger.info("Generating confusion matrix...")
    
    cm = np.array(metrics['confusion_matrix'])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Absolute counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['LGG', 'HGG'], yticklabels=['LGG', 'HGG'],
                ax=axes[0], square=True, linewidths=2, linecolor='black')
    axes[0].set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=14, fontweight='bold')
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    
    # Normalized (percentages)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', cbar=True,
                xticklabels=['LGG', 'HGG'], yticklabels=['LGG', 'HGG'],
                ax=axes[1], square=True, linewidths=2, linecolor='black',
                vmin=0, vmax=1)
    axes[1].set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=14, fontweight='bold')
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    
    plt.suptitle('Confusion Matrix: Logistic Regression Meta-Learner',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved confusion matrix to: {save_path}")


def plot_feature_importance(metrics, save_path: Path):
    """Plot feature importance based on learned coefficients."""
    logger.info("Generating feature importance plot...")
    
    coefficients = metrics['model_coefficients']
    feature_names = list(coefficients.keys())
    coef_values = list(coefficients.values())
    abs_coef_values = [abs(v) for v in coef_values]
    
    # Sort by absolute value
    sorted_indices = np.argsort(abs_coef_values)[::-1]
    sorted_names = [feature_names[i] for i in sorted_indices]
    sorted_coefs = [coef_values[i] for i in sorted_indices]
    sorted_abs_coefs = [abs_coef_values[i] for i in sorted_indices]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Absolute coefficients (importance)
    colors = ['#2E86AB' if c > 0 else '#A23B72' for c in sorted_coefs]
    bars1 = axes[0].barh(range(len(sorted_names)), sorted_abs_coefs, color=colors)
    axes[0].set_yticks(range(len(sorted_names)))
    axes[0].set_yticklabels([name.replace('hgg_prob_', '').replace('_', ' ').title() 
                              for name in sorted_names])
    axes[0].set_xlabel('Absolute Coefficient Value', fontsize=14, fontweight='bold')
    axes[0].set_title('Feature Importance (Absolute Coefficient)', 
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, sorted_abs_coefs)):
        axes[0].text(val, i, f' {val:.4f}', va='center', fontsize=11)
    
    # Actual coefficients (with sign)
    bars2 = axes[1].barh(range(len(sorted_names)), sorted_coefs, color=colors)
    axes[1].axvline(x=0, color='black', linestyle='-', linewidth=1)
    axes[1].set_yticks(range(len(sorted_names)))
    axes[1].set_yticklabels([name.replace('hgg_prob_', '').replace('_', ' ').title() 
                              for name in sorted_names])
    axes[1].set_xlabel('Coefficient Value', fontsize=14, fontweight='bold')
    axes[1].set_title('Feature Coefficients (Signed)', 
                      fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, sorted_coefs)):
        label_x = val + (0.1 if val >= 0 else -0.1)
        axes[1].text(label_x, i, f'{val:.4f}', va='center', 
                    ha='left' if val >= 0 else 'right', fontsize=11)
    
    plt.suptitle('Feature Importance: Logistic Regression Meta-Learner',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved feature importance plot to: {save_path}")


def plot_performance_metrics_summary(metrics, save_path: Path):
    """
    Plot performance metrics summary.
    Note: Logistic Regression doesn't have epochs, so we show final metrics.
    """
    logger.info("Generating performance metrics summary...")
    
    # Extract metrics
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        metrics['auc_roc']
    ]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot
    colors = plt.cm.viridis(np.linspace(0, 1, len(metric_names)))
    bars = axes[0].bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Score', fontsize=14, fontweight='bold')
    axes[0].set_title('Performance Metrics Summary', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1.0])
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Horizontal bar plot for better readability
    y_pos = np.arange(len(metric_names))
    bars2 = axes[1].barh(y_pos, metric_values, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(metric_names)
    axes[1].set_xlabel('Score', fontsize=14, fontweight='bold')
    axes[1].set_title('Performance Metrics Summary (Horizontal)', 
                      fontsize=14, fontweight='bold')
    axes[1].set_xlim([0, 1.0])
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, metric_values)):
        axes[1].text(val, i, f' {val:.4f}', va='center', fontsize=11, fontweight='bold')
    
    plt.suptitle('Performance Metrics: Logistic Regression Meta-Learner',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved performance metrics summary to: {save_path}")


def plot_per_class_performance(metrics, save_path: Path):
    """Plot per-class performance metrics."""
    logger.info("Generating per-class performance plot...")
    
    class_report = metrics['classification_report']
    
    classes = ['LGG (Class 0)', 'HGG (Class 1)']
    precision = [class_report['0']['precision'], class_report['1']['precision']]
    recall = [class_report['0']['recall'], class_report['1']['recall']]
    f1 = [class_report['0']['f1-score'], class_report['1']['f1-score']]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2E86AB', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#A23B72', edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#F18F01', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Class', fontsize=14, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim([0, 1.0])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved per-class performance plot to: {save_path}")


def plot_prediction_distribution(y_true, y_pred_proba, save_path: Path):
    """Plot distribution of predicted probabilities by true class."""
    logger.info("Generating prediction distribution plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    axes[0].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, 
                 label='LGG (True Class 0)', color='blue', edgecolor='black')
    axes[0].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7,
                 label='HGG (True Class 1)', color='red', edgecolor='black')
    axes[0].axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    axes[0].set_xlabel('Predicted HGG Probability', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=14, fontweight='bold')
    axes[0].set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [y_pred_proba[y_true == 0], y_pred_proba[y_true == 1]]
    bp = axes[1].boxplot(data_to_plot, tick_labels=['LGG (True Class 0)', 'HGG (True Class 1)'],
                         patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    axes[1].axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    axes[1].set_ylabel('Predicted HGG Probability', fontsize=14, fontweight='bold')
    axes[1].set_title('Prediction Probability Distribution (Box Plot)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Prediction Probability Distributions by True Class',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved prediction distribution plot to: {save_path}")


def main():
    """Main function to generate all visualizations."""
    logger.info("=" * 80)
    logger.info("Generating Meta-Learner Visualizations")
    logger.info("=" * 80)
    
    # Create output directory
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {VIS_DIR}")
    
    # Load data and model
    df, model, metrics, X, y, y_pred, y_pred_proba = load_data_and_model()
    
    # Generate all visualizations
    logger.info("\nGenerating visualizations...")
    
    # 1. ROC Curve
    plot_roc_curve(y, y_pred_proba, metrics, VIS_DIR / 'roc_curve.png')
    
    # 2. Precision-Recall Curve
    plot_precision_recall_curve(y, y_pred_proba, metrics, VIS_DIR / 'precision_recall_curve.png')
    
    # 3. Confusion Matrix
    plot_confusion_matrix(y, y_pred, metrics, VIS_DIR / 'confusion_matrix.png')
    
    # 4. Feature Importance
    plot_feature_importance(metrics, VIS_DIR / 'feature_importance.png')
    
    # 5. Performance Metrics Summary
    plot_performance_metrics_summary(metrics, VIS_DIR / 'performance_metrics_summary.png')
    
    # 6. Per-Class Performance
    plot_per_class_performance(metrics, VIS_DIR / 'per_class_performance.png')
    
    # 7. Prediction Distribution
    plot_prediction_distribution(y, y_pred_proba, VIS_DIR / 'prediction_distribution.png')
    
    logger.info("\n" + "=" * 80)
    logger.info("Visualization Generation Complete")
    logger.info("=" * 80)
    logger.info(f"All visualizations saved to: {VIS_DIR}")
    logger.info("\nGenerated files:")
    for viz_file in sorted(VIS_DIR.glob('*.png')):
        logger.info(f"  - {viz_file.name}")


if __name__ == '__main__':
    main()

