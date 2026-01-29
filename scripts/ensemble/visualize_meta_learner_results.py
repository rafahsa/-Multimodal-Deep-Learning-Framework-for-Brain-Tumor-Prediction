"""
Visualize Ensemble Meta-Learner Results

This script regenerates all ensemble visualization figures from stored OOF predictions
and saved model/metrics. It ensures full reproducibility of all figures used in Chapter 6.

Usage:
    python scripts/ensemble/visualize_meta_learner_results.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib
import logging
from typing import Dict, Tuple
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
MERGED_OOF_FILE = Path('ensemble/oof_predictions/merged_oof_predictions.csv')
ENSEMBLE_DIR = Path('ensemble')
MODELS_DIR = ENSEMBLE_DIR / 'models'
RESULTS_DIR = ENSEMBLE_DIR / 'results'
VISUALIZATIONS_DIR = ENSEMBLE_DIR / 'visualizations'
METRICS_FILE = RESULTS_DIR / 'meta_learner_metrics.json'
MODEL_FILE = MODELS_DIR / 'meta_learner_logistic_regression.joblib'

# Feature columns
FEATURE_COLUMNS = ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']
TARGET_COLUMN = 'label'

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_data_and_model() -> Tuple[pd.DataFrame, object, Dict]:
    """
    Load OOF predictions, trained model, and saved metrics.
    
    Returns:
        df: DataFrame with OOF predictions
        model: Trained LogisticRegression model
        metrics: Dictionary with saved metrics
    """
    logger.info("Loading data and model...")
    
    # Load OOF predictions
    if not MERGED_OOF_FILE.exists():
        raise FileNotFoundError(f"OOF predictions file not found: {MERGED_OOF_FILE}")
    df = pd.read_csv(MERGED_OOF_FILE)
    logger.info(f"Loaded {len(df)} OOF predictions")
    
    # Load model
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)
    logger.info(f"Loaded trained model from: {MODEL_FILE}")
    
    # Load metrics
    if not METRICS_FILE.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_FILE}")
    with open(METRICS_FILE, 'r') as f:
        metrics = json.load(f)
    logger.info(f"Loaded metrics from: {METRICS_FILE}")
    
    return df, model, metrics


def prepare_predictions(df: pd.DataFrame, model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate predictions from the model.
    
    Returns:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (HGG class)
    """
    X = df[FEATURE_COLUMNS].values
    y_true = df[TARGET_COLUMN].values
    
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of HGG class
    
    return y_true, y_pred, y_pred_proba


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                   metrics: Dict, save_path: Path):
    """Plot ROC curve."""
    logger.info("Generating ROC curve...")
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = metrics['auc_roc']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkblue', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', 
            label='Random classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve: Ensemble Meta-Learner', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ROC curve to: {save_path}")


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                metrics: Dict, save_path: Path):
    """Plot Precision-Recall curve."""
    logger.info("Generating Precision-Recall curve...")
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Get baseline (random classifier)
    n_positive = np.sum(y_true == 1)
    n_total = len(y_true)
    baseline_precision = n_positive / n_total
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='darkred', lw=2,
            label=f'PR curve (AUC = {pr_auc:.4f})')
    ax.axhline(y=baseline_precision, color='gray', lw=1, linestyle='--',
               label=f'Baseline (P = {baseline_precision:.4f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve: Ensemble Meta-Learner', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Precision-Recall curve to: {save_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          metrics: Dict, save_path: Path):
    """Plot confusion matrix."""
    logger.info("Generating confusion matrix...")
    
    cm = np.array(metrics['confusion_matrix'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['LGG', 'HGG'], yticklabels=['LGG', 'HGG'],
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix: Ensemble Meta-Learner', 
                 fontsize=14, fontweight='bold')
    
    # Add accuracy text
    accuracy = metrics['accuracy']
    ax.text(0.5, -0.15, f'Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)',
            transform=ax.transAxes, ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to: {save_path}")


def plot_feature_importance(metrics: Dict, save_path: Path):
    """Plot feature importance (coefficient magnitudes)."""
    logger.info("Generating feature importance plot...")
    
    feature_importance = metrics['feature_importance']
    features = list(feature_importance.keys())
    importance_values = list(feature_importance.values())
    
    # Create readable feature names
    feature_names = {
        'hgg_prob_resnet': 'ResNet50-3D',
        'hgg_prob_swin': 'SwinUNETR-3D',
        'hgg_prob_mil': 'DualStreamMIL-3D'
    }
    display_names = [feature_names.get(f, f) for f in features]
    
    # Sort by importance
    sorted_idx = np.argsort(importance_values)[::-1]
    sorted_features = [display_names[i] for i in sorted_idx]
    sorted_values = [importance_values[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(sorted_features, sorted_values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Feature Importance (|Coefficient|)', fontsize=12)
    ax.set_title('Feature Importance: Ensemble Meta-Learner', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, sorted_values)):
        ax.text(val + max(sorted_values) * 0.01, i, f'{val:.4f}',
                va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved feature importance plot to: {save_path}")


def plot_prediction_distribution(y_pred_proba: np.ndarray, y_true: np.ndarray,
                                 save_path: Path):
    """Plot distribution of predicted probabilities by class."""
    logger.info("Generating prediction distribution plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate predictions by true class
    lgg_probs = y_pred_proba[y_true == 0]
    hgg_probs = y_pred_proba[y_true == 1]
    
    ax.hist(lgg_probs, bins=30, alpha=0.6, label='LGG (True)', 
            color='lightblue', edgecolor='black')
    ax.hist(hgg_probs, bins=30, alpha=0.6, label='HGG (True)', 
            color='lightcoral', edgecolor='black')
    
    ax.axvline(x=0.5, color='red', linestyle='--', lw=2, 
               label='Decision Threshold (0.5)')
    ax.set_xlabel('Predicted Probability (HGG)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Predicted Probabilities: Ensemble Meta-Learner',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved prediction distribution plot to: {save_path}")


def plot_per_class_performance(metrics: Dict, save_path: Path):
    """Plot per-class performance metrics."""
    logger.info("Generating per-class performance plot...")
    
    class_report = metrics['classification_report']
    
    classes = ['LGG', 'HGG']
    precision = [class_report['0']['precision'], class_report['1']['precision']]
    recall = [class_report['0']['recall'], class_report['1']['recall']]
    f1 = [class_report['0']['f1-score'], class_report['1']['f1-score']]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', 
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', 
                   color='darkgreen', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', 
                   color='darkorange', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_title('Per-Class Performance Metrics: Ensemble Meta-Learner',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved per-class performance plot to: {save_path}")


def plot_performance_summary(metrics: Dict, save_path: Path):
    """Plot overall performance metrics summary."""
    logger.info("Generating performance summary plot...")
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    metric_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        metrics['auc_roc']
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(metric_names, metric_values, color='steelblue', alpha=0.7)
    ax.set_xlim([0, 1.0])
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Performance Metrics Summary: Ensemble Meta-Learner',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{val:.4f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved performance summary plot to: {save_path}")


def main():
    """Main function to generate all visualizations."""
    logger.info("=" * 80)
    logger.info("Ensemble Meta-Learner Visualization Generation")
    logger.info("=" * 80)
    
    try:
        # Create visualizations directory
        VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {VISUALIZATIONS_DIR}")
        
        # Load data and model
        df, model, metrics = load_data_and_model()
        
        # Generate predictions
        y_true, y_pred, y_pred_proba = prepare_predictions(df, model)
        logger.info(f"Generated predictions for {len(y_true)} samples")
        
        # Generate all visualizations
        logger.info("\n" + "=" * 80)
        logger.info("Generating Visualizations")
        logger.info("=" * 80)
        
        plot_roc_curve(y_true, y_pred_proba, metrics,
                      VISUALIZATIONS_DIR / 'roc_curve.png')
        
        plot_precision_recall_curve(y_true, y_pred_proba, metrics,
                                   VISUALIZATIONS_DIR / 'precision_recall_curve.png')
        
        plot_confusion_matrix(y_true, y_pred, metrics,
                             VISUALIZATIONS_DIR / 'confusion_matrix.png')
        
        plot_feature_importance(metrics,
                               VISUALIZATIONS_DIR / 'feature_importance.png')
        
        plot_prediction_distribution(y_pred_proba, y_true,
                                    VISUALIZATIONS_DIR / 'prediction_distribution.png')
        
        plot_per_class_performance(metrics,
                                 VISUALIZATIONS_DIR / 'per_class_performance.png')
        
        plot_performance_summary(metrics,
                               VISUALIZATIONS_DIR / 'performance_metrics_summary.png')
        
        logger.info("\n" + "=" * 80)
        logger.info("Visualization Generation Complete")
        logger.info("=" * 80)
        logger.info(f"All figures saved to: {VISUALIZATIONS_DIR}")
        logger.info("\nGenerated figures:")
        logger.info("  - roc_curve.png")
        logger.info("  - precision_recall_curve.png")
        logger.info("  - confusion_matrix.png")
        logger.info("  - feature_importance.png")
        logger.info("  - prediction_distribution.png")
        logger.info("  - per_class_performance.png")
        logger.info("  - performance_metrics_summary.png")
        
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

