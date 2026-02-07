"""
Visualize Ensemble Meta-Learner Results

This script generates visualization figures from evaluation JSON files for multiple thresholds.

Usage:
    # Auto-detect standard threshold files
    python scripts/ensemble/visualize_meta_learner_results.py
    
    # Specify multiple eval JSON files
    python scripts/ensemble/visualize_meta_learner_results.py --eval-jsons \
      ensemble/results/eval_threshold_0_50.json \
      ensemble/results/eval_threshold_0_22.json \
      ensemble/results/eval_threshold_0_19.json
    
    # Single file (backward compatible)
    python scripts/ensemble/visualize_meta_learner_results.py --eval-json ensemble/results/eval_threshold_0_22.json
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
import logging
from typing import Dict, List, Tuple, Optional
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

# Default paths
ENSEMBLE_DIR = Path('ensemble')
RESULTS_DIR = ENSEMBLE_DIR / 'results'
VISUALIZATIONS_DIR = ENSEMBLE_DIR / 'results' / 'visualizations'
MERGED_OOF_FILE = Path('ensemble/oof_predictions/merged_oof_predictions.csv')

# Feature columns
FEATURE_COLUMNS = ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']
TARGET_COLUMN = 'label'

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_eval_json(json_path: Path) -> Dict:
    """Load an evaluation JSON file."""
    if not json_path.exists():
        raise FileNotFoundError(f"Eval JSON not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded eval JSON: {json_path.name} (threshold: {data.get('threshold', 'unknown')})")
    return data


def auto_detect_eval_jsons(results_dir: Path) -> List[Path]:
    """Auto-detect standard eval JSON files."""
    standard_files = [
        results_dir / 'eval_threshold_0_50.json',
        results_dir / 'eval_threshold_0_22.json',
        results_dir / 'eval_threshold_0_19.json'
    ]
    
    found_files = [f for f in standard_files if f.exists()]
    if found_files:
        logger.info(f"Auto-detected {len(found_files)} eval JSON files")
        for f in found_files:
            logger.info(f"  - {f.name}")
    else:
        logger.warning("No standard eval JSON files found")
    
    return found_files


def load_probabilities_if_available() -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Try to load probabilities from OOF predictions if available."""
    try:
        if not MERGED_OOF_FILE.exists():
            return None
        
        df = pd.read_csv(MERGED_OOF_FILE)
        y_true = df[TARGET_COLUMN].values
        
        # Try to load model and compute probabilities
        model_file = ENSEMBLE_DIR / 'models' / 'meta_learner_logistic_regression.joblib'
        if model_file.exists():
            import joblib
            model = joblib.load(model_file)
            X = df[FEATURE_COLUMNS].values
            y_pred_proba = model.predict_proba(X)[:, 1]
            return y_true, y_pred_proba
    except Exception as e:
        logger.debug(f"Could not load probabilities: {e}")
    
    return None


def plot_confusion_matrix_from_json(eval_data: Dict, threshold: float, save_path: Path):
    """Plot confusion matrix from eval JSON data."""
    logger.info(f"Generating confusion matrix for threshold {threshold:.2f}...")
    
    cm = np.array(eval_data['confusion_matrix'])
    tn, fp, fn, tp = eval_data['tn'], eval_data['fp'], eval_data['fn'], eval_data['tp']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['LGG', 'HGG'], yticklabels=['LGG', 'HGG'],
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix: Threshold = {threshold:.2f}', 
                 fontsize=14, fontweight='bold')
    
    # Add metrics text
    accuracy = eval_data['accuracy']
    precision = eval_data['precision']
    recall = eval_data['recall']
    f1 = eval_data['f1_score']
    
    textstr = f'Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}\n'
    textstr += f'TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}'
    ax.text(0.5, -0.2, textstr, transform=ax.transAxes, ha='center', 
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to: {save_path}")


def plot_per_class_performance_from_json(eval_data: Dict, threshold: float, save_path: Path):
    """Plot per-class performance from eval JSON data."""
    logger.info(f"Generating per-class performance for threshold {threshold:.2f}...")
    
    class_report = eval_data['classification_report']
    
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
    ax.set_title(f'Per-Class Performance: Threshold = {threshold:.2f}',
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


def plot_feature_importance_from_json(eval_data: Dict, save_path: Path):
    """Plot feature importance from eval JSON data."""
    logger.info("Generating feature importance plot...")
    
    if 'feature_importance' not in eval_data:
        logger.warning("Feature importance not found in eval JSON, skipping plot")
        return
    
    feature_importance = eval_data['feature_importance']
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


def plot_roc_curve_with_thresholds(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   eval_data_list: List[Dict], save_path: Path):
    """Plot ROC curve with threshold markers."""
    logger.info("Generating ROC curve with threshold markers...")
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkblue', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', 
            label='Random classifier')
    
    # Add threshold markers
    colors = ['red', 'green', 'orange']
    labels = ['Baseline (0.50)', 'Balanced (0.22)', 'High-sensitivity (0.19)']
    for i, eval_data in enumerate(eval_data_list):
        threshold = eval_data['threshold']
        # Find point on ROC curve closest to this threshold
        # We need to compute FPR/TPR at this threshold
        y_pred_at_thresh = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_at_thresh)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            tpr_at = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr_at = fp / (fp + tn) if (fp + tn) > 0 else 0
            ax.plot(fpr_at, tpr_at, 'o', color=colors[i % len(colors)], 
                   markersize=10, label=labels[i % len(labels)])
    
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


def plot_precision_recall_curve_with_thresholds(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                                eval_data_list: List[Dict], save_path: Path):
    """Plot Precision-Recall curve with threshold markers."""
    logger.info("Generating Precision-Recall curve with threshold markers...")
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # Get baseline
    n_positive = np.sum(y_true == 1)
    n_total = len(y_true)
    baseline_precision = n_positive / n_total
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='darkred', lw=2,
            label=f'PR curve (AUC = {pr_auc:.4f})')
    ax.axhline(y=baseline_precision, color='gray', lw=1, linestyle='--',
               label=f'Baseline (P = {baseline_precision:.4f})')
    
    # Add threshold markers
    colors = ['red', 'green', 'orange']
    labels = ['Baseline (0.50)', 'Balanced (0.22)', 'High-sensitivity (0.19)']
    for i, eval_data in enumerate(eval_data_list):
        threshold = eval_data['threshold']
        y_pred_at_thresh = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_at_thresh)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            recall_at = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision_at = tp / (tp + fp) if (tp + fp) > 0 else 0
            ax.plot(recall_at, precision_at, 'o', color=colors[i % len(colors)], 
                   markersize=10, label=labels[i % len(labels)])
    
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


def plot_prediction_distribution_with_thresholds(y_pred_proba: np.ndarray, y_true: np.ndarray,
                                                 eval_data_list: List[Dict], save_path: Path):
    """Plot distribution of predicted probabilities with threshold markers."""
    logger.info("Generating prediction distribution plot with threshold markers...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate predictions by true class
    lgg_probs = y_pred_proba[y_true == 0]
    hgg_probs = y_pred_proba[y_true == 1]
    
    ax.hist(lgg_probs, bins=30, alpha=0.6, label='LGG (True)', 
            color='lightblue', edgecolor='black')
    ax.hist(hgg_probs, bins=30, alpha=0.6, label='HGG (True)', 
            color='lightcoral', edgecolor='black')
    
    # Add threshold markers
    colors = ['red', 'green', 'orange']
    labels = ['Baseline (0.50)', 'Balanced (0.22)', 'High-sensitivity (0.19)']
    for i, eval_data in enumerate(eval_data_list):
        threshold = eval_data['threshold']
        ax.axvline(x=threshold, color=colors[i % len(colors)], linestyle='--', lw=2, 
                  label=labels[i % len(labels)])
    
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


def plot_performance_summary_legacy(eval_data: Dict, save_path: Path):
    """Plot legacy performance summary (single threshold)."""
    logger.info("Generating legacy performance summary plot...")
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    metric_values = [
        eval_data['accuracy'],
        eval_data['precision'],
        eval_data['recall'],
        eval_data['f1_score'],
        eval_data.get('auc_roc', 0.0)
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
    logger.info(f"Saved legacy performance summary plot to: {save_path}")


def plot_performance_metrics_comparison(eval_data_list: List[Dict], save_path: Path):
    """Plot comparison of performance metrics across thresholds."""
    logger.info("Generating performance metrics comparison plot...")
    
    # Extract metrics for each threshold
    thresholds = [d['threshold'] for d in eval_data_list]
    precisions = [d['precision'] for d in eval_data_list]
    recalls = [d['recall'] for d in eval_data_list]
    f1_scores = [d['f1_score'] for d in eval_data_list]
    accuracies = [d['accuracy'] for d in eval_data_list]
    fns = [d['fn'] for d in eval_data_list]
    fps = [d['fp'] for d in eval_data_list]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Precision, Recall, F1, Accuracy
    ax1 = axes[0, 0]
    x = np.arange(len(thresholds))
    width = 0.2
    ax1.bar(x - 1.5*width, precisions, width, label='Precision', alpha=0.8)
    ax1.bar(x - 0.5*width, recalls, width, label='Recall', alpha=0.8)
    ax1.bar(x + 0.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
    ax1.bar(x + 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
    ax1.set_xlabel('Threshold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Performance Metrics by Threshold', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{t:.2f}' for t in thresholds])
    ax1.set_ylim([0, 1.1])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: FN and FP
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, fns, width, label='False Negatives', color='red', alpha=0.8)
    ax2.bar(x + width/2, fps, width, label='False Positives', color='orange', alpha=0.8)
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('False Negatives and False Positives by Threshold', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{t:.2f}' for t in thresholds])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Precision vs Recall
    ax3 = axes[1, 0]
    ax3.plot(thresholds, precisions, 'o-', label='Precision', linewidth=2, markersize=8)
    ax3.plot(thresholds, recalls, 's-', label='Recall', linewidth=2, markersize=8)
    ax3.set_xlabel('Threshold', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Precision vs Recall Trade-off', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary table (text)
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Threshold', 'Precision', 'Recall', 'F1', 'Accuracy', 'FN', 'FP']
    for i, d in enumerate(eval_data_list):
        row = [
            f"{d['threshold']:.2f}",
            f"{d['precision']:.4f}",
            f"{d['recall']:.4f}",
            f"{d['f1_score']:.4f}",
            f"{d['accuracy']:.4f}",
            f"{d['fn']}",
            f"{d['fp']}"
        ]
        table_data.append(row)
    
    table = ax4.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax4.set_title('Performance Summary Table', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved performance metrics comparison to: {save_path}")


def create_readme(eval_data_list: List[Dict], out_dir: Path, command_used: str):
    """Create or update README.md in visualizations directory."""
    readme_path = out_dir / 'README.md'
    
    # Sort by threshold
    sorted_data = sorted(eval_data_list, key=lambda x: x['threshold'])
    
    with open(readme_path, 'w') as f:
        f.write("# Ensemble Meta-Learner Visualizations\n\n")
        f.write("This directory contains visualization plots generated from ensemble evaluation results.\n\n")
        
        f.write("## Operating Points\n\n")
        f.write("- **Baseline**: threshold 0.50 (default threshold)\n")
        f.write("- **Balanced**: threshold 0.22 (optimal F1 score, balanced precision and recall)\n")
        f.write("- **High-sensitivity**: threshold 0.19 (precision ≥ 0.80, higher recall for HGG detection)\n\n")
        
        f.write("## Generated Plots\n\n")
        
        # Threshold-specific plots
        f.write("### Threshold-Specific Plots\n\n")
        f.write("These plots are generated for **each detected threshold**:\n\n")
        for data in sorted_data:
            threshold = data['threshold']
            threshold_str = f"{threshold:.2f}".replace('.', '_')
            f.write(f"- `confusion_matrix_thr_{threshold_str}.png`: Confusion matrix at threshold {threshold:.2f}\n")
            f.write(f"- `per_class_performance_thr_{threshold_str}.png`: Per-class metrics at threshold {threshold:.2f}\n")
        f.write("\n")
        
        # Note about legacy mode
        f.write("**Legacy Mode**: If `--legacy-main-output` flag is used, additional legacy filenames are generated for the main threshold (default: 0.22):\n")
        f.write("- `confusion_matrix.png`\n")
        f.write("- `per_class_performance.png`\n")
        f.write("- `performance_metrics_summary.png`\n\n")
        
        # Shared plots
        f.write("### Shared Plots (All Thresholds)\n\n")
        f.write("- `roc_curve.png`: ROC curve with markers for all operating points\n")
        f.write("- `precision_recall_curve.png`: Precision-Recall curve with markers for all operating points\n")
        f.write("- `prediction_distribution.png`: Distribution of predicted probabilities with threshold markers\n")
        f.write("- `feature_importance.png`: Feature importance (coefficient magnitudes)\n")
        f.write("- `performance_metrics_comparison.png`: Comparison of all metrics across thresholds\n\n")
        
        f.write("## Performance Summary\n\n")
        f.write("| Threshold | Precision | Recall | F1 | Accuracy | FN | FP |\n")
        f.write("|-----------|-----------|--------|----|----------|----|----|\n")
        for data in sorted_data:
            f.write(f"| {data['threshold']:.2f} | {data['precision']:.4f} | "
                   f"{data['recall']:.4f} | {data['f1_score']:.4f} | "
                   f"{data['accuracy']:.4f} | {data['fn']} | {data['fp']} |\n")
        f.write("\n")
        
        f.write("## How to Reproduce\n\n")
        f.write("Run the visualization script with:\n\n")
        f.write("```bash\n")
        f.write(command_used)
        f.write("\n```\n\n")
        
        f.write("## Notes\n\n")
        f.write("- All plots are generated from evaluation JSON files in `ensemble/results/`\n")
        f.write("- ROC/PR curves and prediction distribution require probability data from OOF predictions\n")
        f.write("- If probability data is unavailable, those plots will be skipped gracefully\n")
    
    logger.info(f"Created/updated README.md at: {readme_path}")


def main():
    """Main function to generate all visualizations."""
    parser = argparse.ArgumentParser(
        description='Generate visualizations from ensemble evaluation JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--eval-json',
        type=str,
        help='Path to a single eval JSON file (backward compatible mode)'
    )
    parser.add_argument(
        '--eval-jsons',
        nargs='+',
        help='Paths to multiple eval JSON files for comparison (space-separated)'
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default='ensemble/results/visualizations',
        help='Output directory for visualizations (default: ensemble/results/visualizations)'
    )
    parser.add_argument(
        '--main-threshold',
        type=float,
        default=0.22,
        help='Main threshold for threshold-specific plots (default: 0.22)'
    )
    parser.add_argument(
        '--secondary-threshold',
        type=float,
        default=0.19,
        help='Secondary threshold for threshold-specific plots (default: 0.19) [DEPRECATED: now generates for all thresholds]'
    )
    parser.add_argument(
        '--legacy-main-output',
        action='store_true',
        help='Also generate legacy filenames (confusion_matrix.png, etc.) for main threshold (default: 0.22)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Ensemble Meta-Learner Visualization Generation")
    logger.info("=" * 80)
    
    # Determine which eval JSONs to load
    eval_json_paths = []
    if args.eval_jsons:
        # Multi-threshold mode
        eval_json_paths = [Path(p) for p in args.eval_jsons]
        logger.info(f"Multi-threshold mode: loading {len(eval_json_paths)} eval JSON files")
    elif args.eval_json:
        # Single file mode (backward compatible)
        eval_json_paths = [Path(args.eval_json)]
        logger.info(f"Single file mode: loading 1 eval JSON file")
    else:
        # Auto-detect mode
        results_dir = Path('ensemble/results')
        eval_json_paths = auto_detect_eval_jsons(results_dir)
        if not eval_json_paths:
            logger.error("No eval JSON files found. Please specify --eval-json or --eval-jsons")
            return
    
    # Load all eval JSONs
    eval_data_list = []
    for json_path in eval_json_paths:
        try:
            data = load_eval_json(json_path)
            eval_data_list.append(data)
        except Exception as e:
            logger.error(f"Failed to load {json_path}: {e}")
    
    if not eval_data_list:
        logger.error("No valid eval JSON files loaded")
        return
    
    # Sort by threshold
    eval_data_list.sort(key=lambda x: x.get('threshold', 0))
    thresholds = [d.get('threshold', 0) for d in eval_data_list]
    logger.info(f"Loaded {len(eval_data_list)} eval JSON files with thresholds: {thresholds}")
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")
    
    # Try to load probabilities for ROC/PR curves
    prob_data = load_probabilities_if_available()
    has_probabilities = prob_data is not None
    
    if has_probabilities:
        y_true, y_pred_proba = prob_data
        logger.info("Loaded probability data for ROC/PR curves")
    else:
        logger.warning("Probability data not available. ROC/PR curves and prediction distribution will be skipped.")
        y_true, y_pred_proba = None, None
    
    # Generate visualizations
    logger.info("\n" + "=" * 80)
    logger.info("Generating Visualizations")
    logger.info("=" * 80)
    
    generated_plots = []
    skipped_plots = []
    
    # 1. Confusion matrices for ALL detected thresholds
    for eval_data in eval_data_list:
        threshold = eval_data.get('threshold', 0)
        threshold_str = f"{threshold:.2f}".replace('.', '_')
        plot_confusion_matrix_from_json(eval_data, threshold, 
                                       out_dir / f'confusion_matrix_thr_{threshold_str}.png')
        generated_plots.append(f'confusion_matrix_thr_{threshold_str}.png')
        
        # Legacy output for main threshold if requested
        if args.legacy_main_output and abs(threshold - args.main_threshold) < 0.01:
            plot_confusion_matrix_from_json(eval_data, threshold, 
                                           out_dir / 'confusion_matrix.png')
            generated_plots.append('confusion_matrix.png (legacy)')
    
    # 2. Per-class performance for ALL detected thresholds
    for eval_data in eval_data_list:
        threshold = eval_data.get('threshold', 0)
        threshold_str = f"{threshold:.2f}".replace('.', '_')
        plot_per_class_performance_from_json(eval_data, threshold,
                                             out_dir / f'per_class_performance_thr_{threshold_str}.png')
        generated_plots.append(f'per_class_performance_thr_{threshold_str}.png')
        
        # Legacy output for main threshold if requested
        if args.legacy_main_output and abs(threshold - args.main_threshold) < 0.01:
            plot_per_class_performance_from_json(eval_data, threshold,
                                                 out_dir / 'per_class_performance.png')
            generated_plots.append('per_class_performance.png (legacy)')
    
    # 3. Feature importance (once, from first eval JSON)
    if eval_data_list:
        if 'feature_importance' in eval_data_list[0]:
            plot_feature_importance_from_json(eval_data_list[0], 
                                             out_dir / 'feature_importance.png')
            generated_plots.append('feature_importance.png')
        else:
            logger.warning("Feature importance not found in eval JSON, skipping")
            skipped_plots.append('feature_importance.png (not in JSON)')
    
    # 4. ROC curve (if probabilities available)
    if has_probabilities:
        plot_roc_curve_with_thresholds(y_true, y_pred_proba, eval_data_list,
                                      out_dir / 'roc_curve.png')
        generated_plots.append('roc_curve.png')
    else:
        logger.info("ROC curve skipped: probabilities not available in eval JSON")
        skipped_plots.append('roc_curve.png (no probabilities)')
    
    # 5. Precision-Recall curve (if probabilities available)
    if has_probabilities:
        plot_precision_recall_curve_with_thresholds(y_true, y_pred_proba, eval_data_list,
                                                    out_dir / 'precision_recall_curve.png')
        generated_plots.append('precision_recall_curve.png')
    else:
        logger.info("Precision-Recall curve skipped: probabilities not available in eval JSON")
        skipped_plots.append('precision_recall_curve.png (no probabilities)')
    
    # 6. Prediction distribution (if probabilities available)
    if has_probabilities:
        plot_prediction_distribution_with_thresholds(y_pred_proba, y_true, eval_data_list,
                                                    out_dir / 'prediction_distribution.png')
        generated_plots.append('prediction_distribution.png')
    else:
        logger.info("Prediction distribution skipped: probabilities not available in eval JSON")
        skipped_plots.append('prediction_distribution.png (no probabilities)')
    
    # 7. Performance metrics comparison
    plot_performance_metrics_comparison(eval_data_list,
                                       out_dir / 'performance_metrics_comparison.png')
    generated_plots.append('performance_metrics_comparison.png')
    
    # 8. Legacy performance summary for main threshold if requested
    if args.legacy_main_output:
        main_eval_data = next((d for d in eval_data_list if abs(d.get('threshold', 0) - args.main_threshold) < 0.01), None)
        if main_eval_data:
            # Create a simple performance summary plot for legacy compatibility
            try:
                plot_performance_summary_legacy(main_eval_data, out_dir / 'performance_metrics_summary.png')
                generated_plots.append('performance_metrics_summary.png (legacy)')
            except Exception as e:
                logger.warning(f"Could not generate legacy performance_metrics_summary.png: {e}")
                skipped_plots.append('performance_metrics_summary.png (legacy, skipped)')
    
    # Create README
    command_used = ' '.join(['python', 'scripts/ensemble/visualize_meta_learner_results.py'] + 
                           (['--eval-jsons'] + [str(p) for p in eval_json_paths] if len(eval_json_paths) > 1 
                            else ['--eval-json', str(eval_json_paths[0])] if len(eval_json_paths) == 1 
                            else ['# auto-detect']))
    create_readme(eval_data_list, out_dir, command_used)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Visualization Generation Complete")
    logger.info("=" * 80)
    logger.info(f"All figures saved to: {out_dir}")
    logger.info(f"\nGenerated plots ({len(generated_plots)}):")
    for plot in generated_plots:
        logger.info(f"  ✓ {plot}")
    
    if skipped_plots:
        logger.info(f"\nSkipped plots ({len(skipped_plots)}):")
        for plot in skipped_plots:
            logger.info(f"  ✗ {plot}")


if __name__ == '__main__':
    main()
