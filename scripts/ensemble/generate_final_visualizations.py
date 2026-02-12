"""
Generate Final Visualization Set for Ensemble Meta-Learner

This script generates a complete set of visualizations based on the FINALIZED
ensemble configuration:
- Meta-learner: Logistic Regression
- Calibration: Platt (from 2026-02-07_22-29-29_platt_seed42)
- Threshold: 0.35 (final adopted)

All plots use calibrated probabilities and the final threshold.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib
import logging
from typing import Tuple, Dict
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FINALIZED CONFIGURATION
FINAL_CALIBRATOR_PATH = Path('ensemble/results/calibration/2026-02-07_22-29-29_platt_seed42/calibrator_platt.joblib')
FINAL_THRESHOLD = 0.35
META_LEARNER_PATH = Path('ensemble/models/meta_learner_logistic_regression.joblib')
OOF_PREDICTIONS_PATH = Path('ensemble/oof_predictions/merged_oof_predictions.csv')
THRESHOLD_SWEEP_PATH = Path('ensemble/results/calibration/2026-02-07_22-29-29_platt_seed42/threshold_sweep_platt.json')
CALIBRATION_SUMMARY_PATH = Path('ensemble/results/calibration/2026-02-07_22-29-29_platt_seed42/calibration_summary.json')
META_LEARNER_METRICS_PATH = Path('ensemble/results/meta_learner_metrics.json')

# Output directory
OUTPUT_DIR = Path('ensemble/results/visualizations_final')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def verify_calibrator() -> Dict:
    """Verify and load the FINALIZED calibrator."""
    if not FINAL_CALIBRATOR_PATH.exists():
        raise FileNotFoundError(
            f"FINALIZED calibrator not found: {FINAL_CALIBRATOR_PATH}\n"
            f"Must use calibrator from: 2026-02-07_22-29-29_platt_seed42"
        )
    
    calibrator = joblib.load(FINAL_CALIBRATOR_PATH)
    
    # Verify it's the correct type
    if not isinstance(calibrator, dict) or calibrator.get('type') != 'platt':
        raise ValueError(
            f"Invalid calibrator type. Expected Platt calibrator from "
            f"2026-02-07_22-29-29_platt_seed42"
        )
    
    logger.info(f"✓ Loaded FINALIZED calibrator: {FINAL_CALIBRATOR_PATH}")
    return calibrator


def apply_calibrator(calibrator: Dict, y_proba_uncal: np.ndarray) -> np.ndarray:
    """Apply Platt calibrator to uncalibrated probabilities."""
    if calibrator['type'] == 'platt':
        y_proba_clipped = np.clip(y_proba_uncal, 1e-7, 1 - 1e-7)
        log_odds = np.log(y_proba_clipped / (1 - y_proba_clipped))
        y_proba_cal = calibrator['model'].predict_proba(log_odds.reshape(-1, 1))[:, 1]
        return y_proba_cal
    else:
        raise ValueError(f"Unsupported calibrator type: {calibrator['type']}")


def load_data_and_compute_probabilities() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load OOF data and compute calibrated probabilities."""
    logger.info("Loading OOF predictions...")
    df = pd.read_csv(OOF_PREDICTIONS_PATH)
    
    # Extract features and labels
    feature_cols = ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']
    X = df[feature_cols].values
    y = df['label'].values
    
    # Load meta-learner
    logger.info("Loading meta-learner...")
    meta_learner = joblib.load(META_LEARNER_PATH)
    
    # Compute uncalibrated probabilities
    logger.info("Computing uncalibrated probabilities...")
    y_proba_uncal = meta_learner.predict_proba(X)[:, 1]
    
    # Load and apply calibrator
    logger.info("Applying Platt calibration...")
    calibrator = verify_calibrator()
    y_proba_cal = apply_calibrator(calibrator, y_proba_uncal)
    
    logger.info(f"✓ Loaded {len(y)} samples")
    logger.info(f"  Uncalibrated prob range: [{y_proba_uncal.min():.4f}, {y_proba_uncal.max():.4f}]")
    logger.info(f"  Calibrated prob range: [{y_proba_cal.min():.4f}, {y_proba_cal.max():.4f}]")
    
    return y, y_proba_uncal, y_proba_cal, X


def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def plot_1_reliability_diagram(y_true: np.ndarray, y_proba_uncal: np.ndarray, 
                                y_proba_cal: np.ndarray, save_path: Path):
    """Plot 1: Reliability Diagram (Calibration Quality)."""
    logger.info("Generating reliability diagram...")
    
    n_bins = 10
    fraction_pos_uncal, mean_pred_uncal = calibration_curve(
        y_true, y_proba_uncal, n_bins=n_bins, strategy='uniform'
    )
    fraction_pos_cal, mean_pred_cal = calibration_curve(
        y_true, y_proba_cal, n_bins=n_bins, strategy='uniform'
    )
    
    # Compute ECE
    ece_uncal = compute_ece(y_true, y_proba_uncal, n_bins)
    ece_cal = compute_ece(y_true, y_proba_cal, n_bins)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(mean_pred_uncal, fraction_pos_uncal, 'o-', label=f'Uncalibrated (ECE={ece_uncal:.3f})',
            color='red', linewidth=2.5, markersize=8, alpha=0.8)
    ax.plot(mean_pred_cal, mean_pred_cal, 's-', label=f'Platt Calibrated (ECE={ece_cal:.3f})',
            color='blue', linewidth=2.5, markersize=8, alpha=0.8)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2, alpha=0.6)
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=13, fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontsize=13, fontweight='bold')
    ax.set_title('Reliability Diagram: Before vs After Platt Calibration', fontsize=15, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: {save_path.name}")


def plot_2_confusion_matrix(y_true: np.ndarray, y_proba_cal: np.ndarray, 
                            threshold: float, save_path: Path):
    """Plot 2: Confusion Matrix (FINAL)."""
    logger.info(f"Generating confusion matrix (threshold={threshold})...")
    
    y_pred = (y_proba_cal >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['LGG (Negative)', 'HGG (Positive)'],
                yticklabels=['LGG (Negative)', 'HGG (Positive)'],
                ax=ax, annot_kws={'size': 16, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title(f'Confusion Matrix (Threshold = {threshold:.2f})', fontsize=15, fontweight='bold')
    
    # Add annotations for FN and FP
    ax.text(0.5, -0.15, f'FN = {fn}', transform=ax.transAxes, 
            ha='center', fontsize=11, color='red', fontweight='bold')
    ax.text(1.5, -0.15, f'FP = {fp}', transform=ax.transAxes,
            ha='center', fontsize=11, color='orange', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: {save_path.name} (TN={tn}, FP={fp}, FN={fn}, TP={tp})")


def plot_3_per_class_performance(y_true: np.ndarray, y_proba_cal: np.ndarray,
                                 threshold: float, save_path: Path):
    """Plot 3: Per-Class Performance Bar Chart."""
    logger.info(f"Generating per-class performance (threshold={threshold})...")
    
    y_pred = (y_proba_cal >= threshold).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Extract metrics for each class
    lgg_metrics = {
        'Precision': report['0']['precision'],
        'Recall': report['0']['recall'],
        'F1': report['0']['f1-score']
    }
    hgg_metrics = {
        'Precision': report['1']['precision'],
        'Recall': report['1']['recall'],
        'F1': report['1']['f1-score']
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(lgg_metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, list(lgg_metrics.values()), width, label='LGG', 
                   color='lightblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, list(hgg_metrics.values()), width, label='HGG',
                   color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Metric', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title(f'Per-Class Performance (Threshold = {threshold:.2f})', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(list(lgg_metrics.keys()))
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: {save_path.name}")


def plot_4_prediction_distribution(y_true: np.ndarray, y_proba_cal: np.ndarray,
                                    threshold: float, save_path: Path):
    """Plot 4: Probability Distribution Plot."""
    logger.info(f"Generating probability distribution (threshold={threshold})...")
    
    lgg_proba = y_proba_cal[y_true == 0]
    hgg_proba = y_proba_cal[y_true == 1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(lgg_proba, bins=30, alpha=0.6, label='LGG', color='lightblue', 
            edgecolor='black', linewidth=1.2)
    ax.hist(hgg_proba, bins=30, alpha=0.6, label='HGG', color='lightcoral',
            edgecolor='black', linewidth=1.2)
    
    # Add threshold line
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2.5, 
               label=f'Threshold = {threshold:.2f}', zorder=10)
    
    ax.set_xlabel('Calibrated Ensemble Probability', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of Calibrated Ensemble Probabilities', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: {save_path.name}")


def plot_5_roc_curve(y_true: np.ndarray, y_proba_cal: np.ndarray, threshold: float, save_path: Path):
    """Plot 5: ROC Curve (Informational Only)."""
    logger.info("Generating ROC curve...")
    
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba_cal)
    roc_auc = auc(fpr, tpr)
    
    # Find point on ROC curve closest to threshold
    idx = np.argmin(np.abs(roc_thresholds - threshold))
    threshold_fpr = fpr[idx]
    threshold_tpr = tpr[idx]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, color='blue', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', alpha=0.7, label='Random Classifier')
    
    # Mark threshold point
    ax.plot(threshold_fpr, threshold_tpr, 'ro', markersize=12, 
            label=f'Threshold = {threshold:.2f}', zorder=10)
    
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curve (Calibrated Probabilities)\nNote: ROC not used for threshold selection', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: {save_path.name} (AUC={roc_auc:.3f})")


def plot_6_precision_recall_curve(y_true: np.ndarray, y_proba_cal: np.ndarray, 
                                  threshold: float, save_path: Path):
    """Plot 6: Precision-Recall Curve."""
    logger.info("Generating Precision-Recall curve...")
    
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba_cal)
    pr_auc = auc(recall, precision)
    
    # Find point on PR curve closest to threshold
    idx = np.argmin(np.abs(pr_thresholds - threshold))
    threshold_precision = precision[idx]
    threshold_recall = recall[idx]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(recall, precision, color='blue', lw=2.5, label=f'PR Curve (AUC = {pr_auc:.3f})')
    
    # Mark threshold point
    ax.plot(threshold_recall, threshold_precision, 'ro', markersize=12,
            label=f'Threshold = {threshold:.2f}', zorder=10)
    
    ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax.set_title('Precision-Recall Curve (Calibrated Probabilities)', 
                 fontsize=15, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: {save_path.name} (PR-AUC={pr_auc:.3f})")


def plot_7_feature_importance(save_path: Path):
    """Plot 7: Meta-Learner Feature Importance."""
    logger.info("Generating feature importance plot...")
    
    with open(META_LEARNER_METRICS_PATH) as f:
        metrics = json.load(f)
    
    coefficients = metrics['model_coefficients']
    feature_names = {
        'hgg_prob_resnet': 'ResNet50-3D',
        'hgg_prob_swin': 'SwinUNETR-3D',
        'hgg_prob_mil': 'DualStreamMIL-3D'
    }
    
    features = [feature_names[k] for k in coefficients.keys()]
    values = list(coefficients.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.barh(features, values, color=['#3498db', '#e74c3c', '#2ecc71'], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Logistic Regression Coefficient', fontsize=13, fontweight='bold')
    ax.set_ylabel('Base Model', fontsize=13, fontweight='bold')
    ax.set_title('Meta-Learner Feature Importance\n(Logistic Regression Coefficients)', 
                 fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 0.1 if val >= 0 else val - 0.1, i, f'{val:.3f}',
               va='center', ha='left' if val >= 0 else 'right', 
               fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: {save_path.name}")


def plot_8_fn_fp_tradeoff(save_path: Path, threshold: float):
    """Plot 8: FN-FP Trade-off Curve (CRITICAL)."""
    logger.info("Generating FN-FP trade-off curve...")
    
    with open(THRESHOLD_SWEEP_PATH) as f:
        sweep_data = json.load(f)
    
    thresholds = [entry['threshold'] for entry in sweep_data]
    fns = [entry['fn'] for entry in sweep_data]
    fps = [entry['fp'] for entry in sweep_data]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot curve
    ax.plot(fps, fns, 'b-', linewidth=2.5, alpha=0.7, label='FN-FP Trade-off')
    
    # Highlight key thresholds
    key_thresholds = {
        0.41: {'label': 'Previous Balanced (0.41)', 'color': 'orange', 'marker': 's'},
        0.36: {'label': 'Single-Run Cost-Sensitive (0.36)', 'color': 'purple', 'marker': '^'},
        0.35: {'label': 'FINAL Adopted (0.35)', 'color': 'red', 'marker': 'o'}
    }
    
    for thr, style in key_thresholds.items():
        # Find closest entry
        idx = np.argmin(np.abs(np.array(thresholds) - thr))
        fp_val = fps[idx]
        fn_val = fns[idx]
        
        ax.plot(fp_val, fn_val, style['marker'], color=style['color'], 
               markersize=14, label=style['label'], zorder=10, 
               markeredgecolor='black', markeredgewidth=2)
        ax.annotate(f'{thr:.2f}', (fp_val, fn_val), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold', color=style['color'],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('False Positives (FP)', fontsize=13, fontweight='bold')
    ax.set_ylabel('False Negatives (FN)', fontsize=13, fontweight='bold')
    ax.set_title('FN-FP Trade-off Curve (Calibrated Probabilities)\nMedical Decision Justification', 
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Lower FN is better (top of plot)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: {save_path.name}")


def create_readme(save_path: Path, threshold: float):
    """Create README.md documentation."""
    logger.info("Creating README.md...")
    
    content = f"""# Final Visualization Set

## System Configuration

**Meta-Learner**: Logistic Regression  
**Calibration**: Platt scaling (from run: 2026-02-07_22-29-29_platt_seed42)  
**Final Threshold**: {threshold:.2f} (cost-sensitive, stability-averaged)  
**Medical Priority**: Minimizing false negatives (HGG misses)

All plots use **calibrated ensemble probabilities** and reflect the FINAL system configuration.

---

## Generated Plots

### 1. `reliability_diagram_before_after.png`
Reliability diagram comparing uncalibrated vs Platt-calibrated probabilities. Shows Expected Calibration Error (ECE) for both. Demonstrates improved probability reliability after calibration.

### 2. `confusion_matrix_final_thr_0_35.png`
Confusion matrix at the final threshold ({threshold:.2f}). Clearly labels LGG (negative) and HGG (positive) classes. Annotates FN and FP counts for medical interpretation.

### 3. `per_class_performance_thr_0_35.png`
Bar chart showing Precision, Recall, and F1-score for LGG and HGG classes at threshold {threshold:.2f}. Provides class-specific performance metrics.

### 4. `prediction_distribution_thr_0_35.png`
Histogram of calibrated ensemble probabilities, separated by true class (LGG vs HGG). Vertical line marks the decision threshold ({threshold:.2f}).

### 5. `roc_curve_calibrated.png`
ROC curve using calibrated probabilities. Threshold {threshold:.2f} is marked on the curve. **Note**: ROC is NOT used for threshold selection; shown for informational purposes only.

### 6. `precision_recall_curve_calibrated.png`
Precision-Recall curve using calibrated probabilities. Threshold {threshold:.2f} is marked on the curve.

### 7. `meta_learner_feature_importance.png`
Logistic Regression coefficients showing the relative contribution of each base model (ResNet50-3D, SwinUNETR-3D, DualStreamMIL-3D) to the ensemble meta-learner.

### 8. `fn_fp_tradeoff_curve.png`
**CRITICAL**: FN-FP trade-off curve showing the relationship between false positives and false negatives across different thresholds. Highlights three key operating points:
- **0.41**: Previous balanced threshold
- **0.36**: Single-run cost-sensitive threshold
- **0.35**: FINAL adopted threshold (stability-averaged)

This plot visually justifies the medical decision to prioritize FN reduction.

---

## Medical Interpretation

**Threshold Selection Rationale**: The final threshold ({threshold:.2f}) was selected through cost-sensitive optimization with stability analysis across multiple calibration runs. This ensures:
- **Minimized False Negatives**: Critical for HGG detection (missed diagnoses can lead to delayed treatment)
- **Stable Performance**: Robust across different data splits
- **Medical Justification**: Acceptable trade-off between FN and FP, prioritizing patient safety

**All plots correspond to the FINAL system configuration and are ready for presentation/publication.**
"""
    
    with open(save_path, 'w') as f:
        f.write(content)
    
    logger.info(f"✓ Saved: {save_path.name}")


def main():
    """Main function to generate all visualizations."""
    logger.info("="*80)
    logger.info("GENERATING FINAL VISUALIZATION SET")
    logger.info("="*80)
    
    # Verify calibrator
    calibrator = verify_calibrator()
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Load data and compute probabilities
    y_true, y_proba_uncal, y_proba_cal, X = load_data_and_compute_probabilities()
    
    # Generate all plots
    logger.info("\n" + "="*80)
    logger.info("GENERATING PLOTS")
    logger.info("="*80)
    
    plot_1_reliability_diagram(y_true, y_proba_uncal, y_proba_cal,
                               OUTPUT_DIR / 'reliability_diagram_before_after.png')
    
    plot_2_confusion_matrix(y_true, y_proba_cal, FINAL_THRESHOLD,
                            OUTPUT_DIR / 'confusion_matrix_final_thr_0_35.png')
    
    plot_3_per_class_performance(y_true, y_proba_cal, FINAL_THRESHOLD,
                                 OUTPUT_DIR / 'per_class_performance_thr_0_35.png')
    
    plot_4_prediction_distribution(y_true, y_proba_cal, FINAL_THRESHOLD,
                                   OUTPUT_DIR / 'prediction_distribution_thr_0_35.png')
    
    plot_5_roc_curve(y_true, y_proba_cal, FINAL_THRESHOLD,
                    OUTPUT_DIR / 'roc_curve_calibrated.png')
    
    plot_6_precision_recall_curve(y_true, y_proba_cal, FINAL_THRESHOLD,
                                  OUTPUT_DIR / 'precision_recall_curve_calibrated.png')
    
    plot_7_feature_importance(OUTPUT_DIR / 'meta_learner_feature_importance.png')
    
    plot_8_fn_fp_tradeoff(OUTPUT_DIR / 'fn_fp_tradeoff_curve.png', FINAL_THRESHOLD)
    
    # Create README
    create_readme(OUTPUT_DIR / 'README.md', FINAL_THRESHOLD)
    
    logger.info("\n" + "="*80)
    logger.info("✓ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Total files: {len(list(OUTPUT_DIR.glob('*.png'))) + 1} (8 plots + 1 README)")


if __name__ == '__main__':
    main()

