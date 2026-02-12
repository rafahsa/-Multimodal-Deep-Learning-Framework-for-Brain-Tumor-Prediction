#!/usr/bin/env python3
"""
Post-Hoc Thresholding for Swin-1: Uncertainty-Aware Decision Policies

This script implements multiple decision policies to improve Swin-1 performance
using ONLY post-hoc thresholding, WITHOUT retraining Swin-1.

Policies:
A. Baseline (threshold=0.5)
B. Reject-band policy
C. Confidence-aware thresholding
D. Fold-specific calibrated threshold

All evaluation is strict 5-fold OOF (no leakage).
"""

import sys
from pathlib import Path

# Set project root explicitly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
from scipy.stats import entropy
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OOF_FILE = PROJECT_ROOT / 'ensemble' / 'oof_predictions' / 'merged_oof_predictions.csv'
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'results' / 'posthoc_thresholding'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target constraints
TARGET_FN_MAX = 10
TARGET_FP_MAX = 10
TARGET_PRECISION_MIN = 0.90
TARGET_RECALL_MIN = 0.90


def compute_entropy(prob: float) -> float:
    """Compute binary entropy: -p*log(p) - (1-p)*log(1-p)"""
    if prob <= 0 or prob >= 1:
        return 0.0
    return -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)


def policy_baseline(y_proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Baseline policy: simple threshold."""
    return (y_proba >= threshold).astype(int)


def policy_reject_band(y_proba: np.ndarray, lower: float = 0.35, upper: float = 0.65) -> np.ndarray:
    """
    Reject-band policy:
    - if prob in [lower, upper] → predict HGG (1)
    - else → threshold at 0.5
    """
    y_pred = np.zeros_like(y_proba, dtype=int)
    # Reject band: predict HGG
    y_pred[(y_proba >= lower) & (y_proba <= upper)] = 1
    # Outside reject band: use 0.5 threshold
    y_pred[y_proba > upper] = 1
    y_pred[y_proba < lower] = 0
    return y_pred


def policy_confidence_aware(
    y_proba: np.ndarray,
    y_true_train: np.ndarray,
    y_proba_train: np.ndarray,
    uncertainty_percentile: float = 75.0
) -> np.ndarray:
    """
    Confidence-aware thresholding:
    - Compute entropy for all train predictions
    - Define high-uncertainty region via percentile
    - In high-uncertainty: use aggressive HGG decision (lower threshold)
    - Outside: use conservative threshold (higher threshold)
    """
    # Compute entropy for train set
    train_entropies = np.array([compute_entropy(p) for p in y_proba_train])
    entropy_threshold = np.percentile(train_entropies, uncertainty_percentile)
    
    # Compute entropy for predictions
    pred_entropies = np.array([compute_entropy(p) for p in y_proba])
    
    # High uncertainty: lower threshold (more aggressive HGG)
    # Low uncertainty: higher threshold (more conservative)
    high_uncertainty_mask = pred_entropies >= entropy_threshold
    
    y_pred = np.zeros_like(y_proba, dtype=int)
    # High uncertainty: use lower threshold (0.3)
    y_pred[high_uncertainty_mask & (y_proba >= 0.3)] = 1
    # Low uncertainty: use higher threshold (0.6)
    y_pred[~high_uncertainty_mask & (y_proba >= 0.6)] = 1
    
    return y_pred


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_fn_max: int = TARGET_FN_MAX,
    target_fp_max: int = TARGET_FP_MAX,
    target_precision_min: float = TARGET_PRECISION_MIN,
    target_recall_min: float = TARGET_RECALL_MIN
) -> Optional[float]:
    """
    Find optimal threshold that satisfies ALL constraints.
    Returns None if no threshold satisfies all constraints.
    """
    thresholds = np.arange(0.01, 0.99, 0.01)
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        # Check ALL constraints
        if (fn <= target_fn_max and
            fp <= target_fp_max and
            precision >= target_precision_min and
            recall >= target_recall_min):
            return threshold
    
    return None


def policy_fold_specific_calibrated(
    df: pd.DataFrame,
    fold: int
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Fold-specific calibrated threshold:
    - Use TRAIN folds only to find optimal threshold
    - Apply to validation fold
    """
    # Get train folds (all except current fold)
    train_mask = df['fold'] != fold
    val_mask = df['fold'] == fold
    
    y_true_train = df.loc[train_mask, 'label'].values
    y_proba_train = df.loc[train_mask, 'hgg_prob_swin'].values
    
    # Find optimal threshold on train folds
    optimal_threshold = find_optimal_threshold(
        y_true_train, y_proba_train,
        target_fn_max=TARGET_FN_MAX,
        target_fp_max=TARGET_FP_MAX,
        target_precision_min=TARGET_PRECISION_MIN,
        target_recall_min=TARGET_RECALL_MIN
    )
    
    # Apply to validation fold
    y_proba_val = df.loc[val_mask, 'hgg_prob_swin'].values
    
    if optimal_threshold is not None:
        y_pred_val = (y_proba_val >= optimal_threshold).astype(int)
    else:
        # Fallback to 0.5 if no threshold satisfies constraints
        y_pred_val = (y_proba_val >= 0.5).astype(int)
        optimal_threshold = 0.5
    
    return y_pred_val, optimal_threshold


def evaluate_policy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    policy_name: str
) -> Dict:
    """Evaluate a decision policy."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)
    
    # Check if meets ALL target constraints
    meets_constraints = (
        fn <= TARGET_FN_MAX and
        fp <= TARGET_FP_MAX and
        precision >= TARGET_PRECISION_MIN and
        recall >= TARGET_RECALL_MIN
    )
    
    return {
        'policy': policy_name,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'meets_all_constraints': bool(meets_constraints),
        'fn_excellent': bool(fn < 5)
    }


def main():
    logger.info("="*80)
    logger.info("POST-HOC THRESHOLDING FOR SWIN-1")
    logger.info("="*80)
    
    # Load OOF predictions
    logger.info(f"Loading OOF predictions from: {OOF_FILE}")
    df = pd.read_csv(OOF_FILE)
    logger.info(f"Loaded {len(df)} patients")
    logger.info(f"Columns: {list(df.columns)}")
    
    if 'hgg_prob_swin' not in df.columns:
        raise ValueError("Column 'hgg_prob_swin' not found in OOF predictions")
    
    # Verify fold structure
    folds = sorted(df['fold'].unique())
    logger.info(f"Folds: {folds}")
    
    # Store results
    all_results = {}
    
    # Policy A: Baseline
    logger.info("\n" + "="*80)
    logger.info("POLICY A: BASELINE (threshold=0.5)")
    logger.info("="*80)
    
    baseline_results = []
    for fold in folds:
        fold_mask = df['fold'] == fold
        y_true = df.loc[fold_mask, 'label'].values
        y_proba = df.loc[fold_mask, 'hgg_prob_swin'].values
        y_pred = policy_baseline(y_proba, threshold=0.5)
        
        metrics = evaluate_policy(y_true, y_pred, y_proba, f"Baseline_fold_{fold}")
        baseline_results.append(metrics)
        
        logger.info(f"Fold {fold}: FN={metrics['fn']}, FP={metrics['fp']}, "
                   f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    # Aggregate baseline
    baseline_agg = aggregate_results(baseline_results, "Baseline")
    all_results['baseline'] = {'fold_results': baseline_results, 'aggregated': baseline_agg}
    
    # Policy B: Reject-band
    logger.info("\n" + "="*80)
    logger.info("POLICY B: REJECT-BAND (prob in [0.35, 0.65] → HGG)")
    logger.info("="*80)
    
    reject_band_results = []
    for fold in folds:
        fold_mask = df['fold'] == fold
        y_true = df.loc[fold_mask, 'label'].values
        y_proba = df.loc[fold_mask, 'hgg_prob_swin'].values
        y_pred = policy_reject_band(y_proba, lower=0.35, upper=0.65)
        
        metrics = evaluate_policy(y_true, y_pred, y_proba, f"RejectBand_fold_{fold}")
        reject_band_results.append(metrics)
        
        logger.info(f"Fold {fold}: FN={metrics['fn']}, FP={metrics['fp']}, "
                   f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    reject_band_agg = aggregate_results(reject_band_results, "RejectBand")
    all_results['reject_band'] = {'fold_results': reject_band_results, 'aggregated': reject_band_agg}
    
    # Policy C: Confidence-aware
    logger.info("\n" + "="*80)
    logger.info("POLICY C: CONFIDENCE-AWARE THRESHOLDING")
    logger.info("="*80)
    
    confidence_results = []
    for fold in folds:
        fold_mask = df['fold'] == fold
        train_mask = df['fold'] != fold
        
        y_true_val = df.loc[fold_mask, 'label'].values
        y_proba_val = df.loc[fold_mask, 'hgg_prob_swin'].values
        y_true_train = df.loc[train_mask, 'label'].values
        y_proba_train = df.loc[train_mask, 'hgg_prob_swin'].values
        
        y_pred = policy_confidence_aware(y_proba_val, y_true_train, y_proba_train, uncertainty_percentile=75.0)
        
        metrics = evaluate_policy(y_true_val, y_pred, y_proba_val, f"ConfidenceAware_fold_{fold}")
        confidence_results.append(metrics)
        
        logger.info(f"Fold {fold}: FN={metrics['fn']}, FP={metrics['fp']}, "
                   f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    confidence_agg = aggregate_results(confidence_results, "ConfidenceAware")
    all_results['confidence_aware'] = {'fold_results': confidence_results, 'aggregated': confidence_agg}
    
    # Policy D: Fold-specific calibrated
    logger.info("\n" + "="*80)
    logger.info("POLICY D: FOLD-SPECIFIC CALIBRATED THRESHOLD")
    logger.info("="*80)
    
    calibrated_results = []
    for fold in folds:
        fold_mask = df['fold'] == fold
        y_true_val = df.loc[fold_mask, 'label'].values
        y_proba_val = df.loc[fold_mask, 'hgg_prob_swin'].values
        
        y_pred, optimal_threshold = policy_fold_specific_calibrated(df, fold)
        
        metrics = evaluate_policy(y_true_val, y_pred, y_proba_val, f"FoldCalibrated_fold_{fold}")
        metrics['optimal_threshold'] = float(optimal_threshold) if optimal_threshold is not None else None
        calibrated_results.append(metrics)
        
        logger.info(f"Fold {fold}: Threshold={optimal_threshold:.3f}, FN={metrics['fn']}, FP={metrics['fp']}, "
                   f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    calibrated_agg = aggregate_results(calibrated_results, "FoldCalibrated")
    all_results['fold_calibrated'] = {'fold_results': calibrated_results, 'aggregated': calibrated_agg}
    
    # Save results
    json_path = OUTPUT_DIR / 'thresholding_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\n✓ Saved results to: {json_path}")
    
    # Generate markdown report
    generate_markdown_report(all_results, OUTPUT_DIR)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    
    for policy_name, policy_data in all_results.items():
        agg = policy_data['aggregated']
        logger.info(f"\n{policy_name.upper()}:")
        logger.info(f"  FN: {agg['fn_mean']:.1f} ± {agg['fn_std']:.1f} (target: <{TARGET_FN_MAX})")
        logger.info(f"  FP: {agg['fp_mean']:.1f} ± {agg['fp_std']:.1f} (target: <{TARGET_FP_MAX})")
        logger.info(f"  Precision: {agg['precision_mean']:.4f} ± {agg['precision_std']:.4f} (target: ≥{TARGET_PRECISION_MIN})")
        logger.info(f"  Recall: {agg['recall_mean']:.4f} ± {agg['recall_std']:.4f} (target: ≥{TARGET_RECALL_MIN})")
        logger.info(f"  Meets ALL constraints: {agg['meets_all_constraints']}")
        if agg['meets_all_constraints']:
            logger.info(f"  ✓ EXCELLENT: FN < 5 achieved: {agg['fn_excellent']}")


def aggregate_results(fold_results: List[Dict], policy_name: str) -> Dict:
    """Aggregate results across folds."""
    fn_values = [r['fn'] for r in fold_results]
    fp_values = [r['fp'] for r in fold_results]
    precision_values = [r['precision'] for r in fold_results]
    recall_values = [r['recall'] for r in fold_results]
    f1_values = [r['f1'] for r in fold_results]
    auc_roc_values = [r['auc_roc'] for r in fold_results]
    
    return {
        'policy': policy_name,
        'fn_mean': float(np.mean(fn_values)),
        'fn_std': float(np.std(fn_values)),
        'fp_mean': float(np.mean(fp_values)),
        'fp_std': float(np.std(fp_values)),
        'precision_mean': float(np.mean(precision_values)),
        'precision_std': float(np.std(precision_values)),
        'recall_mean': float(np.mean(recall_values)),
        'recall_std': float(np.std(recall_values)),
        'f1_mean': float(np.mean(f1_values)),
        'f1_std': float(np.std(f1_values)),
        'auc_roc_mean': float(np.mean(auc_roc_values)),
        'auc_roc_std': float(np.std(auc_roc_values)),
        'meets_all_constraints': bool(
            np.mean(fn_values) <= TARGET_FN_MAX and
            np.mean(fp_values) <= TARGET_FP_MAX and
            np.mean(precision_values) >= TARGET_PRECISION_MIN and
            np.mean(recall_values) >= TARGET_RECALL_MIN
        ),
        'fn_excellent': bool(np.mean(fn_values) < 5)
    }


def generate_markdown_report(all_results: Dict, output_dir: Path):
    """Generate markdown comparison report."""
    md_content = "# Post-Hoc Thresholding Results for Swin-1\n\n"
    md_content += "## Target Constraints\n\n"
    md_content += f"- FN < {TARGET_FN_MAX} (FN < 5 is excellent)\n"
    md_content += f"- FP < {TARGET_FP_MAX}\n"
    md_content += f"- Precision ≥ {TARGET_PRECISION_MIN}\n"
    md_content += f"- Recall ≥ {TARGET_RECALL_MIN}\n\n"
    md_content += "**All constraints must be met simultaneously.**\n\n"
    
    md_content += "## Results Comparison\n\n"
    md_content += "| Policy | FN (mean±std) | FP (mean±std) | Precision (mean±std) | Recall (mean±std) | Meets All? |\n"
    md_content += "|--------|---------------|---------------|---------------------|-------------------|------------|\n"
    
    for policy_name, policy_data in all_results.items():
        agg = policy_data['aggregated']
        meets = "✅ YES" if agg['meets_all_constraints'] else "❌ NO"
        md_content += f"| {agg['policy']} | {agg['fn_mean']:.1f}±{agg['fn_std']:.1f} | "
        md_content += f"{agg['fp_mean']:.1f}±{agg['fp_std']:.1f} | "
        md_content += f"{agg['precision_mean']:.4f}±{agg['precision_std']:.4f} | "
        md_content += f"{agg['recall_mean']:.4f}±{agg['recall_std']:.4f} | {meets} |\n"
    
    md_content += "\n## Executive Summary\n\n"
    
    # Find best policy
    best_policy = None
    for policy_name, policy_data in all_results.items():
        agg = policy_data['aggregated']
        if agg['meets_all_constraints']:
            if best_policy is None or agg['fn_mean'] < all_results[best_policy]['aggregated']['fn_mean']:
                best_policy = policy_name
    
    if best_policy:
        best_agg = all_results[best_policy]['aggregated']
        md_content += f"**Best Policy: {best_agg['policy']}**\n\n"
        md_content += f"- FN: {best_agg['fn_mean']:.1f} ± {best_agg['fn_std']:.1f}\n"
        md_content += f"- FP: {best_agg['fp_mean']:.1f} ± {best_agg['fp_std']:.1f}\n"
        md_content += f"- Precision: {best_agg['precision_mean']:.4f} ± {best_agg['precision_std']:.4f}\n"
        md_content += f"- Recall: {best_agg['recall_mean']:.4f} ± {best_agg['recall_std']:.4f}\n"
        if best_agg['fn_excellent']:
            md_content += f"\n✅ **EXCELLENT: FN < 5 achieved!**\n"
        else:
            md_content += f"\n⚠️ FN < 5 not achieved (FN < {TARGET_FN_MAX} is acceptable)\n"
    else:
        md_content += "**❌ NO POLICY MEETS ALL CONSTRAINTS**\n\n"
        md_content += "None of the thresholding policies achieve:\n"
        md_content += f"- FN < {TARGET_FN_MAX} AND\n"
        md_content += f"- FP < {TARGET_FP_MAX} AND\n"
        md_content += f"- Precision ≥ {TARGET_PRECISION_MIN} AND\n"
        md_content += f"- Recall ≥ {TARGET_RECALL_MIN}\n\n"
        md_content += "**Next Step:** Proceed to Part B (Feature-level rescue)\n"
    
    md_path = output_dir / 'thresholding_results.md'
    with open(md_path, 'w') as f:
        f.write(md_content)
    logger.info(f"✓ Saved markdown report to: {md_path}")


if __name__ == '__main__':
    main()

