#!/usr/bin/env python3
"""
Threshold Tuning for Ensemble Classifier

This script performs threshold tuning on OOF predictions to reduce False Negatives
and improve Recall for HGG classification.

Usage:
    python scripts/ensemble/threshold_tuning.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple, List
import logging

from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
MERGED_OOF_FILE = Path('ensemble/oof_predictions/merged_oof_predictions.csv')
METRICS_FILE = Path('ensemble/results/meta_learner_metrics.json')
OUTPUT_DIR = Path('ensemble/results')
THRESHOLD_RESULTS_FILE = OUTPUT_DIR / 'threshold_tuning_results.json'

# Meta-learner coefficients (from meta_learner_metrics.json)
# These will be loaded from the metrics file
FEATURE_COLUMNS = ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']


def load_data() -> Tuple[pd.DataFrame, Dict]:
    """Load OOF predictions and meta-learner coefficients."""
    logger.info("Loading data...")
    
    # Load OOF predictions
    if not MERGED_OOF_FILE.exists():
        raise FileNotFoundError(f"OOF predictions file not found: {MERGED_OOF_FILE}")
    
    df = pd.read_csv(MERGED_OOF_FILE)
    logger.info(f"Loaded {len(df)} samples from {MERGED_OOF_FILE}")
    
    # Load meta-learner coefficients
    if not METRICS_FILE.exists():
        raise FileNotFoundError(f"Meta-learner metrics file not found: {METRICS_FILE}")
    
    with open(METRICS_FILE, 'r') as f:
        metrics = json.load(f)
    
    coefficients = metrics['model_coefficients']
    intercept = metrics['model_intercept']
    
    logger.info(f"Meta-learner coefficients:")
    for feature, coef in coefficients.items():
        logger.info(f"  {feature}: {coef:.6f}")
    logger.info(f"  Intercept: {intercept:.6f}")
    
    return df, {'coefficients': coefficients, 'intercept': intercept}


def compute_ensemble_probabilities(df: pd.DataFrame, meta_params: Dict) -> np.ndarray:
    """
    Compute ensemble probabilities from base model probabilities using meta-learner.
    
    Formula: P(HGG) = sigmoid(coef_resnet * prob_resnet + coef_swin * prob_swin + 
                              coef_mil * prob_mil + intercept)
    """
    logger.info("Computing ensemble probabilities...")
    
    coefficients = meta_params['coefficients']
    intercept = meta_params['intercept']
    
    # Compute logit (linear combination)
    logit = (
        coefficients['hgg_prob_resnet'] * df['hgg_prob_resnet'] +
        coefficients['hgg_prob_swin'] * df['hgg_prob_swin'] +
        coefficients['hgg_prob_mil'] * df['hgg_prob_mil'] +
        intercept
    )
    
    # Apply sigmoid to get probabilities
    ensemble_probs = 1 / (1 + np.exp(-logit))
    
    logger.info(f"Ensemble probabilities: min={ensemble_probs.min():.6f}, "
                f"max={ensemble_probs.max():.6f}, mean={ensemble_probs.mean():.6f}")
    
    return ensemble_probs


def compute_metrics_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, 
                                  threshold: float) -> Dict:
    """Compute classification metrics at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    return {
        'threshold': threshold,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy)
    }


def threshold_sweep(y_true: np.ndarray, y_proba: np.ndarray, 
                    thresholds: np.ndarray) -> List[Dict]:
    """Perform threshold sweep and compute metrics for each threshold."""
    logger.info(f"Performing threshold sweep from {thresholds.min():.2f} to "
                f"{thresholds.max():.2f} (step: {thresholds[1] - thresholds[0]:.2f})")
    
    results = []
    for threshold in thresholds:
        metrics = compute_metrics_at_threshold(y_true, y_proba, threshold)
        results.append(metrics)
    
    logger.info(f"Computed metrics for {len(results)} thresholds")
    return results


def find_best_thresholds(results: List[Dict]) -> Dict:
    """Find best thresholds for different policies."""
    df_results = pd.DataFrame(results)
    
    best_thresholds = {}
    
    # Policy A: Target Recall >= 0.85 AND Precision >= 0.80
    recall_85 = df_results[(df_results['recall'] >= 0.85) & (df_results['precision'] >= 0.80)]
    if len(recall_85) > 0:
        best_85 = recall_85.loc[recall_85['threshold'].idxmin()]
        best_thresholds['policy_a_recall_85'] = best_85.to_dict()
        logger.info(f"Policy A (Recall >= 0.85 AND Precision >= 0.80): threshold={best_85['threshold']:.3f}, "
                   f"recall={best_85['recall']:.4f}, precision={best_85['precision']:.4f}, fn={best_85['fn']}, fp={best_85['fp']}")
    else:
        logger.warning("Policy A (Recall >= 0.85 AND Precision >= 0.80): Not achievable")
    
    # Policy A: Target Recall >= 0.90 AND Precision >= 0.80
    recall_90 = df_results[(df_results['recall'] >= 0.90) & (df_results['precision'] >= 0.80)]
    if len(recall_90) > 0:
        best_90 = recall_90.loc[recall_90['threshold'].idxmin()]
        best_thresholds['policy_a_recall_90'] = best_90.to_dict()
        logger.info(f"Policy A (Recall >= 0.90 AND Precision >= 0.80): threshold={best_90['threshold']:.3f}, "
                   f"recall={best_90['recall']:.4f}, precision={best_90['precision']:.4f}, fn={best_90['fn']}, fp={best_90['fp']}")
    else:
        logger.warning("Policy A (Recall >= 0.90 AND Precision >= 0.80): Not achievable")
    
    # Policy B: Max F1
    best_f1_idx = df_results['f1'].idxmax()
    best_f1 = df_results.loc[best_f1_idx]
    best_thresholds['policy_b_max_f1'] = best_f1.to_dict()
    logger.info(f"Policy B (Max F1): threshold={best_f1['threshold']:.3f}, "
               f"f1={best_f1['f1']:.4f}, recall={best_f1['recall']:.4f}, "
               f"fn={best_f1['fn']}, fp={best_f1['fp']}")
    
    # Policy C: Min FN with Precision >= 0.90
    precision_90 = df_results[df_results['precision'] >= 0.90]
    if len(precision_90) > 0:
        best_fn = precision_90.loc[precision_90['fn'].idxmin()]
        best_thresholds['policy_c_min_fn_precision_90'] = best_fn.to_dict()
        logger.info(f"Policy C (Min FN, Precision >= 0.90): threshold={best_fn['threshold']:.3f}, "
                   f"fn={best_fn['fn']}, fp={best_fn['fp']}, recall={best_fn['recall']:.4f}, "
                   f"precision={best_fn['precision']:.4f}")
    else:
        logger.warning("Policy C (Min FN, Precision >= 0.90): Not achievable")
        # Try with lower precision constraint
        precision_85 = df_results[df_results['precision'] >= 0.85]
        if len(precision_85) > 0:
            best_fn = precision_85.loc[precision_85['fn'].idxmin()]
            best_thresholds['policy_c_min_fn_precision_85'] = best_fn.to_dict()
            logger.info(f"Policy C (Min FN, Precision >= 0.85): threshold={best_fn['threshold']:.3f}, "
                       f"fn={best_fn['fn']}, fp={best_fn['fp']}, recall={best_fn['recall']:.4f}, "
                       f"precision={best_fn['precision']:.4f}")
    
    return best_thresholds


def main():
    """Main function."""
    logger.info("=" * 80)
    logger.info("Threshold Tuning for Ensemble Classifier")
    logger.info("=" * 80)
    
    # Load data
    df, meta_params = load_data()
    
    # Extract true labels
    y_true = df['label'].values
    logger.info(f"Class distribution: LGG={np.sum(y_true == 0)}, HGG={np.sum(y_true == 1)}")
    
    # Compute ensemble probabilities
    ensemble_probs = compute_ensemble_probabilities(df, meta_params)
    
    # Baseline metrics (threshold = 0.5)
    baseline_metrics = compute_metrics_at_threshold(y_true, ensemble_probs, 0.5)
    logger.info("\n" + "=" * 80)
    logger.info("Baseline Metrics (Threshold = 0.5)")
    logger.info("=" * 80)
    logger.info(f"  TN: {baseline_metrics['tn']}, FP: {baseline_metrics['fp']}")
    logger.info(f"  FN: {baseline_metrics['fn']}, TP: {baseline_metrics['tp']}")
    logger.info(f"  Precision: {baseline_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {baseline_metrics['recall']:.4f}")
    logger.info(f"  F1-Score:  {baseline_metrics['f1']:.4f}")
    logger.info(f"  Accuracy:  {baseline_metrics['accuracy']:.4f}")
    
    # Threshold sweep
    thresholds = np.arange(0.05, 0.96, 0.01)
    results = threshold_sweep(y_true, ensemble_probs, thresholds)
    
    # Find best thresholds
    logger.info("\n" + "=" * 80)
    logger.info("Best Thresholds by Policy")
    logger.info("=" * 80)
    best_thresholds = find_best_thresholds(results)
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        'baseline_metrics': baseline_metrics,
        'threshold_sweep_results': results,
        'best_thresholds': best_thresholds,
        'metadata': {
            'n_samples': len(df),
            'class_distribution': {
                'LGG': int(np.sum(y_true == 0)),
                'HGG': int(np.sum(y_true == 1))
            },
            'threshold_range': [float(thresholds.min()), float(thresholds.max())],
            'threshold_step': float(thresholds[1] - thresholds[0])
        }
    }
    
    with open(THRESHOLD_RESULTS_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\nResults saved to: {THRESHOLD_RESULTS_FILE}")
    
    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("Summary Table: Best Thresholds")
    logger.info("=" * 80)
    
    summary_data = []
    summary_data.append({
        'Policy': 'Baseline (0.5)',
        'Threshold': baseline_metrics['threshold'],
        'FN': baseline_metrics['fn'],
        'FP': baseline_metrics['fp'],
        'Recall': baseline_metrics['recall'],
        'Precision': baseline_metrics['precision'],
        'F1': baseline_metrics['f1'],
        'Accuracy': baseline_metrics['accuracy']
    })
    
    for policy_name, policy_data in best_thresholds.items():
        summary_data.append({
            'Policy': policy_name,
            'Threshold': policy_data['threshold'],
            'FN': policy_data['fn'],
            'FP': policy_data['fp'],
            'Recall': policy_data['recall'],
            'Precision': policy_data['precision'],
            'F1': policy_data['f1'],
            'Accuracy': policy_data['accuracy']
        })
    
    summary_df = pd.DataFrame(summary_data)
    logger.info("\n" + summary_df.to_string(index=False))
    
    logger.info("\n" + "=" * 80)
    logger.info("Threshold Tuning Complete")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

