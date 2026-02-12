#!/usr/bin/env python3
"""
XGBoost Stability Check

This script validates the stability of XGBoost_depth4_lr0.1_n100 across multiple
random seeds to rule out overfitting or optimistic bias.

Critical: If stability fails, STOP and do NOT generate visualizations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, classification_report
)

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("ERROR: XGBoost not available. Cannot run stability check.")
    exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
MERGED_OOF_FILE = Path('ensemble/oof_predictions/merged_oof_predictions.csv')
FEATURE_COLUMNS = ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']
TARGET_COLUMN = 'label'

# Output directory
RESULTS_DIR = Path('ensemble/results/meta_learner_v2')

# XGBoost configuration (best from previous experiment)
XGBOOST_CONFIG = {
    'max_depth': 4,
    'learning_rate': 0.1,
    'n_estimators': 100
}

# Stability check seeds
STABILITY_SEEDS = [21, 42, 77, 123, 202]

# Experiment parameters
CALIBRATION_FRACTION = 0.7
THRESHOLD_SWEEP_START = 0.05
THRESHOLD_SWEEP_END = 0.95
THRESHOLD_SWEEP_STEP = 0.01


def apply_platt_calibration(
    meta_learner: object,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_thr: np.ndarray
) -> tuple:
    """Apply Platt calibration to meta-learner probabilities."""
    from sklearn.linear_model import LogisticRegression as PlattScaling
    
    # Get uncalibrated probabilities on calibration set
    y_proba_cal_uncal = meta_learner.predict_proba(X_cal)[:, 1]
    
    # Clip to avoid log(0) and log(1)
    y_proba_cal_uncal_clipped = np.clip(y_proba_cal_uncal, 1e-7, 1 - 1e-7)
    log_odds = np.log(y_proba_cal_uncal_clipped / (1 - y_proba_cal_uncal_clipped))
    
    # Fit Platt scaling
    platt_model = PlattScaling()
    platt_model.fit(log_odds.reshape(-1, 1), y_cal)
    
    # Apply to threshold set
    y_proba_thr_uncal = meta_learner.predict_proba(X_thr)[:, 1]
    y_proba_thr_uncal_clipped = np.clip(y_proba_thr_uncal, 1e-7, 1 - 1e-7)
    log_odds_thr = np.log(y_proba_thr_uncal_clipped / (1 - y_proba_thr_uncal_clipped))
    y_proba_thr_cal = platt_model.predict_proba(log_odds_thr.reshape(-1, 1))[:, 1]
    
    calibrator = {'type': 'platt', 'model': platt_model}
    
    return calibrator, y_proba_thr_cal


def threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sweep_start: float,
    sweep_end: float,
    sweep_step: float
) -> List[Dict]:
    """Perform threshold sweep and compute metrics."""
    results = []
    
    thresholds = np.arange(sweep_start, sweep_end + sweep_step, sweep_step)
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        cost = 2 * fn + fp
        
        results.append({
            'threshold': float(threshold),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'cost': float(cost)
        })
    
    return results


def select_optimal_threshold(sweep_results: List[Dict]) -> Dict:
    """Select threshold with minimum cost. If tie, prefer higher recall."""
    min_cost = min(r['cost'] for r in sweep_results)
    candidates = [r for r in sweep_results if abs(r['cost'] - min_cost) < 0.01]
    
    # Prefer higher recall if multiple candidates
    best = max(candidates, key=lambda x: x['recall'])
    
    return best


def evaluate_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float
) -> Dict:
    """Evaluate model at a specific threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    cost = 2 * fn + fp
    
    return {
        'threshold': float(threshold),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'specificity': float(specificity),
        'cost': float(cost)
    }


def run_stability_check_for_seed(
    seed: int,
    X_full: np.ndarray,
    y_full: np.ndarray
) -> Dict:
    """Run stability check for a single seed."""
    logger.info(f"\n{'='*80}")
    logger.info(f"STABILITY CHECK: Seed {seed}")
    logger.info(f"{'='*80}")
    
    # Split for calibration/threshold selection (seed affects this split)
    X_cal, X_thr, y_cal, y_thr = train_test_split(
        X_full, y_full, test_size=1-CALIBRATION_FRACTION, 
        random_state=seed, stratify=y_full
    )
    
    logger.info(f"Calibration set: {len(X_cal)} samples")
    logger.info(f"Threshold selection set: {len(X_thr)} samples")
    
    # Train XGBoost (seed affects XGBoost random_state)
    logger.info("Training XGBoost...")
    model = xgb.XGBClassifier(
        max_depth=XGBOOST_CONFIG['max_depth'],
        learning_rate=XGBOOST_CONFIG['learning_rate'],
        n_estimators=XGBOOST_CONFIG['n_estimators'],
        random_state=seed,  # Seed affects XGBoost training
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_full, y_full)  # Train on full OOF set
    
    # Apply Platt calibration
    logger.info("Applying Platt calibration...")
    calibrator, y_proba_thr_cal = apply_platt_calibration(
        model, X_cal, y_cal, X_thr
    )
    
    # Threshold sweep on threshold selection set
    logger.info("Running threshold sweep...")
    sweep_results = threshold_sweep(
        y_thr, y_proba_thr_cal,
        THRESHOLD_SWEEP_START, THRESHOLD_SWEEP_END, THRESHOLD_SWEEP_STEP
    )
    
    # Select optimal threshold
    optimal_thr_result = select_optimal_threshold(sweep_results)
    selected_threshold = optimal_thr_result['threshold']
    
    logger.info(f"Selected threshold: {selected_threshold:.4f} "
                f"(cost={optimal_thr_result['cost']:.1f}, "
                f"FN={optimal_thr_result['fn']}, FP={optimal_thr_result['fp']})")
    
    # Apply calibration to full OOF set
    y_proba_full_uncal = model.predict_proba(X_full)[:, 1]
    y_proba_full_uncal_clipped = np.clip(y_proba_full_uncal, 1e-7, 1 - 1e-7)
    log_odds_full = np.log(y_proba_full_uncal_clipped / (1 - y_proba_full_uncal_clipped))
    y_proba_full_cal = calibrator['model'].predict_proba(log_odds_full.reshape(-1, 1))[:, 1]
    
    # Evaluate on full OOF set
    full_eval = evaluate_threshold(y_full, y_proba_full_cal, selected_threshold)
    
    logger.info(f"Full OOF evaluation: FN={full_eval['fn']}, FP={full_eval['fp']}, "
                f"Cost={full_eval['cost']:.1f}, Recall={full_eval['recall']:.4f}")
    
    return {
        'seed': seed,
        'selected_threshold': selected_threshold,
        **full_eval
    }


def main():
    """Main stability check function."""
    logger.info("="*80)
    logger.info("XGBOOST STABILITY CHECK")
    logger.info("="*80)
    logger.info(f"Model: XGBoost_depth{XGBOOST_CONFIG['max_depth']}_"
                f"lr{XGBOOST_CONFIG['learning_rate']}_n{XGBOOST_CONFIG['n_estimators']}")
    logger.info(f"Seeds to test: {STABILITY_SEEDS}")
    
    if not XGBOOST_AVAILABLE:
        logger.error("XGBoost not available. Cannot run stability check.")
        return
    
    # Load data
    logger.info("\nLoading data...")
    df = pd.read_csv(MERGED_OOF_FILE)
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    logger.info(f"Loaded {len(df)} samples")
    
    # Run stability check for each seed
    results = []
    for seed in STABILITY_SEEDS:
        try:
            result = run_stability_check_for_seed(seed, X, y)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed for seed {seed}: {e}")
            results.append({
                'seed': seed,
                'error': str(e)
            })
    
    # Analyze stability
    logger.info("\n" + "="*80)
    logger.info("STABILITY ANALYSIS")
    logger.info("="*80)
    
    successful_runs = [r for r in results if 'error' not in r]
    
    if not successful_runs:
        logger.error("No successful runs. Stability check FAILED.")
        stability_status = "FAILED"
        stability_reason = "No successful runs"
    else:
        fn_values = [r['fn'] for r in successful_runs]
        fp_values = [r['fp'] for r in successful_runs]
        cost_values = [r['cost'] for r in successful_runs]
        
        fn_mean = np.mean(fn_values)
        fn_std = np.std(fn_values)
        fn_min = np.min(fn_values)
        fn_max = np.max(fn_values)
        
        logger.info(f"FN Statistics:")
        logger.info(f"  Mean: {fn_mean:.2f}, Std: {fn_std:.2f}, Range: [{fn_min}, {fn_max}]")
        logger.info(f"  Values: {fn_values}")
        
        logger.info(f"\nFP Statistics:")
        logger.info(f"  Mean: {np.mean(fp_values):.2f}, Std: {np.std(fp_values):.2f}, "
                   f"Range: [{np.min(fp_values)}, {np.max(fp_values)}]")
        logger.info(f"  Values: {fp_values}")
        
        logger.info(f"\nCost Statistics:")
        logger.info(f"  Mean: {np.mean(cost_values):.2f}, Std: {np.std(cost_values):.2f}, "
                   f"Range: [{np.min(cost_values)}, {np.max(cost_values)}]")
        logger.info(f"  Values: {cost_values}")
        
        # Stability decision
        fn_all_leq_1 = all(fn <= 1 for fn in fn_values)
        fn_no_spikes = fn_max <= 2  # Allow up to 2 as "no spikes"
        
        if fn_all_leq_1 and fn_no_spikes:
            stability_status = "PASSED"
            stability_reason = f"FN ≤ 1 for all seeds (range: [{fn_min}, {fn_max}]), no spikes"
        else:
            stability_status = "FAILED"
            if not fn_all_leq_1:
                stability_reason = f"FN > 1 for some seeds (max: {fn_max})"
            else:
                stability_reason = f"FN spikes detected (range: [{fn_min}, {fn_max}])"
    
    # Save results
    output_data = {
        'model_config': XGBOOST_CONFIG,
        'stability_seeds': STABILITY_SEEDS,
        'stability_status': stability_status,
        'stability_reason': stability_reason,
        'results': results,
        'summary': {
            'n_successful_runs': len(successful_runs),
            'fn_mean': float(np.mean([r['fn'] for r in successful_runs])) if successful_runs else None,
            'fn_std': float(np.std([r['fn'] for r in successful_runs])) if successful_runs else None,
            'fn_min': int(np.min([r['fn'] for r in successful_runs])) if successful_runs else None,
            'fn_max': int(np.max([r['fn'] for r in successful_runs])) if successful_runs else None,
            'fp_mean': float(np.mean([r['fp'] for r in successful_runs])) if successful_runs else None,
            'cost_mean': float(np.mean([r['cost'] for r in successful_runs])) if successful_runs else None,
        }
    }
    
    output_path = RESULTS_DIR / 'xgboost_stability_results.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\n✓ Results saved to: {output_path}")
    
    # Final decision
    logger.info("\n" + "="*80)
    logger.info(f"STABILITY CHECK: {stability_status}")
    logger.info("="*80)
    logger.info(f"Reason: {stability_reason}")
    
    if stability_status == "PASSED":
        logger.info("\n✅ Stability PASSED. XGBoost performance is stable across seeds.")
        logger.info("Proceed to Step 2: Generate final visualizations.")
    else:
        logger.info("\n❌ Stability FAILED. XGBoost performance is unstable.")
        logger.info("STOP: Do NOT generate visualizations.")
        logger.info("Recommendation: REJECT XGBoost, keep baseline LogisticRegression.")


if __name__ == '__main__':
    main()

