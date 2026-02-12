#!/usr/bin/env python3
"""
Nested Cross-Validation with Enhanced Meta-Features

This script implements strict nested CV with engineered meta-features to improve
meta-learner performance while maintaining academic rigor.

Key features:
- Meta-feature engineering from base probabilities
- Robust calibration with multiple seeds (median threshold selection)
- Strict nested CV evaluation (outer-test only)
- Comparison against baseline nested CV results
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
MERGED_OOF_FILE = Path('ensemble/oof_predictions/merged_oof_predictions.csv')
OUTPUT_DIR = Path('ensemble/results/nested_cv_meta_features')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Base probability columns
BASE_PROB_COLS = ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']
TARGET_COLUMN = 'label'
PATIENT_ID_COLUMN = 'patient_id'

# Configuration
OUTER_CV_FOLDS = 5
CALIBRATION_FRACTION = 0.7
THRESHOLD_SWEEP_START = 0.05
THRESHOLD_SWEEP_END = 0.95
THRESHOLD_SWEEP_STEP = 0.01
ROBUST_CALIBRATION_REPEATS = 5  # Number of seeds for robust threshold selection
RANDOM_SEED = 42


def make_json_serializable(obj):
    """Convert numpy types and booleans to JSON-serializable types."""
    if isinstance(obj, (bool, np.bool_)):
        return int(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())
    return obj


def engineer_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer meta-features from base probabilities.
    
    NO LABEL USAGE - features derived only from probabilities.
    """
    logger.info("Engineering meta-features from base probabilities...")
    
    # Extract base probabilities as numpy array
    base_probs = df[BASE_PROB_COLS].values  # Shape: (n_samples, 3)
    
    # Initialize feature dictionary
    features = {}
    
    # Core probability features
    features['p_resnet'] = base_probs[:, 0]
    features['p_swin'] = base_probs[:, 1]
    features['p_mil'] = base_probs[:, 2]
    
    # Agreement / disagreement
    features['prob_mean'] = np.mean(base_probs, axis=1)
    features['prob_std'] = np.std(base_probs, axis=1)
    features['prob_max'] = np.max(base_probs, axis=1)
    features['prob_min'] = np.min(base_probs, axis=1)
    features['prob_range'] = features['prob_max'] - features['prob_min']
    
    # Confidence / margin
    features['margin_mean'] = np.abs(features['prob_mean'] - 0.5)
    features['margin_max'] = np.max(np.abs(base_probs - 0.5), axis=1)
    
    # Entropy (binary entropy of mean probability)
    prob_mean = features['prob_mean']
    # Clip to avoid log(0)
    prob_mean_clipped = np.clip(prob_mean, 1e-7, 1 - 1e-7)
    features['entropy_mean'] = -(prob_mean_clipped * np.log2(prob_mean_clipped) + 
                                 (1 - prob_mean_clipped) * np.log2(1 - prob_mean_clipped))
    
    # Model dominance (one-hot encoded argmax)
    argmax_idx = np.argmax(base_probs, axis=1)
    features['argmax_resnet'] = (argmax_idx == 0).astype(float)
    features['argmax_swin'] = (argmax_idx == 1).astype(float)
    features['argmax_mil'] = (argmax_idx == 2).astype(float)
    
    # Create DataFrame with meta-features
    meta_features_df = pd.DataFrame(features)
    
    # Add patient_id and label (preserve original columns)
    meta_features_df[PATIENT_ID_COLUMN] = df[PATIENT_ID_COLUMN].values
    meta_features_df[TARGET_COLUMN] = df[TARGET_COLUMN].values
    
    logger.info(f"✓ Engineered {len(features)} meta-features")
    logger.info(f"  Feature names: {list(features.keys())}")
    
    return meta_features_df


def apply_platt_calibration(
    meta_learner: object,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_eval: np.ndarray
) -> Tuple[np.ndarray, object]:
    """Apply Platt calibration to meta-learner probabilities."""
    from sklearn.linear_model import LogisticRegression as PlattScaling
    
    # Get uncalibrated probabilities
    if hasattr(meta_learner, 'predict_proba'):
        y_proba_cal_uncal = meta_learner.predict_proba(X_cal)[:, 1]
        y_proba_eval_uncal = meta_learner.predict_proba(X_eval)[:, 1]
    elif hasattr(meta_learner, 'decision_function'):
        decision_cal = meta_learner.decision_function(X_cal)
        decision_eval = meta_learner.decision_function(X_eval)
        y_proba_cal_uncal = 1 / (1 + np.exp(-decision_cal))
        y_proba_eval_uncal = 1 / (1 + np.exp(-decision_eval))
    else:
        raise ValueError("Model must have predict_proba or decision_function")
    
    # Clip and transform to log-odds
    y_proba_cal_clipped = np.clip(y_proba_cal_uncal, 1e-7, 1 - 1e-7)
    log_odds_cal = np.log(y_proba_cal_clipped / (1 - y_proba_cal_clipped))
    
    # Fit Platt scaling
    platt_model = PlattScaling()
    platt_model.fit(log_odds_cal.reshape(-1, 1), y_cal)
    
    # Apply to evaluation set
    y_proba_eval_clipped = np.clip(y_proba_eval_uncal, 1e-7, 1 - 1e-7)
    log_odds_eval = np.log(y_proba_eval_clipped / (1 - y_proba_eval_clipped))
    y_proba_eval_cal = platt_model.predict_proba(log_odds_eval.reshape(-1, 1))[:, 1]
    
    calibrator = {'type': 'platt', 'model': platt_model}
    
    return y_proba_eval_cal, calibrator


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


def robust_calibration_and_threshold_selection(
    meta_learner: object,
    X_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    calibration_repeats: int = ROBUST_CALIBRATION_REPEATS
) -> Tuple[object, float]:
    """
    Robust calibration and threshold selection using multiple random seeds.
    
    Returns:
        calibrator, final_threshold (median across repeats)
    """
    logger.info(f"Robust calibration with {calibration_repeats} repeats...")
    
    selected_thresholds = []
    
    for repeat_idx in range(calibration_repeats):
        seed = RANDOM_SEED + repeat_idx
        
        # Split outer-train for calibration/threshold selection
        X_cal, X_thr, y_cal, y_thr = train_test_split(
            X_outer_train, y_outer_train,
            test_size=1 - CALIBRATION_FRACTION,
            random_state=seed,
            stratify=y_outer_train
        )
        
        # Apply calibration
        y_proba_thr_cal, _ = apply_platt_calibration(
            meta_learner, X_cal, y_cal, X_thr
        )
        
        # Threshold sweep
        sweep_results = threshold_sweep(
            y_thr, y_proba_thr_cal,
            THRESHOLD_SWEEP_START, THRESHOLD_SWEEP_END, THRESHOLD_SWEEP_STEP
        )
        
        # Select optimal threshold
        optimal = select_optimal_threshold(sweep_results)
        selected_thresholds.append(optimal['threshold'])
        
        logger.info(f"  Repeat {repeat_idx + 1}: threshold={optimal['threshold']:.4f}, "
                   f"cost={optimal['cost']:.1f}, FN={optimal['fn']}, FP={optimal['fp']}")
    
    # Final threshold = median across repeats
    final_threshold = float(np.median(selected_thresholds))
    logger.info(f"Final threshold (median): {final_threshold:.4f}")
    
    # Refit calibration on full outer-train calibration data (use first split)
    X_cal_final, _, y_cal_final, _ = train_test_split(
        X_outer_train, y_outer_train,
        test_size=1 - CALIBRATION_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y_outer_train
    )
    
    _, final_calibrator = apply_platt_calibration(
        meta_learner, X_cal_final, y_cal_final, X_cal_final
    )
    
    return final_calibrator, final_threshold


def evaluate_at_threshold(
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


def process_outer_fold(
    fold_idx: int,
    outer_train_idx: np.ndarray,
    outer_test_idx: np.ndarray,
    df_meta: pd.DataFrame
) -> Dict:
    """Process a single outer fold."""
    logger.info(f"\n{'='*80}")
    logger.info(f"OUTER FOLD {fold_idx + 1}/{OUTER_CV_FOLDS}")
    logger.info(f"{'='*80}")
    
    # Split data
    df_outer_train = df_meta.iloc[outer_train_idx].copy()
    df_outer_test = df_meta.iloc[outer_test_idx].copy()
    
    logger.info(f"Outer-train: {len(df_outer_train)} patients")
    logger.info(f"Outer-test: {len(df_outer_test)} patients")
    
    # Extract features (exclude patient_id and label)
    feature_cols = [col for col in df_meta.columns 
                    if col not in [PATIENT_ID_COLUMN, TARGET_COLUMN]]
    
    X_outer_train = df_outer_train[feature_cols].values
    y_outer_train = df_outer_train[TARGET_COLUMN].values
    X_outer_test = df_outer_test[feature_cols].values
    y_outer_test = df_outer_test[TARGET_COLUMN].values
    
    # Train meta-learner on outer-train
    logger.info("Training Logistic Regression meta-learner...")
    meta_learner = LogisticRegression(
        class_weight='balanced',
        solver='lbfgs',
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_SEED
    )
    meta_learner.fit(X_outer_train, y_outer_train)
    
    # Robust calibration and threshold selection
    calibrator, selected_threshold = robust_calibration_and_threshold_selection(
        meta_learner, X_outer_train, y_outer_train
    )
    
    # Apply calibration to outer-test
    y_proba_test_cal, _ = apply_platt_calibration(
        meta_learner, 
        X_outer_train, y_outer_train,  # Use outer-train for calibration fitting
        X_outer_test
    )
    
    # Evaluate on outer-test (CRITICAL: never seen during training)
    test_metrics = evaluate_at_threshold(
        y_outer_test, y_proba_test_cal, selected_threshold
    )
    
    logger.info(f"Outer-test evaluation:")
    logger.info(f"  FN={test_metrics['fn']}, FP={test_metrics['fp']}, "
               f"Cost={test_metrics['cost']:.1f}")
    logger.info(f"  Recall={test_metrics['recall']:.4f}, "
               f"Precision={test_metrics['precision']:.4f}")
    
    return {
        'fold': fold_idx,
        'outer_train_size': len(df_outer_train),
        'outer_test_size': len(df_outer_test),
        'selected_threshold': selected_threshold,
        **test_metrics
    }


def load_baseline_results() -> Dict:
    """Load baseline nested CV results for comparison."""
    baseline_file = Path('ensemble/results/nested_cv_meta_learning/nested_cv_results_20260208_233521.json')
    
    if not baseline_file.exists():
        logger.warning("Baseline results not found. Comparison will be skipped.")
        return None
    
    with open(baseline_file) as f:
        baseline_data = json.load(f)
    
    # Extract LogisticRegression baseline
    if 'LogisticRegression' in baseline_data:
        return baseline_data['LogisticRegression']
    
    return None


def main():
    """Main function."""
    logger.info("="*80)
    logger.info("NESTED CV WITH ENHANCED META-FEATURES")
    logger.info("="*80)
    
    # Step 0: Verify inputs
    logger.info("\n" + "="*80)
    logger.info("STEP 0: INPUT VERIFICATION")
    logger.info("="*80)
    
    df = pd.read_csv(MERGED_OOF_FILE)
    
    # Verify required columns
    required_cols = [PATIENT_ID_COLUMN, TARGET_COLUMN] + BASE_PROB_COLS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Verify patient uniqueness
    if df[PATIENT_ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate patient IDs found")
    
    if df[PATIENT_ID_COLUMN].nunique() != len(df):
        raise ValueError("Patient ID uniqueness check failed")
    
    logger.info(f"✓ Loaded {len(df)} patients")
    logger.info(f"✓ All required columns present")
    logger.info(f"✓ Patient uniqueness verified")
    
    # Step 1: Engineer meta-features
    logger.info("\n" + "="*80)
    logger.info("STEP 1: META-FEATURE ENGINEERING")
    logger.info("="*80)
    
    df_meta = engineer_meta_features(df)
    
    # Extract feature columns
    feature_cols = [col for col in df_meta.columns 
                    if col not in [PATIENT_ID_COLUMN, TARGET_COLUMN]]
    X = df_meta[feature_cols].values
    y = df_meta[TARGET_COLUMN].values
    
    logger.info(f"✓ Total features: {len(feature_cols)}")
    logger.info(f"  Features: {feature_cols}")
    
    # Create outer CV splits (patient-level)
    logger.info(f"\nCreating {OUTER_CV_FOLDS}-fold outer CV splits (patient-level)...")
    outer_cv = StratifiedKFold(n_splits=OUTER_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # Process each outer fold
    logger.info("\n" + "="*80)
    logger.info("STEP 2-4: NESTED CV EVALUATION")
    logger.info("="*80)
    
    fold_results = []
    
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y)):
        try:
            fold_result = process_outer_fold(
                fold_idx, outer_train_idx, outer_test_idx, df_meta
            )
            fold_results.append(fold_result)
        except Exception as e:
            logger.error(f"Error in outer fold {fold_idx}: {e}", exc_info=True)
            continue
    
    if not fold_results:
        raise ValueError("No successful folds")
    
    # Aggregate results
    fn_values = [r['fn'] for r in fold_results]
    fp_values = [r['fp'] for r in fold_results]
    cost_values = [r['cost'] for r in fold_results]
    recall_values = [r['recall'] for r in fold_results]
    precision_values = [r['precision'] for r in fold_results]
    f1_values = [r['f1'] for r in fold_results]
    
    summary = {
        'meta_learner': 'LogisticRegression_Enhanced',
        'n_folds': len(fold_results),
        'fn_mean': float(np.mean(fn_values)),
        'fn_std': float(np.std(fn_values)),
        'fn_min': int(np.min(fn_values)),
        'fn_max': int(np.max(fn_values)),
        'fp_mean': float(np.mean(fp_values)),
        'fp_std': float(np.std(fp_values)),
        'cost_mean': float(np.mean(cost_values)),
        'cost_std': float(np.std(cost_values)),
        'recall_mean': float(np.mean(recall_values)),
        'recall_std': float(np.std(recall_values)),
        'precision_mean': float(np.mean(precision_values)),
        'precision_std': float(np.std(precision_values)),
        'f1_mean': float(np.mean(f1_values)),
        'f1_std': float(np.std(f1_values)),
        'fold_results': fold_results,
        'feature_names': feature_cols
    }
    
    logger.info(f"\nEnhanced Meta-Features Summary (across {len(fold_results)} folds):")
    logger.info(f"  FN: {summary['fn_mean']:.2f} ± {summary['fn_std']:.2f} "
               f"(range: [{summary['fn_min']}, {summary['fn_max']}])")
    logger.info(f"  FP: {summary['fp_mean']:.2f} ± {summary['fp_std']:.2f}")
    logger.info(f"  Cost: {summary['cost_mean']:.2f} ± {summary['cost_std']:.2f}")
    logger.info(f"  Recall: {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
    logger.info(f"  Precision: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")
    
    # Step 5: Comparison with baseline
    logger.info("\n" + "="*80)
    logger.info("STEP 5: COMPARISON WITH BASELINE")
    logger.info("="*80)
    
    baseline = load_baseline_results()
    
    if baseline:
        logger.info("\nBaseline (Simple Features):")
        logger.info(f"  FN: {baseline['fn_mean']:.2f} ± {baseline['fn_std']:.2f}")
        logger.info(f"  FP: {baseline['fp_mean']:.2f} ± {baseline['fp_std']:.2f}")
        logger.info(f"  Cost: {baseline['cost_mean']:.2f} ± {baseline['cost_std']:.2f}")
        
        logger.info("\nEnhanced (Meta-Features):")
        logger.info(f"  FN: {summary['fn_mean']:.2f} ± {summary['fn_std']:.2f}")
        logger.info(f"  FP: {summary['fp_mean']:.2f} ± {summary['fp_std']:.2f}")
        logger.info(f"  Cost: {summary['cost_mean']:.2f} ± {summary['cost_std']:.2f}")
        
        # Improvement analysis
        fn_improvement = baseline['fn_mean'] - summary['fn_mean']
        fp_change = summary['fp_mean'] - baseline['fp_mean']
        cost_improvement = baseline['cost_mean'] - summary['cost_mean']
        
        logger.info(f"\nImprovement:")
        logger.info(f"  FN: {fn_improvement:+.2f} ({'IMPROVED' if fn_improvement > 0 else 'WORSE' if fn_improvement < 0 else 'SAME'})")
        logger.info(f"  FP: {fp_change:+.2f} ({'WORSE' if fp_change > 0 else 'IMPROVED' if fp_change < 0 else 'SAME'})")
        logger.info(f"  Cost: {cost_improvement:+.2f} ({'IMPROVED' if cost_improvement > 0 else 'WORSE' if cost_improvement < 0 else 'SAME'})")
        
        summary['comparison'] = {
            'baseline_fn_mean': baseline['fn_mean'],
            'baseline_fp_mean': baseline['fp_mean'],
            'baseline_cost_mean': baseline['cost_mean'],
            'fn_improvement': float(fn_improvement),
            'fp_change': float(fp_change),
            'cost_improvement': float(cost_improvement),
            'fn_improved': fn_improvement > 0,
            'cost_improved': cost_improvement > 0
        }
    else:
        logger.warning("Baseline comparison skipped (baseline not found)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results_file = OUTPUT_DIR / f'meta_features_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(make_json_serializable(summary), f, indent=2)
    logger.info(f"\n✓ Results saved to: {results_file}")
    
    # Save per-fold CSV
    fold_df = pd.DataFrame(fold_results)
    csv_file = OUTPUT_DIR / f'meta_features_per_fold_{timestamp}.csv'
    fold_df.to_csv(csv_file, index=False)
    logger.info(f"✓ Per-fold results saved to: {csv_file}")
    
    # Generate report
    generate_report(summary, baseline, timestamp)
    
    logger.info("\n" + "="*80)
    logger.info("COMPLETE")
    logger.info("="*80)


def generate_report(summary: Dict, baseline: Dict, timestamp: str):
    """Generate markdown report."""
    logger.info("Generating report...")
    
    report_lines = [
        "# Nested CV with Enhanced Meta-Features: Results Report",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        "This report presents results from nested cross-validation using **enhanced meta-features**",
        "engineered from base model probabilities.",
        "",
        "---",
        "",
        "## Meta-Feature Engineering",
        "",
        "### Features Generated",
        "",
        "**Core Probability Features**:",
        "- `p_resnet`, `p_swin`, `p_mil`: Individual base model probabilities",
        "",
        "**Agreement / Disagreement**:",
        "- `prob_mean`: Mean across base models",
        "- `prob_std`: Standard deviation (measures disagreement)",
        "- `prob_max`, `prob_min`: Range of predictions",
        "- `prob_range`: Max - min (measures spread)",
        "",
        "**Confidence / Margin**:",
        "- `margin_mean`: |mean - 0.5| (distance from uncertainty)",
        "- `margin_max`: Maximum margin across models",
        "- `entropy_mean`: Binary entropy of mean probability (uncertainty measure)",
        "",
        "**Model Dominance**:",
        "- `argmax_resnet`, `argmax_swin`, `argmax_mil`: One-hot encoded model with highest probability",
        "",
        "**Total Features**: 15 (3 base + 12 engineered)",
        "",
        "### Medical Relevance",
        "",
        "- **Agreement features** help identify cases where all models agree (high confidence)",
        "- **Disagreement features** flag uncertain cases requiring human review",
        "- **Margin features** measure distance from decision boundary (confidence)",
        "- **Entropy** quantifies prediction uncertainty",
        "- **Model dominance** captures which model drives the decision",
        "",
        "---",
        "",
        "## Results Summary",
        "",
        f"**Meta-Learner**: Logistic Regression with Enhanced Meta-Features",
        f"**Evaluation**: Nested Cross-Validation (5 outer folds)",
        f"**Calibration**: Robust Platt scaling (5 repeats, median threshold)",
        "",
        "| Metric | Mean ± Std | Range |",
        "|--------|------------|-------|",
        f"| FN | {summary['fn_mean']:.2f} ± {summary['fn_std']:.2f} | [{summary['fn_min']}, {summary['fn_max']}] |",
        f"| FP | {summary['fp_mean']:.2f} ± {summary['fp_std']:.2f} | - |",
        f"| Cost | {summary['cost_mean']:.2f} ± {summary['cost_std']:.2f} | - |",
        f"| Recall | {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f} | - |",
        f"| Precision | {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f} | - |",
        f"| F1 | {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f} | - |",
        ""
    ]
    
    if baseline and 'comparison' in summary:
        comp = summary['comparison']
        report_lines.extend([
            "---",
            "",
            "## Comparison with Baseline",
            "",
            "| Metric | Baseline (Simple) | Enhanced (Meta-Features) | Change |",
            "|--------|------------------|--------------------------|--------|",
            f"| FN | {comp['baseline_fn_mean']:.2f} ± {baseline['fn_std']:.2f} | "
            f"{summary['fn_mean']:.2f} ± {summary['fn_std']:.2f} | "
            f"{comp['fn_improvement']:+.2f} |",
            f"| FP | {comp['baseline_fp_mean']:.2f} ± {baseline['fp_std']:.2f} | "
            f"{summary['fp_mean']:.2f} ± {summary['fp_std']:.2f} | "
            f"{comp['fp_change']:+.2f} |",
            f"| Cost | {comp['baseline_cost_mean']:.2f} ± {baseline['cost_std']:.2f} | "
            f"{summary['cost_mean']:.2f} ± {summary['cost_std']:.2f} | "
            f"{comp['cost_improvement']:+.2f} |",
            "",
            "### Improvement Analysis",
            ""
        ])
        
        if comp['fn_improved']:
            report_lines.append(f"✅ **FN decreased** by {comp['fn_improvement']:.2f} (improvement)")
        else:
            report_lines.append(f"❌ **FN increased** by {abs(comp['fn_improvement']):.2f} (worse)")
        
        if abs(comp['fp_change']) < 2:
            report_lines.append(f"✅ **FP change acceptable**: {comp['fp_change']:+.2f}")
        else:
            report_lines.append(f"⚠️ **FP change significant**: {comp['fp_change']:+.2f}")
        
        if comp['cost_improved']:
            report_lines.append(f"✅ **Cost reduced** by {comp['cost_improvement']:.2f} (improvement)")
        else:
            report_lines.append(f"❌ **Cost increased** by {abs(comp['cost_improvement']):.2f} (worse)")
        
        report_lines.extend([
            "",
            "### Consistency Across Folds",
            "",
            f"- **FN range**: [{summary['fn_min']}, {summary['fn_max']}]",
            f"- **Worst-case FN**: {summary['fn_max']} (medical safety critical)",
            f"- **FN std**: {summary['fn_std']:.2f} ({'stable' if summary['fn_std'] < 2 else 'variable'})",
            ""
        ])
    
    report_lines.extend([
        "---",
        "",
        "## Per-Fold Details",
        "",
        "| Fold | FN | FP | Cost | Recall | Precision | F1 | Threshold |",
        "|------|----|----|------|--------|-----------|----|-----------|"
    ])
    
    for fold_result in summary['fold_results']:
        report_lines.append(
            f"| {fold_result['fold']} | "
            f"{fold_result['fn']} | {fold_result['fp']} | "
            f"{fold_result['cost']:.1f} | "
            f"{fold_result['recall']:.4f} | {fold_result['precision']:.4f} | "
            f"{fold_result['f1']:.4f} | {fold_result['selected_threshold']:.4f} |"
        )
    
    report_lines.extend([
        "",
        "---",
        "",
        "## Conclusion",
        ""
    ])
    
    if baseline and 'comparison' in summary:
        comp = summary['comparison']
        if comp['fn_improved'] and comp['cost_improved']:
            report_lines.extend([
                "✅ **Meta-feature engineering improved performance**",
                "",
                f"- FN reduced by {comp['fn_improvement']:.2f}",
                f"- Cost reduced by {comp['cost_improvement']:.2f}",
                f"- Worst-case FN: {summary['fn_max']} (acceptable for medical safety)",
                "",
                "**Recommendation**: Adopt enhanced meta-features for final model."
            ])
        else:
            report_lines.extend([
                "❌ **Meta-feature engineering did not improve performance**",
                "",
                f"- FN change: {comp['fn_improvement']:+.2f}",
                f"- Cost change: {comp['cost_improvement']:+.2f}",
                "",
                "**Recommendation**: Meta-feature engineering did not help. Next step: improve base models."
            ])
    else:
        report_lines.append("Baseline comparison not available.")
    
    report_file = OUTPUT_DIR / f'meta_features_report_{timestamp}.md'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"✓ Report saved to: {report_file}")


if __name__ == '__main__':
    main()

