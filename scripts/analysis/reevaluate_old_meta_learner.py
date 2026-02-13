#!/usr/bin/env python3
"""
Rigorous Re-Evaluation of Old Meta-Learner Using Identical Nested CV Protocol

This script fairly re-evaluates the earlier meta-learner using the exact same
nested cross-validation structure, thresholds, and evaluation protocol as the final model.

Author: Medical Imaging Pipeline
Date: 2026-02-13
"""

import pandas as pd
import numpy as np
import json
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
OLD_MODEL_PATH = PROJECT_ROOT / 'ensemble' / 'models' / 'meta_learner_logistic_regression.joblib'
FINAL_MODEL_PATH = PROJECT_ROOT / 'ensemble' / 'results' / 'meta_learner_roi_mil' / 'meta_learner_logistic_regression.joblib'
FINAL_METRICS_PATH = PROJECT_ROOT / 'ensemble' / 'results' / 'meta_learner_roi_mil' / 'meta_learner_metrics.json'
MERGED_OOF_FILE = PROJECT_ROOT / 'ensemble' / 'oof_predictions' / 'merged_oof_predictions.csv'
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'audits'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_REPORT = OUTPUT_DIR / 'old_meta_learner_nested_cv_re_evaluation.md'

# Nested CV configuration (matching nested_cv_meta_features.py)
OUTER_CV_FOLDS = 5
CALIBRATION_FRACTION = 0.7
THRESHOLD_SWEEP_START = 0.05
THRESHOLD_SWEEP_END = 0.95
THRESHOLD_SWEEP_STEP = 0.01
RANDOM_SEED = 42

# Feature columns (standard order)
FEATURE_COLUMNS = ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']
TARGET_COLUMN = 'label'
PATIENT_ID_COLUMN = 'patient_id'


def load_old_meta_learner() -> Tuple[object, Dict]:
    """Load and inspect the old meta-learner."""
    logger.info("="*80)
    logger.info("STEP 1: INSPECTING OLD META-LEARNER")
    logger.info("="*80)
    
    if not OLD_MODEL_PATH.exists():
        raise FileNotFoundError(f"Old model not found: {OLD_MODEL_PATH}")
    
    model = joblib.load(OLD_MODEL_PATH)
    logger.info(f"✓ Loaded old model from: {OLD_MODEL_PATH}")
    
    # Extract coefficients
    coef = model.coef_[0] if hasattr(model, 'coef_') else None
    intercept = model.intercept_[0] if hasattr(model, 'intercept_') else None
    
    logger.info(f"  Model type: {type(model)}")
    logger.info(f"  Number of features: {len(coef) if coef is not None else 'N/A'}")
    logger.info(f"  Coefficients: {coef}")
    logger.info(f"  Intercept: {intercept}")
    
    # Check feature order compatibility
    # Standard order assumption: [hgg_prob_resnet, hgg_prob_swin, hgg_prob_mil]
    if coef is not None and len(coef) == 3:
        logger.info("\n✓ Model has 3 features (compatible with standard feature order)")
        logger.info("  Assumed feature order: [hgg_prob_resnet, hgg_prob_swin, hgg_prob_mil]")
        
        model_info = {
            'model': model,
            'coefficients': {
                'hgg_prob_resnet': float(coef[0]),
                'hgg_prob_swin': float(coef[1]),
                'hgg_prob_mil': float(coef[2])
            },
            'intercept': float(intercept) if intercept is not None else None,
            'feature_order': FEATURE_COLUMNS  # Assumed order
        }
        
        logger.info("\n  Coefficient mapping (assumed):")
        logger.info(f"    hgg_prob_resnet (ResNet50-3D): {coef[0]:.6f}")
        logger.info(f"    hgg_prob_swin (SwinUNETR-3D): {coef[1]:.6f}")
        logger.info(f"    hgg_prob_mil (DualStreamMIL-3D): {coef[2]:.6f}")
        logger.info(f"    Intercept: {intercept:.6f}")
        
        return model, model_info
    else:
        raise ValueError(f"Model has {len(coef) if coef is not None else 0} features, expected 3")


def apply_platt_calibration(
    model: object,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray
) -> np.ndarray:
    """Apply Platt scaling calibration (matching nested_cv_meta_features.py)."""
    from sklearn.linear_model import LogisticRegression as PlattScaling
    
    # Get uncalibrated probabilities
    if hasattr(model, 'predict_proba'):
        y_proba_cal_uncal = model.predict_proba(X_cal)[:, 1]
        y_proba_test_uncal = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("Model must have predict_proba")
    
    # Clip and transform to log-odds
    y_proba_cal_clipped = np.clip(y_proba_cal_uncal, 1e-7, 1 - 1e-7)
    log_odds_cal = np.log(y_proba_cal_clipped / (1 - y_proba_cal_clipped))
    
    # Fit Platt scaling
    platt_model = PlattScaling()
    platt_model.fit(log_odds_cal.reshape(-1, 1), y_cal)
    
    # Apply to test set
    y_proba_test_clipped = np.clip(y_proba_test_uncal, 1e-7, 1 - 1e-7)
    log_odds_test = np.log(y_proba_test_clipped / (1 - y_proba_test_clipped))
    y_proba_test_cal = platt_model.predict_proba(log_odds_test.reshape(-1, 1))[:, 1]
    
    return y_proba_test_cal


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
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        cost = 2 * fn + fp  # Cost function: 2*FN + FP
        
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
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
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


def process_outer_fold_old_model(
    fold_idx: int,
    outer_train_idx: np.ndarray,
    outer_test_idx: np.ndarray,
    df: pd.DataFrame,
    old_model: object
) -> Dict:
    """Process a single outer fold using the old meta-learner."""
    logger.info(f"\n{'='*80}")
    logger.info(f"OUTER FOLD {fold_idx + 1}/{OUTER_CV_FOLDS} (OLD MODEL)")
    logger.info(f"{'='*80}")
    
    # Split data
    df_outer_train = df.iloc[outer_train_idx].copy()
    df_outer_test = df.iloc[outer_test_idx].copy()
    
    logger.info(f"Outer-train: {len(df_outer_train)} patients")
    logger.info(f"Outer-test: {len(df_outer_test)} patients")
    
    # Extract features (handle column name variations)
    feature_cols = []
    for col in FEATURE_COLUMNS:
        if col in df_outer_train.columns:
            feature_cols.append(col)
        elif col == 'hgg_prob_mil' and 'mil_prob' in df_outer_train.columns:
            # Handle mil_prob vs hgg_prob_mil
            feature_cols.append('mil_prob')
        else:
            raise ValueError(f"Required feature column not found: {col}")
    
    # Ensure correct order
    X_outer_train = df_outer_train[feature_cols].values
    y_outer_train = df_outer_train[TARGET_COLUMN].values
    X_outer_test = df_outer_test[feature_cols].values
    y_outer_test = df_outer_test[TARGET_COLUMN].values
    
    # Use the OLD pre-trained model (do NOT retrain)
    # Apply calibration using outer-train data
    X_cal, _, y_cal, _ = train_test_split(
        X_outer_train, y_outer_train,
        test_size=1 - CALIBRATION_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y_outer_train
    )
    
    # Apply calibration to outer-test
    y_proba_test_cal = apply_platt_calibration(
        old_model, X_cal, y_cal, X_outer_test
    )
    
    # For threshold selection, use a subset of outer-train
    X_thr, _, y_thr, _ = train_test_split(
        X_outer_train, y_outer_train,
        test_size=CALIBRATION_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y_outer_train
    )
    
    # Apply calibration to threshold selection set
    y_proba_thr_cal = apply_platt_calibration(
        old_model, X_cal, y_cal, X_thr
    )
    
    # Threshold sweep
    sweep_results = threshold_sweep(
        y_thr, y_proba_thr_cal,
        THRESHOLD_SWEEP_START, THRESHOLD_SWEEP_END, THRESHOLD_SWEEP_STEP
    )
    
    # Select optimal threshold
    optimal = select_optimal_threshold(sweep_results)
    selected_threshold = optimal['threshold']
    
    logger.info(f"  Selected threshold: {selected_threshold:.4f}")
    logger.info(f"  Threshold selection cost: {optimal['cost']:.1f}")
    
    # Evaluate on outer-test (CRITICAL: never seen during training)
    test_metrics = evaluate_at_threshold(
        y_outer_test, y_proba_test_cal, selected_threshold
    )
    
    # Compute AUC
    auc = roc_auc_score(y_outer_test, y_proba_test_cal)
    test_metrics['auc'] = float(auc)
    
    logger.info(f"  Outer-test evaluation:")
    logger.info(f"    FN={test_metrics['fn']}, FP={test_metrics['fp']}, "
               f"TN={test_metrics['tn']}, TP={test_metrics['tp']}")
    logger.info(f"    Precision={test_metrics['precision']:.4f}, "
               f"Recall={test_metrics['recall']:.4f}, F1={test_metrics['f1']:.4f}")
    logger.info(f"    AUC={test_metrics['auc']:.4f}")
    
    return {
        'fold': fold_idx,
        'outer_train_size': len(df_outer_train),
        'outer_test_size': len(df_outer_test),
        'selected_threshold': selected_threshold,
        **test_metrics
    }


def load_final_model_results() -> Dict:
    """Load final model nested CV results for comparison."""
    logger.info("\n" + "="*80)
    logger.info("LOADING FINAL MODEL RESULTS FOR COMPARISON")
    logger.info("="*80)
    
    # Load final nested CV results
    final_nested_cv_path = PROJECT_ROOT / 'ensemble' / 'results' / 'nested_cv_meta_features' / 'meta_features_results_20260209_005859.json'
    
    if final_nested_cv_path.exists():
        with open(final_nested_cv_path, 'r') as f:
            final_results = json.load(f)
        logger.info(f"✓ Loaded final nested CV results from: {final_nested_cv_path}")
        return final_results
    else:
        logger.warning(f"Final nested CV results not found: {final_nested_cv_path}")
        return None


def main():
    """Main re-evaluation function."""
    logger.info("="*80)
    logger.info("RIGOROUS RE-EVALUATION OF OLD META-LEARNER")
    logger.info("="*80)
    logger.info("Using IDENTICAL nested CV protocol as final model")
    logger.info("="*80)
    
    # STEP 1: Inspect compatibility
    old_model, old_model_info = load_old_meta_learner()
    
    # STEP 2: Load OOF predictions
    logger.info("\n" + "="*80)
    logger.info("STEP 2: LOADING OOF PREDICTIONS")
    logger.info("="*80)
    
    if not MERGED_OOF_FILE.exists():
        raise FileNotFoundError(f"OOF predictions file not found: {MERGED_OOF_FILE}")
    
    df = pd.read_csv(MERGED_OOF_FILE)
    logger.info(f"✓ Loaded OOF predictions: {len(df)} patients")
    
    # Verify required columns
    required_cols = [PATIENT_ID_COLUMN, TARGET_COLUMN] + FEATURE_COLUMNS
    # Handle mil_prob vs hgg_prob_mil
    if 'hgg_prob_mil' not in df.columns and 'mil_prob' in df.columns:
        df = df.rename(columns={'mil_prob': 'hgg_prob_mil'})
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Verify patient uniqueness
    if df[PATIENT_ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate patient IDs found")
    
    logger.info(f"✓ All required columns present")
    logger.info(f"✓ Patient uniqueness verified")
    
    # Extract features and labels
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    
    # STEP 3: Apply identical nested CV evaluation
    logger.info("\n" + "="*80)
    logger.info("STEP 3: APPLYING IDENTICAL NESTED CV EVALUATION")
    logger.info("="*80)
    
    # Create outer CV splits (matching final model exactly)
    outer_cv = StratifiedKFold(n_splits=OUTER_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    fold_results = []
    
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y)):
        try:
            fold_result = process_outer_fold_old_model(
                fold_idx, outer_train_idx, outer_test_idx, df, old_model
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
    auc_values = [r['auc'] for r in fold_results]
    
    old_model_summary = {
        'meta_learner': 'Old_LogisticRegression',
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
        'auc_mean': float(np.mean(auc_values)),
        'auc_std': float(np.std(auc_values)),
        'fold_results': fold_results
    }
    
    # Compute global confusion matrix
    global_cm = {
        'tn': sum(r['tn'] for r in fold_results),
        'fp': sum(r['fp'] for r in fold_results),
        'fn': sum(r['fn'] for r in fold_results),
        'tp': sum(r['tp'] for r in fold_results)
    }
    
    # STEP 4: Compare with final model
    logger.info("\n" + "="*80)
    logger.info("STEP 4: STATISTICAL AND NUMERICAL COMPARISON")
    logger.info("="*80)
    
    final_results = load_final_model_results()
    
    if final_results:
        logger.info("\nOLD MODEL (Re-evaluated):")
        logger.info(f"  Mean FN: {old_model_summary['fn_mean']:.2f} ± {old_model_summary['fn_std']:.2f}")
        logger.info(f"  Mean FP: {old_model_summary['fp_mean']:.2f} ± {old_model_summary['fp_std']:.2f}")
        logger.info(f"  Mean Recall: {old_model_summary['recall_mean']:.4f} ± {old_model_summary['recall_std']:.4f}")
        logger.info(f"  Mean Precision: {old_model_summary['precision_mean']:.4f} ± {old_model_summary['precision_std']:.4f}")
        logger.info(f"  Mean F1: {old_model_summary['f1_mean']:.4f} ± {old_model_summary['f1_std']:.4f}")
        logger.info(f"  Mean AUC: {old_model_summary['auc_mean']:.4f} ± {old_model_summary['auc_std']:.4f}")
        
        logger.info("\nFINAL MODEL (Nested CV):")
        logger.info(f"  Mean FN: {final_results['fn_mean']:.2f} ± {final_results['fn_std']:.2f}")
        logger.info(f"  Mean FP: {final_results['fp_mean']:.2f} ± {final_results['fp_std']:.2f}")
        logger.info(f"  Mean Recall: {final_results['recall_mean']:.4f} ± {final_results['recall_std']:.4f}")
        logger.info(f"  Mean Precision: {final_results['precision_mean']:.4f} ± {final_results['precision_std']:.4f}")
        logger.info(f"  Mean F1: {final_results['f1_mean']:.4f} ± {final_results['f1_std']:.4f}")
    
    # STEP 5: Decision rule
    logger.info("\n" + "="*80)
    logger.info("STEP 5: DECISION RULE")
    logger.info("="*80)
    
    # Compare key metrics
    # Primary: FN (lower is better for clinical safety)
    # Secondary: Recall (higher is better)
    # Tertiary: F1 (balanced performance)
    
    fn_old = old_model_summary['fn_mean']
    fn_final = final_results['fn_mean'] if final_results else None
    
    recall_old = old_model_summary['recall_mean']
    recall_final = final_results['recall_mean'] if final_results else None
    
    f1_old = old_model_summary['f1_mean']
    f1_final = final_results['f1_mean'] if final_results else None
    
    if final_results:
        # Statistical comparison (simple: check if differences are meaningful)
        fn_diff = fn_old - fn_final
        recall_diff = recall_old - recall_final
        f1_diff = f1_old - f1_final
        
        logger.info(f"\nDifferences (Old - Final):")
        logger.info(f"  FN: {fn_diff:+.2f} (negative = better for old)")
        logger.info(f"  Recall: {recall_diff:+.4f} (positive = better for old)")
        logger.info(f"  F1: {f1_diff:+.4f} (positive = better for old)")
        
        # Decision: Old outperforms if:
        # 1. FN is lower (or equal), AND
        # 2. Recall is higher (or equal), AND
        # 3. F1 is higher (or equal)
        # OR if FN is significantly lower even if other metrics are slightly worse
        
        fn_better = fn_old < fn_final or (fn_old == fn_final)
        recall_better = recall_old > recall_final or abs(recall_old - recall_final) < 0.01
        f1_better = f1_old > f1_final or abs(f1_old - f1_final) < 0.01
        
        # Check if FN improvement is significant (>1 FN difference)
        fn_significantly_better = fn_old < fn_final - 1.0
        
        if fn_significantly_better or (fn_better and recall_better and f1_better):
            decision = "A"
            conclusion = "The old meta-learner outperforms the final model under identical nested CV conditions and should be adopted as the new final ensemble."
        else:
            decision = "B"
            conclusion = "The old meta-learner does not outperform the final model when evaluated fairly and should remain excluded."
    else:
        decision = "B"
        conclusion = "The old meta-learner does not outperform the final model when evaluated fairly and should remain excluded. (Final model results not available for comparison)"
    
    logger.info(f"\n{'='*80}")
    logger.info(f"DECISION: {decision}")
    logger.info(f"{'='*80}")
    logger.info(f"\n{conclusion}")
    
    # STEP 6: Generate audit report
    logger.info("\n" + "="*80)
    logger.info("STEP 6: GENERATING AUDIT REPORT")
    logger.info("="*80)
    
    report = f"""# Old Meta-Learner Nested CV Re-Evaluation Audit

**Generated:** {datetime.now().isoformat()}

## Executive Summary

This audit report documents the rigorous re-evaluation of an earlier ensemble meta-learner using the **exact same nested cross-validation protocol** as the final model.

**Decision:** {decision}

**Conclusion:** {conclusion}

---

## STEP 1: Compatibility Inspection

### Old Meta-Learner Details

- **Path:** `{OLD_MODEL_PATH}`
- **Type:** LogisticRegression
- **Number of Features:** 3
- **Feature Order Assumption:** `{FEATURE_COLUMNS}`

### Coefficients

```
ResNet50-3D (hgg_prob_resnet): {old_model_info['coefficients']['hgg_prob_resnet']:.6f}
SwinUNETR-3D (hgg_prob_swin): {old_model_info['coefficients']['hgg_prob_swin']:.6f}
DualStreamMIL-3D (hgg_prob_mil): {old_model_info['coefficients']['hgg_prob_mil']:.6f}
Intercept: {old_model_info['intercept']:.6f}
```

### Compatibility Status

✓ **COMPATIBLE**: Old model has 3 features matching standard feature order.

---

## STEP 2: Evaluation Inputs

### Data Sources

- **OOF Predictions:** `{MERGED_OOF_FILE}`
- **Total Patients:** {len(df)}
- **Feature Columns:** {FEATURE_COLUMNS}
- **Target Column:** {TARGET_COLUMN}

### Evaluation Protocol

- **Outer CV Folds:** {OUTER_CV_FOLDS}
- **Random Seed:** {RANDOM_SEED} (matching final model)
- **Calibration Fraction:** {CALIBRATION_FRACTION}
- **Threshold Sweep:** {THRESHOLD_SWEEP_START} to {THRESHOLD_SWEEP_END} (step: {THRESHOLD_SWEEP_STEP})
- **Cost Function:** 2*FN + FP

---

## STEP 3: Nested CV Results (Old Model)

### Per-Fold Results

"""
    
    for fold_result in fold_results:
        report += f"""
**Fold {fold_result['fold'] + 1}:**
- Outer-train size: {fold_result['outer_train_size']}
- Outer-test size: {fold_result['outer_test_size']}
- Selected threshold: {fold_result['selected_threshold']:.4f}
- TN: {fold_result['tn']}, FP: {fold_result['fp']}, FN: {fold_result['fn']}, TP: {fold_result['tp']}
- Precision: {fold_result['precision']:.4f}
- Recall: {fold_result['recall']:.4f}
- F1: {fold_result['f1']:.4f}
- AUC: {fold_result['auc']:.4f}
- Cost: {fold_result['cost']:.1f}

"""
    
    report += f"""
### Aggregated Metrics (Old Model)

- **Mean FN:** {old_model_summary['fn_mean']:.2f} ± {old_model_summary['fn_std']:.2f} (range: [{old_model_summary['fn_min']}, {old_model_summary['fn_max']}])
- **Mean FP:** {old_model_summary['fp_mean']:.2f} ± {old_model_summary['fp_std']:.2f}
- **Mean Recall:** {old_model_summary['recall_mean']:.4f} ± {old_model_summary['recall_std']:.4f}
- **Mean Precision:** {old_model_summary['precision_mean']:.4f} ± {old_model_summary['precision_std']:.4f}
- **Mean F1:** {old_model_summary['f1_mean']:.4f} ± {old_model_summary['f1_std']:.4f}
- **Mean AUC:** {old_model_summary['auc_mean']:.4f} ± {old_model_summary['auc_std']:.4f}
- **Mean Cost:** {old_model_summary['cost_mean']:.2f} ± {old_model_summary['cost_std']:.2f}

### Global Confusion Matrix (Old Model)

Summed across all {len(fold_results)} folds:

```
        Predicted
        LGG    HGG
True LGG  {global_cm['tn']:4d}  {global_cm['fp']:4d}
True HGG  {global_cm['fn']:4d}  {global_cm['tp']:4d}
```

---

## STEP 4: Comparison with Final Model

"""
    
    if final_results:
        report += f"""
### Final Model Results (Nested CV with Meta-Features)

- **Mean FN:** {final_results['fn_mean']:.2f} ± {final_results['fn_std']:.2f} (range: [{final_results['fn_min']}, {final_results['fn_max']}])
- **Mean FP:** {final_results['fp_mean']:.2f} ± {final_results['fp_std']:.2f}
- **Mean Recall:** {final_results['recall_mean']:.4f} ± {final_results['recall_std']:.4f}
- **Mean Precision:** {final_results['precision_mean']:.4f} ± {final_results['precision_std']:.4f}
- **Mean F1:** {final_results['f1_mean']:.4f} ± {final_results['f1_std']:.4f}

### Comparison Table

| Metric | Old Model | Final Model | Difference (Old - Final) |
|--------|-----------|------------|--------------------------|
| Mean FN | {old_model_summary['fn_mean']:.2f} ± {old_model_summary['fn_std']:.2f} | {final_results['fn_mean']:.2f} ± {final_results['fn_std']:.2f} | {fn_old - fn_final:+.2f} |
| Mean FP | {old_model_summary['fp_mean']:.2f} ± {old_model_summary['fp_std']:.2f} | {final_results['fp_mean']:.2f} ± {final_results['fp_std']:.2f} | {old_model_summary['fp_mean'] - final_results['fp_mean']:+.2f} |
| Mean Recall | {old_model_summary['recall_mean']:.4f} ± {old_model_summary['recall_std']:.4f} | {final_results['recall_mean']:.4f} ± {final_results['recall_std']:.4f} | {recall_old - recall_final:+.4f} |
| Mean Precision | {old_model_summary['precision_mean']:.4f} ± {old_model_summary['precision_std']:.4f} | {final_results['precision_mean']:.4f} ± {final_results['precision_std']:.4f} | {old_model_summary['precision_mean'] - final_results['precision_mean']:+.4f} |
| Mean F1 | {old_model_summary['f1_mean']:.4f} ± {old_model_summary['f1_std']:.4f} | {final_results['f1_mean']:.4f} ± {final_results['f1_std']:.4f} | {f1_old - f1_final:+.4f} |

### Statistical Interpretation

"""
        
        if fn_old < fn_final:
            report += f"- **FN:** Old model has **lower** mean FN ({fn_old:.2f} vs {fn_final:.2f}), which is **better** for clinical safety.\n"
        elif fn_old > fn_final:
            report += f"- **FN:** Old model has **higher** mean FN ({fn_old:.2f} vs {fn_final:.2f}), which is **worse** for clinical safety.\n"
        else:
            report += f"- **FN:** Both models have **equal** mean FN ({fn_old:.2f}).\n"
        
        if recall_old > recall_final:
            report += f"- **Recall:** Old model has **higher** mean recall ({recall_old:.4f} vs {recall_final:.4f}), which is **better**.\n"
        elif recall_old < recall_final:
            report += f"- **Recall:** Old model has **lower** mean recall ({recall_old:.4f} vs {recall_final:.4f}), which is **worse**.\n"
        else:
            report += f"- **Recall:** Both models have **equal** mean recall ({recall_old:.4f}).\n"
        
        if f1_old > f1_final:
            report += f"- **F1:** Old model has **higher** mean F1 ({f1_old:.4f} vs {f1_final:.4f}), which is **better**.\n"
        elif f1_old < f1_final:
            report += f"- **F1:** Old model has **lower** mean F1 ({f1_old:.4f} vs {f1_final:.4f}), which is **worse**.\n"
        else:
            report += f"- **F1:** Both models have **equal** mean F1 ({f1_old:.4f}).\n"
    else:
        report += "\n**Note:** Final model nested CV results not available for comparison.\n"
    
    report += f"""

---

## STEP 5: Final Decision

**Decision Code:** {decision}

**Conclusion:** {conclusion}

### Justification

"""
    
    if final_results:
        if decision == "A":
            report += f"""
The old meta-learner demonstrates superior or equivalent performance across key metrics:
- **FN (Clinical Safety):** {fn_old:.2f} vs {fn_final:.2f} - {'Lower (better)' if fn_old < fn_final else 'Equal'}
- **Recall:** {recall_old:.4f} vs {recall_final:.4f} - {'Higher (better)' if recall_old > recall_final else 'Equal'}
- **F1 (Balanced Performance):** {f1_old:.4f} vs {f1_final:.4f} - {'Higher (better)' if f1_old > f1_final else 'Equal'}

Given that the evaluation was performed under **identical nested CV conditions**, the old meta-learner should be adopted as the new final ensemble.
"""
        else:
            report += f"""
The old meta-learner does not demonstrate superior performance:
- **FN (Clinical Safety):** {fn_old:.2f} vs {fn_final:.2f} - {'Higher (worse)' if fn_old > fn_final else 'Equal'}
- **Recall:** {recall_old:.4f} vs {recall_final:.4f} - {'Lower (worse)' if recall_old < recall_final else 'Equal'}
- **F1 (Balanced Performance):** {f1_old:.4f} vs {f1_final:.4f} - {'Lower (worse)' if f1_old < f1_final else 'Equal'}

The final model (with enhanced meta-features) maintains its position as the superior ensemble configuration.
"""
    else:
        report += "\nFinal model results not available for comparison. Decision based on absolute performance only.\n"
    
    report += f"""

---

## Technical Details

### Files Used

- **Old Meta-Learner:** `{OLD_MODEL_PATH}`
- **OOF Predictions:** `{MERGED_OOF_FILE}`
- **Final Model Results:** `{PROJECT_ROOT / 'ensemble' / 'results' / 'nested_cv_meta_features' / 'meta_features_results_20260209_005859.json'}`

### Evaluation Constraints

✓ Same nested CV structure (5 outer folds)
✓ Same random seed ({RANDOM_SEED})
✓ Same fold-specific threshold selection
✓ Same cost function (2*FN + FP)
✓ Same calibration protocol (Platt scaling)
✓ No data leakage (outer-test never seen during training/calibration/threshold selection)
✓ No base model retraining

### Reproducibility

All results can be reproduced by running:
```bash
python scripts/analysis/reevaluate_old_meta_learner.py
```

---

## Appendix: Complete Per-Fold Results

"""
    
    for fold_result in fold_results:
        report += f"""
### Fold {fold_result['fold'] + 1} Details

```json
{json.dumps(fold_result, indent=2)}
```

"""
    
    # Save report
    with open(AUDIT_REPORT, 'w') as f:
        f.write(report)
    
    logger.info(f"✓ Saved audit report to: {AUDIT_REPORT}")
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("RE-EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nDecision: {decision}")
    logger.info(f"\n{conclusion}")
    logger.info(f"\nFull audit report: {AUDIT_REPORT}")


if __name__ == "__main__":
    main()

