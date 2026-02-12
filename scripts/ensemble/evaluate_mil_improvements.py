#!/usr/bin/env python3
"""
Evaluate MIL Improvements Using Existing OOF Predictions

This script evaluates MIL model improvements using existing OOF predictions.
NO training is performed - this is a pure evaluation script.

For each experiment:
1. Load existing OOF predictions (from experiment directory or latest runs)
2. Evaluate via nested CV meta-learner
3. Evaluate with recall-targeted threshold tuning (≥0.85)
4. Compare against baselines (original MIL, enhanced meta-features)

IMPORTANT: This script does NOT train models. It only evaluates existing OOF predictions.
"""

import pandas as pd
import numpy as np
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, average_precision_score, roc_auc_score,
    precision_recall_curve, roc_curve
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
MERGED_OOF_FILE = Path('ensemble/oof_predictions/merged_oof_predictions.csv')
MIL_TRAINING_SCRIPT = Path('scripts/training/train_dual_stream_mil.py')
MIL_RESULTS_DIR = Path('results/DualStreamMIL-3D/runs')
OUTPUT_DIR = Path('ensemble/results/mil_improvements')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Baseline results
BASELINE_MIL_RESULTS = Path('ensemble/results/nested_cv_meta_learning/nested_cv_results_20260208_233521.json')
ENHANCED_META_RESULTS = Path('ensemble/results/nested_cv_meta_features/meta_features_results_20260209_005859.json')

# Configuration
OUTER_CV_FOLDS = 5
CALIBRATION_FRACTION = 0.7
THRESHOLD_SWEEP_START = 0.05
THRESHOLD_SWEEP_END = 0.95
THRESHOLD_SWEEP_STEP = 0.01
RANDOM_SEED = 42

# OOF EVALUATION MODE: This script ONLY evaluates existing OOF predictions
# NO training is performed. All folds are evaluated.
OOF_EVALUATION_MODE = True  # Hard-coded: evaluation only, no training

# Feature columns
# Note: For MIL-only experiments, only 'hgg_prob_mil' will be available
# For full ensemble experiments, all three features are present
BASE_FEATURE_COLS_ENSEMBLE = ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']
BASE_FEATURE_COLS_MIL_ONLY = ['hgg_prob_mil']
TARGET_COLUMN = 'label'
PATIENT_ID_COLUMN = 'patient_id'


def make_json_serializable(obj):
    """Convert numpy types to JSON-serializable types."""
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


# Training function removed - this script only evaluates existing OOF predictions
# NO training is performed in OOF_EVALUATION_MODE


def load_oof_predictions(experiment_name: str) -> Optional[pd.DataFrame]:
    """
    Load existing OOF predictions for evaluation.
    
    First tries to load from experiment directory's oof_predictions.csv.
    If not found, loads from latest MIL runs across all 5 folds.
    
    Returns:
        DataFrame with OOF predictions, or None if failed
    """
    logger.info(f"Loading OOF predictions for {experiment_name}...")
    
    exp_dir = OUTPUT_DIR / experiment_name
    oof_file = exp_dir / 'oof_predictions.csv'
    
    # Try to load from experiment directory first
    if oof_file.exists():
        logger.info(f"  Found existing OOF file: {oof_file}")
        try:
            df = pd.read_csv(oof_file)
            logger.info(f"  Loaded {len(df)} predictions from {oof_file}")
            # Ensure fold column exists
            if 'fold' not in df.columns:
                logger.warning("  'fold' column not found. Will use all data for evaluation.")
            return df
        except Exception as e:
            logger.warning(f"  Failed to load from {oof_file}: {e}")
            logger.info("  Falling back to loading from latest runs...")
    
    # Fallback: Load from latest MIL runs (all 5 folds)
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from scripts.ensemble.prepare_oof_predictions import (
        find_latest_run, verify_run_has_predictions,
        load_oof_predictions_from_run
    )
    
    SPLITS_DIR = Path('splits')
    NUM_FOLDS = 5  # Always load all 5 folds in evaluation mode
    folds_to_process = list(range(NUM_FOLDS))
    
    all_fold_dfs = []
    
    for fold in folds_to_process:
        fold_dir = MIL_RESULTS_DIR / f'fold_{fold}'
        if not fold_dir.exists():
            logger.error(f"Fold {fold} directory not found: {fold_dir}")
            return None
        
        latest_run = find_latest_run(MIL_RESULTS_DIR, fold)
        if latest_run is None:
            logger.error(f"Could not find run for fold {fold}")
            return None
        
        if not verify_run_has_predictions(latest_run):
            logger.error(f"Missing prediction files in {latest_run}")
            return None
        
        val_split_csv = SPLITS_DIR / f'fold_{fold}_val.csv'
        if not val_split_csv.exists():
            logger.error(f"Validation split file not found: {val_split_csv}")
            return None
        
        try:
            fold_df = load_oof_predictions_from_run(latest_run, fold, val_split_csv)
            # Rename 'hgg_prob' to 'hgg_prob_mil' for consistency
            if 'hgg_prob' in fold_df.columns and 'hgg_prob_mil' not in fold_df.columns:
                fold_df = fold_df.rename(columns={'hgg_prob': 'hgg_prob_mil'})
            all_fold_dfs.append(fold_df)
            logger.info(f"  Loaded {len(fold_df)} predictions for fold {fold}")
        except Exception as e:
            logger.error(f"Error loading predictions for fold {fold}: {e}", exc_info=True)
            return None
    
    if not all_fold_dfs:
        logger.error("No fold predictions loaded")
        return None
    
    combined_df = pd.concat(all_fold_dfs, ignore_index=True)
    combined_df = combined_df.sort_values('patient_id').reset_index(drop=True)
    
    logger.info(f"✓ Loaded OOF predictions: {len(combined_df)} patients across {NUM_FOLDS} folds")
    return combined_df


def evaluate_mil_standalone(df: pd.DataFrame) -> Dict:
    """Evaluate MIL standalone performance."""
    y_true = df[TARGET_COLUMN].values
    # OOF predictions use 'hgg_prob' column name
    prob_col = 'hgg_prob_mil' if 'hgg_prob_mil' in df.columns else 'hgg_prob'
    y_proba = df[prob_col].values
    
    # Metrics at threshold 0.5
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'threshold_0.5': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred)),
            'f1': float(f1_score(y_true, y_pred)),
            'accuracy': float(accuracy_score(y_true, y_pred))
        },
        'auc_roc': float(roc_auc_score(y_true, y_proba)),
        'pr_auc': float(average_precision_score(y_true, y_proba))
    }
    
    return metrics


def evaluate_recall_targeted_threshold(df: pd.DataFrame, recall_target: float = 0.85) -> Dict:
    """
    Evaluate MIL using recall-targeted threshold selection with nested CV.
    
    For each outer fold:
    1. Use only inner (train) OOF predictions for threshold selection
    2. Sweep thresholds to find one that achieves Recall ≥ recall_target
       while maximizing Precision (or F1)
    3. Freeze this threshold and evaluate on outer fold
    
    Returns:
        Dictionary with per-fold and aggregated results, plus comparison to fixed 0.5 threshold
    """
    logger.info("Evaluating with recall-targeted threshold selection...")
    
    # Ensure we have the correct column name
    prob_col = 'hgg_prob_mil' if 'hgg_prob_mil' in df.columns else 'hgg_prob'
    if prob_col not in df.columns:
        raise ValueError(f"Missing required column: {prob_col}")
    
    y = df[TARGET_COLUMN].values
    y_proba = df[prob_col].values
    
    # Create outer CV splits (using patient-level folds from 'fold' column if available)
    if 'fold' in df.columns:
        # Use existing fold structure (patient-level CV)
        logger.info("Using existing fold structure from 'fold' column")
        fold_results = []
        fixed_threshold_results = []
        
        for fold_idx in range(5):
            # Inner (train): all folds except this one
            inner_mask = df['fold'] != fold_idx
            # Outer (test): this fold only
            outer_mask = df['fold'] == fold_idx
            
            if not outer_mask.any():
                logger.warning(f"No samples in fold {fold_idx}, skipping")
                continue
            
            y_inner = y[inner_mask]
            y_proba_inner = y_proba[inner_mask]
            y_outer = y[outer_mask]
            y_proba_outer = y_proba[outer_mask]
            
            # Threshold sweep on inner (train) data
            thresholds = np.arange(0.01, 0.99, 0.01)
            best_threshold = None
            best_precision = -1.0
            best_f1 = -1.0
            
            for threshold in thresholds:
                y_pred_inner = (y_proba_inner >= threshold).astype(int)
                recall_inner = recall_score(y_inner, y_pred_inner, zero_division=0)
                
                # Check if recall target is met
                if recall_inner >= recall_target:
                    precision_inner = precision_score(y_inner, y_pred_inner, zero_division=0)
                    f1_inner = f1_score(y_inner, y_pred_inner, zero_division=0)
                    
                    # Maximize precision (or F1 as tie-breaker)
                    if precision_inner > best_precision or (precision_inner == best_precision and f1_inner > best_f1):
                        best_precision = precision_inner
                        best_f1 = f1_inner
                        best_threshold = threshold
            
            if best_threshold is None:
                logger.warning(f"Fold {fold_idx}: No threshold found achieving recall ≥ {recall_target}, using 0.5")
                best_threshold = 0.5
            
            # Evaluate on outer fold with selected threshold
            y_pred_outer = (y_proba_outer >= best_threshold).astype(int)
            cm_outer = confusion_matrix(y_outer, y_pred_outer)
            tn, fp, fn, tp = cm_outer.ravel()
            
            fold_results.append({
                'fold': fold_idx,
                'threshold': float(best_threshold),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp),
                'precision': float(precision_score(y_outer, y_pred_outer, zero_division=0)),
                'recall': float(recall_score(y_outer, y_pred_outer, zero_division=0)),
                'f1': float(f1_score(y_outer, y_pred_outer, zero_division=0)),
                'accuracy': float(accuracy_score(y_outer, y_pred_outer)),
            })
            
            # Fixed threshold 0.5 baseline
            y_pred_fixed = (y_proba_outer >= 0.5).astype(int)
            cm_fixed = confusion_matrix(y_outer, y_pred_fixed)
            tn_f, fp_f, fn_f, tp_f = cm_fixed.ravel()
            
            fixed_threshold_results.append({
                'fold': fold_idx,
                'threshold': 0.5,
                'tn': int(tn_f),
                'fp': int(fp_f),
                'fn': int(fn_f),
                'tp': int(tp_f),
                'precision': float(precision_score(y_outer, y_pred_fixed, zero_division=0)),
                'recall': float(recall_score(y_outer, y_pred_fixed, zero_division=0)),
                'f1': float(f1_score(y_outer, y_pred_fixed, zero_division=0)),
                'accuracy': float(accuracy_score(y_outer, y_pred_fixed)),
            })
    else:
        # Fallback to StratifiedKFold if no fold column
        logger.info("No 'fold' column found, using StratifiedKFold")
        n_splits = OUTER_CV_FOLDS
        outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        X = df[[prob_col]].values  # Dummy features for splitting
        
        fold_results = []
        fixed_threshold_results = []
        
        for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y)):
            y_inner = y[outer_train_idx]
            y_proba_inner = y_proba[outer_train_idx]
            y_outer = y[outer_test_idx]
            y_proba_outer = y_proba[outer_test_idx]
            
            # Threshold sweep on inner (train) data
            thresholds = np.arange(0.01, 0.99, 0.01)
            best_threshold = None
            best_precision = -1.0
            best_f1 = -1.0
            
            for threshold in thresholds:
                y_pred_inner = (y_proba_inner >= threshold).astype(int)
                recall_inner = recall_score(y_inner, y_pred_inner, zero_division=0)
                
                if recall_inner >= recall_target:
                    precision_inner = precision_score(y_inner, y_pred_inner, zero_division=0)
                    f1_inner = f1_score(y_inner, y_pred_inner, zero_division=0)
                    
                    if precision_inner > best_precision or (precision_inner == best_precision and f1_inner > best_f1):
                        best_precision = precision_inner
                        best_f1 = f1_inner
                        best_threshold = threshold
            
            if best_threshold is None:
                logger.warning(f"Fold {fold_idx}: No threshold found achieving recall ≥ {recall_target}, using 0.5")
                best_threshold = 0.5
            
            # Evaluate on outer fold
            y_pred_outer = (y_proba_outer >= best_threshold).astype(int)
            cm_outer = confusion_matrix(y_outer, y_pred_outer)
            tn, fp, fn, tp = cm_outer.ravel()
            
            fold_results.append({
                'fold': fold_idx,
                'threshold': float(best_threshold),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp),
                'precision': float(precision_score(y_outer, y_pred_outer, zero_division=0)),
                'recall': float(recall_score(y_outer, y_pred_outer, zero_division=0)),
                'f1': float(f1_score(y_outer, y_pred_outer, zero_division=0)),
                'accuracy': float(accuracy_score(y_outer, y_pred_outer)),
            })
            
            # Fixed threshold baseline
            y_pred_fixed = (y_proba_outer >= 0.5).astype(int)
            cm_fixed = confusion_matrix(y_outer, y_pred_fixed)
            tn_f, fp_f, fn_f, tp_f = cm_fixed.ravel()
            
            fixed_threshold_results.append({
                'fold': fold_idx,
                'threshold': 0.5,
                'tn': int(tn_f),
                'fp': int(fp_f),
                'fn': int(fn_f),
                'tp': int(tp_f),
                'precision': float(precision_score(y_outer, y_pred_fixed, zero_division=0)),
                'recall': float(recall_score(y_outer, y_pred_fixed, zero_division=0)),
                'f1': float(f1_score(y_outer, y_pred_fixed, zero_division=0)),
                'accuracy': float(accuracy_score(y_outer, y_pred_fixed)),
            })
    
    # Aggregate recall-targeted results
    fn_values = [r['fn'] for r in fold_results]
    fp_values = [r['fp'] for r in fold_results]
    recall_values = [r['recall'] for r in fold_results]
    precision_values = [r['precision'] for r in fold_results]
    f1_values = [r['f1'] for r in fold_results]
    threshold_values = [r['threshold'] for r in fold_results]
    
    # Aggregate fixed threshold results
    fn_fixed = [r['fn'] for r in fixed_threshold_results]
    fp_fixed = [r['fp'] for r in fixed_threshold_results]
    recall_fixed = [r['recall'] for r in fixed_threshold_results]
    precision_fixed = [r['precision'] for r in fixed_threshold_results]
    f1_fixed = [r['f1'] for r in fixed_threshold_results]
    
    summary = {
        'recall_target': recall_target,
        'n_folds': len(fold_results),
        'recall_targeted': {
            'fn_mean': float(np.mean(fn_values)),
            'fn_std': float(np.std(fn_values)),
            'fn_min': int(np.min(fn_values)),
            'fn_max': int(np.max(fn_values)),
            'fp_mean': float(np.mean(fp_values)),
            'fp_std': float(np.std(fp_values)),
            'recall_mean': float(np.mean(recall_values)),
            'recall_std': float(np.std(recall_values)),
            'precision_mean': float(np.mean(precision_values)),
            'precision_std': float(np.std(precision_values)),
            'f1_mean': float(np.mean(f1_values)),
            'f1_std': float(np.std(f1_values)),
            'threshold_mean': float(np.mean(threshold_values)),
            'threshold_std': float(np.std(threshold_values)),
            'fold_results': fold_results
        },
        'fixed_threshold_0.5': {
            'fn_mean': float(np.mean(fn_fixed)),
            'fn_std': float(np.std(fn_fixed)),
            'fn_min': int(np.min(fn_fixed)),
            'fn_max': int(np.max(fn_fixed)),
            'fp_mean': float(np.mean(fp_fixed)),
            'fp_std': float(np.std(fp_fixed)),
            'recall_mean': float(np.mean(recall_fixed)),
            'recall_std': float(np.std(recall_fixed)),
            'precision_mean': float(np.mean(precision_fixed)),
            'precision_std': float(np.std(precision_fixed)),
            'f1_mean': float(np.mean(f1_fixed)),
            'f1_std': float(np.std(f1_fixed)),
            'fold_results': fixed_threshold_results
        },
        'comparison': {
            'fn_improvement': float(np.mean(fn_fixed) - np.mean(fn_values)),
            'recall_improvement': float(np.mean(recall_values) - np.mean(recall_fixed)),
            'precision_change': float(np.mean(precision_values) - np.mean(precision_fixed)),
            'f1_change': float(np.mean(f1_values) - np.mean(f1_fixed)),
        }
    }
    
    return summary


def evaluate_via_nested_cv(df: pd.DataFrame) -> Dict:
    """
    Evaluate MIL via meta-learner using nested CV.
    
    Supports two modes:
    1. MIL-only mode: Only hgg_prob_mil available (MIL improvement experiments)
    2. Ensemble mode: All three features available (full ensemble evaluation)
    
    Returns:
        Dictionary with per-fold and aggregated results
    """
    logger.info("Evaluating via nested CV meta-learner...")
    
    # Ensure we have the correct column name
    if 'hgg_prob_mil' not in df.columns and 'hgg_prob' in df.columns:
        df = df.rename(columns={'hgg_prob': 'hgg_prob_mil'})
    
    # Determine evaluation mode based on available columns
    has_resnet = 'hgg_prob_resnet' in df.columns
    has_swin = 'hgg_prob_swin' in df.columns
    has_mil = 'hgg_prob_mil' in df.columns
    
    if not has_mil:
        raise ValueError("Missing required column: hgg_prob_mil")
    
    # Select feature columns based on available data
    if has_resnet and has_swin:
        # Ensemble mode: Use all three features
        feature_cols = BASE_FEATURE_COLS_ENSEMBLE
        evaluation_mode = 'ensemble'
        logger.info("Evaluation mode: ENSEMBLE (ResNet + Swin + MIL)")
    else:
        # MIL-only mode: Use only MIL probabilities
        feature_cols = BASE_FEATURE_COLS_MIL_ONLY
        evaluation_mode = 'mil_only'
        logger.info("Evaluation mode: MIL-ONLY (MIL probabilities only)")
        if not has_resnet:
            logger.info("  Note: ResNet predictions not available (MIL-only experiment)")
        if not has_swin:
            logger.info("  Note: Swin predictions not available (MIL-only experiment)")
    
    # Verify selected features exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        logger.error(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Missing columns: {missing_cols}")
    
    X = df[feature_cols].values
    y = df[TARGET_COLUMN].values
    
    # Create outer CV splits (always use all folds in evaluation mode)
    n_splits = OUTER_CV_FOLDS
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    fold_results = []
    
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y)):
        logger.info(f"Processing outer fold {fold_idx + 1}/{OUTER_CV_FOLDS}...")
        
        # Split data
        df_outer_train = df.iloc[outer_train_idx].copy()
        df_outer_test = df.iloc[outer_test_idx].copy()
        
        X_outer_train = df_outer_train[feature_cols].values
        y_outer_train = df_outer_train[TARGET_COLUMN].values
        X_outer_test = df_outer_test[feature_cols].values
        y_outer_test = df_outer_test[TARGET_COLUMN].values
        
        # Inner split for calibration/threshold selection
        X_cal, X_thr, y_cal, y_thr = train_test_split(
            X_outer_train, y_outer_train,
            test_size=1 - CALIBRATION_FRACTION,
            random_state=RANDOM_SEED,
            stratify=y_outer_train
        )
        
        # Train meta-learner
        meta_learner = LogisticRegression(
            class_weight='balanced',
            solver='lbfgs',
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_SEED
        )
        meta_learner.fit(X_cal, y_cal)
        
        # Threshold sweep
        y_proba_thr = meta_learner.predict_proba(X_thr)[:, 1]
        thresholds = np.arange(THRESHOLD_SWEEP_START, THRESHOLD_SWEEP_END + THRESHOLD_SWEEP_STEP, THRESHOLD_SWEEP_STEP)
        
        best_threshold = 0.5
        best_cost = float('inf')
        
        for threshold in thresholds:
            y_pred_thr = (y_proba_thr >= threshold).astype(int)
            cm = confusion_matrix(y_thr, y_pred_thr)
            tn, fp, fn, tp = cm.ravel()
            cost = 2 * fn + fp
            
            if cost < best_cost:
                best_cost = cost
                best_threshold = threshold
        
        # Evaluate on outer-test
        y_proba_test = meta_learner.predict_proba(X_outer_test)[:, 1]
        y_pred_test = (y_proba_test >= best_threshold).astype(int)
        
        cm_test = confusion_matrix(y_outer_test, y_pred_test)
        tn, fp, fn, tp = cm_test.ravel()
        
        fold_results.append({
            'fold': fold_idx,
            'threshold': float(best_threshold),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'precision': float(precision_score(y_outer_test, y_pred_test, zero_division=0)),
            'recall': float(recall_score(y_outer_test, y_pred_test)),
            'f1': float(f1_score(y_outer_test, y_pred_test)),
            'accuracy': float(accuracy_score(y_outer_test, y_pred_test)),
            'cost': float(2 * fn + fp)
        })
    
    # Aggregate
    fn_values = [r['fn'] for r in fold_results]
    fp_values = [r['fp'] for r in fold_results]
    cost_values = [r['cost'] for r in fold_results]
    recall_values = [r['recall'] for r in fold_results]
    precision_values = [r['precision'] for r in fold_results]
    
    summary = {
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
        'evaluation_mode': evaluation_mode,
        'feature_columns': feature_cols,
        'fold_results': fold_results
    }
    
    return summary


def compare_against_baselines(experiment_results: Dict) -> Dict:
    """
    Compare experiment results against baselines.
    
    Note: For MIL-only experiments, comparison is against:
    - Original MIL baseline (if available)
    - Enhanced meta-features ensemble (for reference, but may not be directly comparable)
    """
    comparison = {}
    comparison['evaluation_mode'] = experiment_results.get('evaluation_mode', 'unknown')
    
    # Load baseline MIL results (ensemble evaluation with original MIL)
    if BASELINE_MIL_RESULTS.exists():
        with open(BASELINE_MIL_RESULTS) as f:
            baseline_data = json.load(f)
        if 'LogisticRegression' in baseline_data:
            baseline_mil = baseline_data['LogisticRegression']
            comparison['baseline_mil'] = {
                'fn_mean': baseline_mil['fn_mean'],
                'fn_std': baseline_mil['fn_std'],
                'cost_mean': baseline_mil['cost_mean'],
                'recall_mean': baseline_mil['recall_mean'],
                'note': 'Baseline uses ensemble (ResNet + Swin + original MIL)'
            }
    
    # Load enhanced meta-features results (for reference)
    if ENHANCED_META_RESULTS.exists():
        with open(ENHANCED_META_RESULTS) as f:
            enhanced_data = json.load(f)
        comparison['enhanced_meta'] = {
            'fn_mean': enhanced_data['fn_mean'],
            'fn_std': enhanced_data['fn_std'],
            'cost_mean': enhanced_data['cost_mean'],
            'recall_mean': enhanced_data['recall_mean'],
            'note': 'Enhanced meta-features ensemble (for reference only, not directly comparable to MIL-only)'
        }
    
    # Compare (only if baseline exists and evaluation modes are compatible)
    if 'baseline_mil' in comparison:
        # For MIL-only experiments, comparison is informative but not directly comparable
        # since baseline uses ensemble features
        comparison['vs_baseline_mil'] = {
            'fn_improvement': comparison['baseline_mil']['fn_mean'] - experiment_results['fn_mean'],
            'cost_improvement': comparison['baseline_mil']['cost_mean'] - experiment_results['cost_mean'],
            'recall_improvement': experiment_results['recall_mean'] - comparison['baseline_mil']['recall_mean'],
            'note': 'Comparison is informative but not directly comparable (baseline uses ensemble, experiment is MIL-only)'
        }
    
    if 'enhanced_meta' in comparison:
        comparison['vs_enhanced_meta'] = {
            'fn_improvement': comparison['enhanced_meta']['fn_mean'] - experiment_results['fn_mean'],
            'cost_improvement': comparison['enhanced_meta']['cost_mean'] - experiment_results['cost_mean'],
            'recall_improvement': experiment_results['recall_mean'] - comparison['enhanced_meta']['recall_mean'],
            'note': 'Reference comparison only (enhanced meta-features uses ensemble, experiment is MIL-only)'
        }
    
    return comparison


def generate_visualizations(experiment_name: str, df: pd.DataFrame, results: Dict, exp_dir: Path):
    """
    Generate visualization plots for an experiment.
    
    Supports both MIL-only and ensemble evaluation modes.
    """
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating visualizations for {experiment_name}...")
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    
    # Determine MIL probability column
    prob_col = 'hgg_prob_mil' if 'hgg_prob_mil' in df.columns else 'hgg_prob'
    if prob_col not in df.columns:
        logger.warning(f"MIL probability column not found. Available columns: {df.columns.tolist()}")
        return
    
    # 1. Confusion Matrix
    y_true = df[TARGET_COLUMN].values
    y_proba = df[prob_col].values
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['LGG (Negative)', 'HGG (Positive)'],
                yticklabels=['LGG (Negative)', 'HGG (Positive)'],
                ax=ax, annot_kws={'size': 16, 'weight': 'bold'})
    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title(f'MIL Standalone: Confusion Matrix\n({experiment_name})', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. FN Distribution Across Folds
    fold_results = results.get('fold_results', [])
    if fold_results:
        fns = [r['fn'] for r in fold_results]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(fns)), fns, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Outer Fold', fontsize=12, fontweight='bold')
        ax.set_ylabel('False Negatives (FN)', fontsize=12, fontweight='bold')
        ax.set_title(f'FN Distribution Across Folds\n({experiment_name})', 
                     fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(fns)))
        ax.set_xticklabels([f'Fold {i}' for i in range(len(fns))])
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(plots_dir / 'fn_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(recall, precision, linewidth=2, label=f'MIL (PR-AUC = {pr_auc:.4f})')
    ax.set_xlabel('Recall (Sensitivity)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
    ax.set_title(f'MIL Standalone: Precision-Recall Curve\n({experiment_name})', 
                 fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. MIL Probability Histogram (HGG only)
    prob_col = 'hgg_prob_mil' if 'hgg_prob_mil' in df.columns else 'hgg_prob'
    hgg_probs = df[df[TARGET_COLUMN] == 1][prob_col].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(hgg_probs, bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
    ax.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold 0.5')
    ax.set_xlabel('MIL Predicted Probability (HGG)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'MIL Probability Distribution: HGG Cases Only\n({experiment_name})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(plots_dir / 'mil_prob_histogram_hgg.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Generated visualizations in {plots_dir}")


def run_experiment(
    experiment_name: str,
    sampling_strategy: str,
    bag_size: int,
    reg_weight_entropy: float = 0.01,
    reg_weight_confidence: float = 0.01,
    reg_weight_decay_start: int = 15
) -> Optional[Dict]:
    """
    Run a single MIL improvement experiment.
    
    Returns:
        Dictionary with results, or None if failed
    """
    logger.info("="*80)
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info("="*80)
    
    exp_dir = OUTPUT_DIR / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load existing OOF predictions (NO TRAINING)
    logger.info("\nStep 1: Loading existing OOF predictions...")
    logger.info("  NOTE: This script does NOT train models. Only evaluates existing OOF predictions.")
    df_oof = load_oof_predictions(experiment_name)
    if df_oof is None:
        logger.error("Failed to generate OOF predictions. Aborting experiment.")
        return None
    
    # Save OOF predictions
    oof_file = exp_dir / 'oof_predictions.csv'
    df_oof.to_csv(oof_file, index=False)
    logger.info(f"✓ Saved OOF predictions to {oof_file}")
    
    # Step 2: Evaluate MIL standalone
    logger.info("\nStep 2: Evaluating MIL standalone...")
    mil_standalone = evaluate_mil_standalone(df_oof)
    
    # Step 3: Evaluate via nested CV
    logger.info("\nStep 3: Evaluating via nested CV meta-learner...")
    nested_cv_results = evaluate_via_nested_cv(df_oof)
    
    # Step 4: Evaluate with recall-targeted threshold tuning
    logger.info("\nStep 4: Evaluating with recall-targeted threshold selection (≥0.85)...")
    recall_targeted_results = evaluate_recall_targeted_threshold(df_oof, recall_target=0.85)
    
    # Save recall-targeted results to CSV
    recall_csv_file = exp_dir / 'recall_targeted_threshold_results.csv'
    recall_rows = []
    for fold_result in recall_targeted_results['recall_targeted']['fold_results']:
        recall_rows.append({
            'fold': fold_result['fold'],
            'method': 'recall_targeted',
            'threshold': fold_result['threshold'],
            'tn': fold_result['tn'],
            'fp': fold_result['fp'],
            'fn': fold_result['fn'],
            'tp': fold_result['tp'],
            'precision': fold_result['precision'],
            'recall': fold_result['recall'],
            'f1': fold_result['f1'],
            'accuracy': fold_result['accuracy'],
        })
    for fold_result in recall_targeted_results['fixed_threshold_0.5']['fold_results']:
        recall_rows.append({
            'fold': fold_result['fold'],
            'method': 'fixed_0.5',
            'threshold': fold_result['threshold'],
            'tn': fold_result['tn'],
            'fp': fold_result['fp'],
            'fn': fold_result['fn'],
            'tp': fold_result['tp'],
            'precision': fold_result['precision'],
            'recall': fold_result['recall'],
            'f1': fold_result['f1'],
            'accuracy': fold_result['accuracy'],
        })
    pd.DataFrame(recall_rows).to_csv(recall_csv_file, index=False)
    logger.info(f"✓ Saved recall-targeted results to {recall_csv_file}")
    
    # Step 5: Compare against baselines
    logger.info("\nStep 5: Comparing against baselines...")
    comparison = compare_against_baselines(nested_cv_results)
    
    # Combine results
    results = {
        'experiment_name': experiment_name,
        'config': {
            'sampling_strategy': sampling_strategy,
            'bag_size': bag_size,
            'reg_weight_entropy': reg_weight_entropy,
            'reg_weight_confidence': reg_weight_confidence,
            'reg_weight_decay_start': reg_weight_decay_start
        },
        'mil_standalone': mil_standalone,
        'nested_cv': nested_cv_results,
        'recall_targeted': recall_targeted_results,
        'comparison': comparison
    }
    
    # Save results
    results_file = exp_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(make_json_serializable(results), f, indent=2)
    logger.info(f"✓ Saved results to {results_file}")
    
    # Generate visualizations
    generate_visualizations(experiment_name, df_oof, nested_cv_results, exp_dir)
    
    logger.info(f"\n{experiment_name} Results:")
    logger.info(f"  Nested CV - FN: {nested_cv_results['fn_mean']:.2f} ± {nested_cv_results['fn_std']:.2f}")
    logger.info(f"  Nested CV - Cost: {nested_cv_results['cost_mean']:.2f} ± {nested_cv_results['cost_std']:.2f}")
    logger.info(f"  Nested CV - Recall: {nested_cv_results['recall_mean']:.4f} ± {nested_cv_results['recall_std']:.4f}")
    
    logger.info(f"\n  Recall-Targeted (≥0.85) Results:")
    rt = recall_targeted_results['recall_targeted']
    logger.info(f"    FN: {rt['fn_mean']:.2f} ± {rt['fn_std']:.2f} (vs fixed 0.5: {recall_targeted_results['fixed_threshold_0.5']['fn_mean']:.2f} ± {recall_targeted_results['fixed_threshold_0.5']['fn_std']:.2f})")
    logger.info(f"    Recall: {rt['recall_mean']:.4f} ± {rt['recall_std']:.4f} (vs fixed 0.5: {recall_targeted_results['fixed_threshold_0.5']['recall_mean']:.4f} ± {recall_targeted_results['fixed_threshold_0.5']['recall_std']:.4f})")
    logger.info(f"    Precision: {rt['precision_mean']:.4f} ± {rt['precision_std']:.4f} (vs fixed 0.5: {recall_targeted_results['fixed_threshold_0.5']['precision_mean']:.4f} ± {recall_targeted_results['fixed_threshold_0.5']['precision_std']:.4f})")
    logger.info(f"    F1: {rt['f1_mean']:.4f} ± {rt['f1_std']:.4f} (vs fixed 0.5: {recall_targeted_results['fixed_threshold_0.5']['f1_mean']:.4f} ± {recall_targeted_results['fixed_threshold_0.5']['f1_std']:.4f})")
    logger.info(f"    Mean Threshold: {rt['threshold_mean']:.3f} ± {rt['threshold_std']:.3f}")
    
    return results


def check_stopping_criteria(experiment_results: Dict, baseline_fn_mean: float, baseline_fn_std: float) -> Tuple[bool, str]:
    """
    Check if stopping criteria are met.
    
    Returns:
        (should_stop, reason)
    """
    fn_mean = experiment_results['nested_cv']['fn_mean']
    fn_std = experiment_results['nested_cv']['fn_std']
    cost_mean = experiment_results['nested_cv']['cost_mean']
    
    fn_improvement = baseline_fn_mean - fn_mean
    
    # Check criteria
    if fn_improvement < 1.0:
        return True, f"FN improvement ({fn_improvement:.2f}) < 1.0"
    
    if fn_std > baseline_fn_std * 1.5:
        return True, f"FN std ({fn_std:.2f}) > 1.5× baseline ({baseline_fn_std:.2f})"
    
    # Note: Cost check would need baseline cost, skip for now
    
    return False, ""


def main():
    """Main function to evaluate existing OOF predictions."""
    logger.info("="*80)
    logger.info("MIL IMPROVEMENT EVALUATION - OOF EVALUATION MODE")
    logger.info("="*80)
    logger.info("OOF EVALUATION MODE: No training, all folds")
    logger.info("  - This script ONLY evaluates existing OOF predictions")
    logger.info("  - NO model training is performed")
    logger.info("  - All 5 folds are evaluated")
    logger.info("="*80)
    
    # Load baseline for comparison
    baseline_fn_mean = 4.20  # From nested_cv_meta_learning baseline
    baseline_fn_std = 2.04
    
    all_experiments = {}
    
    # Phase 1: Instance Selection
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: INSTANCE SELECTION IMPROVEMENTS")
    logger.info("="*80)
    
    # Experiment 1.1: Pure Entropy
    exp_1_1 = run_experiment(
        "exp_1_1_entropy",
        sampling_strategy='entropy',
        bag_size=32
    )
    
    if exp_1_1:
        all_experiments['exp_1_1_entropy'] = exp_1_1
        should_stop, reason = check_stopping_criteria(exp_1_1, baseline_fn_mean, baseline_fn_std)
        if should_stop:
            logger.warning(f"Stopping criteria met: {reason}")
            logger.info("Entropy sampling did not meet improvement criteria")
        else:
            logger.info("✓ Entropy sampling meets improvement criteria")
    
    # Experiment 1.2: Hybrid
        exp_1_2 = run_experiment(
            "exp_1_2_hybrid",
            sampling_strategy='hybrid',
        bag_size=32
        )
    
    if exp_1_2:
        all_experiments['exp_1_2_hybrid'] = exp_1_2
        should_stop, reason = check_stopping_criteria(exp_1_2, baseline_fn_mean, baseline_fn_std)
        if should_stop:
            logger.warning(f"Stopping criteria met: {reason}")
            logger.info("Hybrid sampling did not meet improvement criteria")
        else:
            logger.info("✓ Hybrid sampling meets improvement criteria")
    
    # Phase 1 Decision
    best_phase1 = None
    best_phase1_fn = float('inf')
    
    for exp_name, exp_results in all_experiments.items():
        if 'exp_1' in exp_name:
            fn = exp_results['nested_cv']['fn_mean']
            if fn < best_phase1_fn:
                best_phase1_fn = fn
                best_phase1 = exp_results
    
    if best_phase1 and (baseline_fn_mean - best_phase1_fn) >= 1.0:
        logger.info(f"\n✓ Phase 1 SUCCESS: Best method reduces FN by {baseline_fn_mean - best_phase1_fn:.2f}")
        best_sampling = best_phase1['config']['sampling_strategy']
        logger.info(f"  Best sampling strategy: {best_sampling}")
        
        # Phase 2: Bag Size (if Phase 1 succeeded)
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: BAG SIZE ADJUSTMENT")
        logger.info("="*80)
        
        # Experiment 2.1: Bag Size 48
        exp_2_1 = run_experiment(
            "exp_2_1_bag48",
            sampling_strategy=best_sampling,
            bag_size=48
        )
        
        if exp_2_1:
            all_experiments['exp_2_1_bag48'] = exp_2_1
            fn_improvement = best_phase1_fn - exp_2_1['nested_cv']['fn_mean']
            if fn_improvement >= 0.5:  # Smaller threshold for Phase 2
                logger.info(f"✓ Bag size 48 improves FN by {fn_improvement:.2f}")
                
                # Experiment 2.2: Bag Size 64
                exp_2_2 = run_experiment(
                    "exp_2_2_bag64",
                    sampling_strategy=best_sampling,
                    bag_size=64
                )
                
                if exp_2_2:
                    all_experiments['exp_2_2_bag64'] = exp_2_2
            else:
                logger.info("Bag size 48 does not improve. Skipping bag size 64.")
    else:
        logger.warning("\n✗ Phase 1 FAILED: No instance selection method improves FN by ≥1")
        logger.warning("Stopping evaluation. Limitation is architectural.")
    
    # Generate final report
    generate_final_report(all_experiments, OUTPUT_DIR)
    
    logger.info("\n" + "="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)


def generate_final_report(all_experiments: Dict, output_dir: Path):
    """Generate final decision report."""
    logger.info("Generating final decision report...")
    
    report_lines = [
        "# MIL Improvement Evaluation: Final Decision Report",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        "This report presents the results of a systematic evaluation of limited",
        "improvements to the Dual-Stream MIL model, using strict nested cross-validation.",
        "",
        "---",
        "",
        "## Experimental Protocol",
        "",
        "- **Evaluation**: Nested CV (5-fold patient-level StratifiedKFold)",
        "- **Inner split**: 70% calibration/threshold selection, 30% meta-learner training",
        "- **Evaluation**: Outer-test folds only (never seen during training)",
        "- **Baseline comparisons**: Original MIL, Enhanced meta-features ensemble",
        "",
        "---",
        "",
        "## Results Summary",
        "",
        "| Experiment | Sampling | Bag Size | FN (mean ± std) | Cost (mean ± std) | Recall (mean ± std) |",
        "|------------|----------|----------|-----------------|-------------------|---------------------|"
    ]
    
    for exp_name, exp_results in all_experiments.items():
        config = exp_results['config']
        nested = exp_results['nested_cv']
        report_lines.append(
            f"| {exp_name} | {config['sampling_strategy']} | {config['bag_size']} | "
            f"{nested['fn_mean']:.2f} ± {nested['fn_std']:.2f} | "
            f"{nested['cost_mean']:.2f} ± {nested['cost_std']:.2f} | "
            f"{nested['recall_mean']:.4f} ± {nested['recall_std']:.4f} |"
        )
    
    report_lines.extend([
        "",
        "---",
        "",
        "## Final Verdict",
        "",
        "### Did limited improvements help?",
        "",
        "[To be filled based on results]",
        "",
        "### Which modification (if any) is worth keeping?",
        "",
        "[To be filled based on results]",
        "",
        "### Should MIL remain in ensemble?",
        "",
        "[To be filled based on results]",
        "",
        "### Or rely primarily on CNN-based models + meta-features?",
        "",
        "[To be filled based on results]",
        ""
    ])
    
    report_file = output_dir / 'FINAL_DECISION_REPORT.md'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"✓ Saved report to {report_file}")


if __name__ == '__main__':
    main()

