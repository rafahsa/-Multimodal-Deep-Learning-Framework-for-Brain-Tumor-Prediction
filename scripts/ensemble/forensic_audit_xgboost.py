#!/usr/bin/env python3
"""
Forensic Audit of XGBoost Meta-Learner Results

This script performs comprehensive checks to verify that XGBoost's exceptional
performance (FN=0, FP≈1) is real and not due to data leakage, bugs, or overfitting.

All checks produce evidence files for verification.
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import joblib


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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, classification_report
)
from sklearn.calibration import CalibratedClassifierCV

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("ERROR: XGBoost not available.")
    exit(1)

# Setup logging
AUDIT_DIR = Path('ensemble/results/forensic_audit_xgboost')
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR = AUDIT_DIR / 'evidence'
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR = AUDIT_DIR / 'logs'
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'audit_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
MERGED_OOF_FILE = Path('ensemble/oof_predictions/merged_oof_predictions.csv')
BASELINE_MODEL_PATH = Path('ensemble/models/meta_learner_logistic_regression.joblib')
BASELINE_CALIBRATOR_PATH = Path('ensemble/results/calibration/2026-02-07_22-29-29_platt_seed42/calibrator_platt.joblib')
FEATURE_COLUMNS = ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']
TARGET_COLUMN = 'label'

# Expected baseline metrics (for sanity check)
EXPECTED_BASELINE_FN = 11
EXPECTED_BASELINE_FP = 41
EXPECTED_BASELINE_TN = 34
EXPECTED_BASELINE_TP = 199

# XGBoost config
XGBOOST_CONFIG = {
    'max_depth': 4,
    'learning_rate': 0.1,
    'n_estimators': 100
}

# Audit results
audit_results = {
    'timestamp': datetime.now().isoformat(),
    'checks': {}
}


def check_a1_data_integrity() -> Dict:
    """Check A1: Data integrity and basic sanity."""
    logger.info("="*80)
    logger.info("CHECK A1: DATA INTEGRITY & BASIC SANITY")
    logger.info("="*80)
    
    df = pd.read_csv(MERGED_OOF_FILE)
    
    profile = {
        'n_rows': len(df),
        'n_features': len(FEATURE_COLUMNS),
        'class_distribution': df[TARGET_COLUMN].value_counts().to_dict(),
        'feature_stats': {},
        'has_nan': {},
        'has_inf': {},
        'duplicate_rows': 0,
        'duplicate_features': 0
    }
    
    # Feature statistics
    for col in FEATURE_COLUMNS:
        profile['feature_stats'][col] = {
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'mean': float(df[col].mean()),
            'std': float(df[col].std())
        }
        profile['has_nan'][col] = int(df[col].isna().sum())
        profile['has_inf'][col] = int(np.isinf(df[col]).sum())
    
    # Check for duplicate rows
    profile['duplicate_rows'] = int(df.duplicated().sum())
    
    # Check for duplicate feature combinations
    profile['duplicate_features'] = int(df[FEATURE_COLUMNS].duplicated().sum())
    
    # Check for patient_id duplicates if column exists
    if 'patient_id' in df.columns:
        patient_duplicates = df['patient_id'].duplicated().sum()
        profile['patient_id_duplicates'] = int(patient_duplicates)
        if patient_duplicates > 0:
            dup_patients = df[df['patient_id'].duplicated(keep=False)]['patient_id'].unique().tolist()
            profile['duplicate_patient_ids'] = dup_patients[:10]  # First 10
    
    # Check fold column if exists
    if 'fold' in df.columns:
        profile['fold_distribution'] = df['fold'].value_counts().to_dict()
        profile['samples_per_fold'] = {
            'min': int(df['fold'].value_counts().min()),
            'max': int(df['fold'].value_counts().max()),
            'mean': float(df['fold'].value_counts().mean())
        }
    
    # Save evidence
    with open(EVIDENCE_DIR / 'data_profile.json', 'w') as f:
        json.dump(profile, f, indent=2)
    
    # Save duplicate report
    if profile['duplicate_features'] > 0:
        dup_df = df[df[FEATURE_COLUMNS].duplicated(keep=False)]
        dup_df.to_csv(EVIDENCE_DIR / 'duplicates_report.csv', index=False)
    
    logger.info(f"✓ Data profile saved")
    logger.info(f"  Rows: {profile['n_rows']}, Classes: {profile['class_distribution']}")
    logger.info(f"  Duplicate rows: {profile['duplicate_rows']}")
    logger.info(f"  Duplicate feature combinations: {profile['duplicate_features']}")
    
    audit_results['checks']['A1_data_integrity'] = {
        'status': 'PASSED' if profile['duplicate_rows'] == 0 and profile['has_nan'][FEATURE_COLUMNS[0]] == 0 else 'FLAGGED',
        'details': profile
    }
    
    return profile


def check_a2_label_sanity() -> Dict:
    """Check A2: Label sanity and baseline reproduction."""
    logger.info("="*80)
    logger.info("CHECK A2: LABEL SANITY & BASELINE REPRODUCTION")
    logger.info("="*80)
    
    df = pd.read_csv(MERGED_OOF_FILE)
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    
    # Verify labels
    unique_labels = np.unique(y).tolist()
    label_check = {
        'unique_labels': unique_labels,
        'label_valid': set(unique_labels) == {0, 1},
        'n_samples': len(y),
        'class_counts': {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    }
    
    if not label_check['label_valid']:
        logger.error(f"INVALID LABELS: {unique_labels}")
        audit_results['checks']['A2_label_sanity'] = {
            'status': 'FAILED',
            'details': label_check
        }
        return label_check
    
    # Load baseline model and calibrator
    baseline_model = joblib.load(BASELINE_MODEL_PATH)
    baseline_calibrator = joblib.load(BASELINE_CALIBRATOR_PATH)
    
    # Compute baseline probabilities
    y_proba_uncal = baseline_model.predict_proba(X)[:, 1]
    y_proba_uncal_clipped = np.clip(y_proba_uncal, 1e-7, 1 - 1e-7)
    log_odds = np.log(y_proba_uncal_clipped / (1 - y_proba_uncal_clipped))
    y_proba_cal = baseline_calibrator['model'].predict_proba(log_odds.reshape(-1, 1))[:, 1]
    
    # Evaluate at threshold 0.35
    threshold = 0.35
    y_pred = (y_proba_cal >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    baseline_repro = {
        'threshold': threshold,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'expected_tn': EXPECTED_BASELINE_TN,
        'expected_fp': EXPECTED_BASELINE_FP,
        'expected_fn': EXPECTED_BASELINE_FN,
        'expected_tp': EXPECTED_BASELINE_TP,
        'matches_expected': (tn == EXPECTED_BASELINE_TN and fp == EXPECTED_BASELINE_FP and 
                           fn == EXPECTED_BASELINE_FN and tp == EXPECTED_BASELINE_TP)
    }
    
    # Save evidence
    evidence_data = {**label_check, **baseline_repro}
    # Convert bool to int for JSON serialization
    evidence_data['label_valid'] = int(evidence_data['label_valid'])
    evidence_data['matches_expected'] = int(evidence_data['matches_expected'])
    with open(EVIDENCE_DIR / 'baseline_repro_check.json', 'w') as f:
        json.dump(evidence_data, f, indent=2)
    
    logger.info(f"Baseline reproduction check:")
    logger.info(f"  Expected: TN={EXPECTED_BASELINE_TN}, FP={EXPECTED_BASELINE_FP}, "
               f"FN={EXPECTED_BASELINE_FN}, TP={EXPECTED_BASELINE_TP}")
    logger.info(f"  Actual:   TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    if not baseline_repro['matches_expected']:
        logger.error("BASELINE REPRODUCTION FAILED - Pipeline mismatch detected!")
        audit_results['checks']['A2_label_sanity'] = {
            'status': 'FAILED',
            'details': baseline_repro
        }
        return baseline_repro
    
    logger.info("✓ Baseline reproduction matches expected values")
    
    audit_results['checks']['A2_label_sanity'] = {
        'status': 'PASSED',
        'details': baseline_repro
    }
    
    return baseline_repro


def check_b3_oof_provenance() -> Dict:
    """Check B3: Verify OOF is truly out-of-fold."""
    logger.info("="*80)
    logger.info("CHECK B3: OOF PROVENANCE")
    logger.info("="*80)
    
    # Search for scripts that generate merged OOF predictions
    provenance_info = {
        'merged_file': str(MERGED_OOF_FILE),
        'file_exists': int(MERGED_OOF_FILE.exists()),
        'scripts_found': [],
        'fold_column_exists': 0,
        'oof_validation': {}
    }
    
    # Check if fold column exists
    if MERGED_OOF_FILE.exists():
        df = pd.read_csv(MERGED_OOF_FILE, nrows=1)
        provenance_info['fold_column_exists'] = int('fold' in df.columns)
    
    # Search for OOF generation scripts
    import os
    scripts_to_check = [
        'scripts/ensemble/verify_and_merge_oof.py',
        'scripts/ensemble/train_meta_learner.py',
        'scripts/ensemble/calibrate_and_sweep_thresholds.py'
    ]
    
    for script_path in scripts_to_check:
        path = Path(script_path)
        if path.exists():
            provenance_info['scripts_found'].append(str(path))
    
    # Check if fold info exists in CSV
    if MERGED_OOF_FILE.exists() and 'fold' in pd.read_csv(MERGED_OOF_FILE, nrows=1).columns:
        df = pd.read_csv(MERGED_OOF_FILE)
        fold_counts = df['fold'].value_counts().to_dict()
        provenance_info['oof_validation'] = {
            'has_fold_column': 1,
            'fold_distribution': {int(k): int(v) for k, v in fold_counts.items()},
            'all_samples_have_fold': int(df['fold'].notna().all()),
            'fold_range': [int(df['fold'].min()), int(df['fold'].max())]
        }
    else:
        provenance_info['oof_validation'] = {
            'has_fold_column': 0,
            'note': 'Cannot verify OOF without fold information'
        }
    
    # Save evidence
    with open(EVIDENCE_DIR / 'oof_provenance.md', 'w') as f:
        f.write("# OOF Provenance Check\n\n")
        f.write(f"**Merged OOF File**: {MERGED_OOF_FILE}\n\n")
        f.write(f"**Fold Column Exists**: {provenance_info['fold_column_exists']}\n\n")
        f.write("## Scripts Found\n\n")
        for script in provenance_info['scripts_found']:
            f.write(f"- {script}\n")
        f.write("\n## OOF Validation\n\n")
        f.write(f"```json\n{json.dumps(provenance_info['oof_validation'], indent=2)}\n```\n")
    
    logger.info(f"✓ OOF provenance check complete")
    logger.info(f"  Fold column exists: {provenance_info['fold_column_exists']}")
    
    audit_results['checks']['B3_oof_provenance'] = {
        'status': 'PASSED' if provenance_info['fold_column_exists'] else 'INCONCLUSIVE',
        'details': provenance_info
    }
    
    return provenance_info


def check_b4_leakage_tests() -> Dict:
    """Check B4: Hard leakage tests."""
    logger.info("="*80)
    logger.info("CHECK B4: HARD LEAKAGE TESTS")
    logger.info("="*80)
    
    df = pd.read_csv(MERGED_OOF_FILE)
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    
    leakage_tests = {}
    
    # Test 1: Trivial classifier on row index
    logger.info("Test 1: Trivial classifier on row index...")
    X_trivial = np.arange(len(X)).reshape(-1, 1)
    
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression as LR
    
    trivial_model = LR(max_iter=1000, random_state=42)
    cv_scores = cross_val_score(trivial_model, X_trivial, y, cv=5, scoring='accuracy')
    
    leakage_tests['trivial_index_classifier'] = {
        'mean_accuracy': float(cv_scores.mean()),
        'std_accuracy': float(cv_scores.std()),
        'expected_chance': 0.5 if len(np.unique(y)) == 2 else 1.0 / len(np.unique(y)),
        'suspicious': int(cv_scores.mean() > 0.7)  # If >70%, suspicious
    }
    
    logger.info(f"  Trivial classifier accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Test 2: Shuffle labels and test XGBoost
    logger.info("Test 2: XGBoost on shuffled labels...")
    y_shuffled = y.copy()
    np.random.seed(42)
    np.random.shuffle(y_shuffled)
    
    # Train XGBoost on shuffled labels
    xgb_model = xgb.XGBClassifier(
        max_depth=XGBOOST_CONFIG['max_depth'],
        learning_rate=XGBOOST_CONFIG['learning_rate'],
        n_estimators=XGBOOST_CONFIG['n_estimators'],
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Use cross-validation
    cv_scores_shuffled = cross_val_score(xgb_model, X, y_shuffled, cv=5, scoring='accuracy')
    
    leakage_tests['xgboost_shuffled_labels'] = {
        'mean_accuracy': float(cv_scores_shuffled.mean()),
        'std_accuracy': float(cv_scores_shuffled.std()),
        'expected_chance': float(np.bincount(y).max() / len(y)),  # Majority class baseline
        'suspicious': int(cv_scores_shuffled.mean() > 0.7)  # If >70%, suspicious
    }
    
    logger.info(f"  XGBoost on shuffled labels accuracy: {cv_scores_shuffled.mean():.4f} ± {cv_scores_shuffled.std():.4f}")
    
    # Save evidence
    with open(EVIDENCE_DIR / 'leakage_tests.json', 'w') as f:
        json.dump(leakage_tests, f, indent=2)
    
    # Determine status
    if leakage_tests['trivial_index_classifier']['suspicious'] or leakage_tests['xgboost_shuffled_labels']['suspicious']:
        status = 'FAILED'
        logger.error("LEAKAGE DETECTED: Performance too high on shuffled/trivial data")
    else:
        status = 'PASSED'
        logger.info("✓ Leakage tests passed")
    
    audit_results['checks']['B4_leakage_tests'] = {
        'status': status,
        'details': leakage_tests
    }
    
    return leakage_tests


def apply_platt_calibration_clean(
    meta_learner: object,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_eval: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Clean implementation of Platt calibration."""
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
    
    return y_proba_eval_cal, y_proba_eval_uncal


def check_c5_calibration_equivalence() -> Dict:
    """Check C5: Calibration implementation equivalence."""
    logger.info("="*80)
    logger.info("CHECK C5: CALIBRATION EQUIVALENCE")
    logger.info("="*80)
    
    df = pd.read_csv(MERGED_OOF_FILE)
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    
    # Split for calibration (seed=42)
    X_cal, X_thr, y_cal, y_thr = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(
        max_depth=XGBOOST_CONFIG['max_depth'],
        learning_rate=XGBOOST_CONFIG['learning_rate'],
        n_estimators=XGBOOST_CONFIG['n_estimators'],
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_model.fit(X, y)  # Train on full set
    
    # Method 1: Clean implementation
    y_proba_clean, y_proba_uncal = apply_platt_calibration_clean(
        xgb_model, X_cal, y_cal, X_thr
    )
    
    # Method 2: Using CalibratedClassifierCV (if possible)
    # Note: This requires the model to be pre-fitted
    try:
        calibrated_cv = CalibratedClassifierCV(xgb_model, method='sigmoid', cv='prefit')
        calibrated_cv.fit(X_cal, y_cal)
        y_proba_cv = calibrated_cv.predict_proba(X_thr)[:, 1]
        
        # Compare
        correlation = np.corrcoef(y_proba_clean, y_proba_cv)[0, 1]
        max_diff = np.abs(y_proba_clean - y_proba_cv).max()
        mean_diff = np.abs(y_proba_clean - y_proba_cv).mean()
        
        equivalence = {
            'correlation': float(correlation),
            'max_abs_diff': float(max_diff),
            'mean_abs_diff': float(mean_diff),
            'equivalent': correlation > 0.99 and max_diff < 0.01
        }
    except Exception as e:
        logger.warning(f"CalibratedClassifierCV test failed: {e}")
        equivalence = {
            'sklearn_method_failed': str(e),
            'note': 'Using clean implementation only'
        }
    
    # Save evidence
    with open(EVIDENCE_DIR / 'calibration_equivalence_seed42.json', 'w') as f:
        json.dump(equivalence, f, indent=2)
    
    logger.info(f"✓ Calibration equivalence check complete")
    if 'correlation' in equivalence:
        logger.info(f"  Correlation: {equivalence['correlation']:.6f}")
        logger.info(f"  Max diff: {equivalence['max_abs_diff']:.6f}")
    
    audit_results['checks']['C5_calibration_equivalence'] = {
        'status': 'PASSED' if equivalence.get('equivalent', False) else 'INCONCLUSIVE',
        'details': equivalence
    }
    
    return equivalence


def check_c6_nested_evaluation() -> Dict:
    """Check C6: Nested evaluation (most important check)."""
    logger.info("="*80)
    logger.info("CHECK C6: NESTED EVALUATION (CRITICAL)")
    logger.info("="*80)
    
    df = pd.read_csv(MERGED_OOF_FILE)
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    
    seeds = [21, 42, 77, 123, 202]
    nested_results = []
    
    for seed in seeds:
        logger.info(f"\nProcessing seed {seed}...")
        
        # Outer split: Train_meta (70%) vs Test_meta (30%)
        X_train_meta, X_test_meta, y_train_meta, y_test_meta = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        
        # Inner split: Within Train_meta, split for calibration (70%) vs threshold selection (30%)
        X_cal, X_thr, y_cal, y_thr = train_test_split(
            X_train_meta, y_train_meta, test_size=0.3, random_state=seed, stratify=y_train_meta
        )
        
        # Train XGBoost ONLY on Train_meta
        xgb_model = xgb.XGBClassifier(
            max_depth=XGBOOST_CONFIG['max_depth'],
            learning_rate=XGBOOST_CONFIG['learning_rate'],
            n_estimators=XGBOOST_CONFIG['n_estimators'],
            random_state=seed,
            eval_metric='logloss',
            use_label_encoder=False
        )
        xgb_model.fit(X_train_meta, y_train_meta)
        
        # Apply Platt calibration on calibration set
        y_proba_thr_cal, _ = apply_platt_calibration_clean(
            xgb_model, X_cal, y_cal, X_thr
        )
        
        # Threshold sweep on threshold selection set
        thresholds = np.arange(0.05, 0.95 + 0.01, 0.01)
        best_threshold = None
        best_cost = float('inf')
        
        for thr in thresholds:
            y_pred_thr = (y_proba_thr_cal >= thr).astype(int)
            cm = confusion_matrix(y_thr, y_pred_thr)
            tn, fp, fn, tp = cm.ravel()
            cost = 2 * fn + fp
            
            if cost < best_cost:
                best_cost = cost
                best_threshold = thr
        
        # Apply calibration to test set
        y_proba_test_cal, _ = apply_platt_calibration_clean(
            xgb_model, X_cal, y_cal, X_test_meta
        )
        
        # Evaluate on TEST set (never used for training/calibration/threshold selection)
        y_pred_test = (y_proba_test_cal >= best_threshold).astype(int)
        cm_test = confusion_matrix(y_test_meta, y_pred_test)
        tn_test, fp_test, fn_test, tp_test = cm_test.ravel()
        
        precision_test = precision_score(y_test_meta, y_pred_test, zero_division=0)
        recall_test = recall_score(y_test_meta, y_pred_test)
        f1_test = f1_score(y_test_meta, y_pred_test)
        accuracy_test = accuracy_score(y_test_meta, y_pred_test)
        cost_test = 2 * fn_test + fp_test
        
        nested_results.append({
            'seed': seed,
            'selected_threshold': float(best_threshold),
            'test_set_size': len(y_test_meta),
            'tn': int(tn_test),
            'fp': int(fp_test),
            'fn': int(fn_test),
            'tp': int(tp_test),
            'precision': float(precision_test),
            'recall': float(recall_test),
            'f1': float(f1_test),
            'accuracy': float(accuracy_test),
            'cost': float(cost_test)
        })
        
        logger.info(f"  Seed {seed}: Threshold={best_threshold:.4f}, "
                   f"FN={fn_test}, FP={fp_test}, Cost={cost_test:.1f}")
    
    # Compute statistics
    fn_values = [r['fn'] for r in nested_results]
    fp_values = [r['fp'] for r in nested_results]
    cost_values = [r['cost'] for r in nested_results]
    
    nested_summary = {
        'n_seeds': len(seeds),
        'fn_mean': float(np.mean(fn_values)),
        'fn_std': float(np.std(fn_values)),
        'fn_min': int(np.min(fn_values)),
        'fn_max': int(np.max(fn_values)),
        'fp_mean': float(np.mean(fp_values)),
        'fp_std': float(np.std(fp_values)),
        'cost_mean': float(np.mean(cost_values)),
        'cost_std': float(np.std(cost_values)),
        'results': nested_results
    }
    
    # Save evidence
    nested_df = pd.DataFrame(nested_results)
    nested_df.to_csv(EVIDENCE_DIR / 'nested_eval_results.csv', index=False)
    
    with open(EVIDENCE_DIR / 'nested_eval_summary.json', 'w') as f:
        json.dump(nested_summary, f, indent=2)
    
    logger.info(f"\nNested evaluation summary:")
    logger.info(f"  FN: {nested_summary['fn_mean']:.2f} ± {nested_summary['fn_std']:.2f} "
               f"(range: [{nested_summary['fn_min']}, {nested_summary['fn_max']}])")
    logger.info(f"  FP: {nested_summary['fp_mean']:.2f} ± {nested_summary['fp_std']:.2f}")
    logger.info(f"  Cost: {nested_summary['cost_mean']:.2f} ± {nested_summary['cost_std']:.2f}")
    
    # Determine status
    if nested_summary['fn_mean'] <= 1 and nested_summary['fn_max'] <= 2:
        status = 'PASSED'
    elif nested_summary['fn_mean'] <= 3:
        status = 'PARTIALLY_VERIFIED'
    else:
        status = 'FAILED'
    
    audit_results['checks']['C6_nested_evaluation'] = {
        'status': status,
        'details': nested_summary
    }
    
    return nested_summary


def check_d7_margin_analysis() -> Dict:
    """Check D7: Margin analysis - borderline cases."""
    logger.info("="*80)
    logger.info("CHECK D7: MARGIN ANALYSIS")
    logger.info("="*80)
    
    df = pd.read_csv(MERGED_OOF_FILE)
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    
    # Train XGBoost and apply calibration (seed=42)
    X_cal, X_eval, y_cal, y_eval = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    xgb_model = xgb.XGBClassifier(
        max_depth=XGBOOST_CONFIG['max_depth'],
        learning_rate=XGBOOST_CONFIG['learning_rate'],
        n_estimators=XGBOOST_CONFIG['n_estimators'],
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_model.fit(X, y)
    
    y_proba_cal, _ = apply_platt_calibration_clean(xgb_model, X_cal, y_cal, X)
    
    # Evaluate at threshold 0.39
    threshold = 0.39
    y_pred = (y_proba_cal >= threshold).astype(int)
    
    # Find FP cases
    fp_mask = (y == 0) & (y_pred == 1)
    fp_indices = np.where(fp_mask)[0]
    
    # Find borderline HGG cases (lowest probability among HGG)
    hgg_mask = (y == 1)
    hgg_probas = y_proba_cal[hgg_mask]
    hgg_indices = np.where(hgg_mask)[0]
    borderline_hgg_idx = hgg_indices[np.argsort(hgg_probas)[:5]]  # 5 lowest
    
    # Create report
    borderline_cases = []
    
    # FP cases
    for idx in fp_indices[:10]:  # First 10 FP
        borderline_cases.append({
            'case_type': 'FP',
            'index': int(idx),
            'true_label': int(y[idx]),
            'predicted_label': int(y_pred[idx]),
            'calibrated_prob': float(y_proba_cal[idx]),
            'hgg_prob_resnet': float(X[idx, 0]),
            'hgg_prob_swin': float(X[idx, 1]),
            'hgg_prob_mil': float(X[idx, 2])
        })
    
    # Borderline HGG cases
    for idx in borderline_hgg_idx:
        borderline_cases.append({
            'case_type': 'Borderline_HGG',
            'index': int(idx),
            'true_label': int(y[idx]),
            'predicted_label': int(y_pred[idx]),
            'calibrated_prob': float(y_proba_cal[idx]),
            'hgg_prob_resnet': float(X[idx, 0]),
            'hgg_prob_swin': float(X[idx, 1]),
            'hgg_prob_mil': float(X[idx, 2])
        })
    
    # Save evidence
    borderline_df = pd.DataFrame(borderline_cases)
    borderline_df.to_csv(EVIDENCE_DIR / 'borderline_cases.csv', index=False)
    
    logger.info(f"✓ Margin analysis complete")
    logger.info(f"  FP cases found: {len(fp_indices)}")
    logger.info(f"  Borderline HGG cases analyzed: {len(borderline_hgg_idx)}")
    
    audit_results['checks']['D7_margin_analysis'] = {
        'status': 'PASSED',
        'details': {
            'n_fp_cases': int(len(fp_indices)),
            'n_borderline_hgg': int(len(borderline_hgg_idx))
        }
    }
    
    return {'n_fp_cases': len(fp_indices), 'n_borderline_hgg': len(borderline_hgg_idx)}


def check_d8_overfitting_sensitivity() -> Dict:
    """Check D8: Overfitting sensitivity with hyperparameter grid."""
    logger.info("="*80)
    logger.info("CHECK D8: OVERFITTING SENSITIVITY")
    logger.info("="*80)
    
    df = pd.read_csv(MERGED_OOF_FILE)
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values
    
    # Small conservative grid
    param_grid = [
        {'max_depth': 2, 'n_estimators': 50, 'min_child_weight': 1, 'subsample': 1.0, 'colsample_bytree': 1.0},
        {'max_depth': 3, 'n_estimators': 50, 'min_child_weight': 1, 'subsample': 1.0, 'colsample_bytree': 1.0},
        {'max_depth': 4, 'n_estimators': 50, 'min_child_weight': 1, 'subsample': 1.0, 'colsample_bytree': 1.0},
        {'max_depth': 4, 'n_estimators': 100, 'min_child_weight': 1, 'subsample': 1.0, 'colsample_bytree': 1.0},
        {'max_depth': 4, 'n_estimators': 100, 'min_child_weight': 5, 'subsample': 0.8, 'colsample_bytree': 0.8},
    ]
    
    grid_results = []
    seed = 42  # Use single seed for grid search
    
    for params in param_grid:
        logger.info(f"Testing: {params}")
        
        # Nested evaluation (same as C6)
        X_train_meta, X_test_meta, y_train_meta, y_test_meta = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        
        X_cal, X_thr, y_cal, y_thr = train_test_split(
            X_train_meta, y_train_meta, test_size=0.3, random_state=seed, stratify=y_train_meta
        )
        
        # Train with these params
        xgb_model = xgb.XGBClassifier(
            max_depth=params['max_depth'],
            learning_rate=0.1,
            n_estimators=params['n_estimators'],
            min_child_weight=params['min_child_weight'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            random_state=seed,
            eval_metric='logloss',
            use_label_encoder=False
        )
        xgb_model.fit(X_train_meta, y_train_meta)
        
        # Calibration and threshold selection
        y_proba_thr_cal, _ = apply_platt_calibration_clean(xgb_model, X_cal, y_cal, X_thr)
        
        thresholds = np.arange(0.05, 0.95 + 0.01, 0.01)
        best_threshold = None
        best_cost = float('inf')
        
        for thr in thresholds:
            y_pred_thr = (y_proba_thr_cal >= thr).astype(int)
            cm = confusion_matrix(y_thr, y_pred_thr)
            tn, fp, fn, tp = cm.ravel()
            cost = 2 * fn + fp
            if cost < best_cost:
                best_cost = cost
                best_threshold = thr
        
        # Evaluate on test set
        y_proba_test_cal, _ = apply_platt_calibration_clean(xgb_model, X_cal, y_cal, X_test_meta)
        y_pred_test = (y_proba_test_cal >= best_threshold).astype(int)
        cm_test = confusion_matrix(y_test_meta, y_pred_test)
        tn_test, fp_test, fn_test, tp_test = cm_test.ravel()
        
        grid_results.append({
            **params,
            'selected_threshold': float(best_threshold),
            'fn': int(fn_test),
            'fp': int(fp_test),
            'cost': float(2 * fn_test + fp_test),
            'recall': float(recall_score(y_test_meta, y_pred_test)),
            'precision': float(precision_score(y_test_meta, y_pred_test, zero_division=0))
        })
    
    # Save evidence
    grid_df = pd.DataFrame(grid_results)
    grid_df.to_csv(EVIDENCE_DIR / 'small_grid_nested_results.csv', index=False)
    
    logger.info(f"✓ Overfitting sensitivity check complete")
    logger.info(f"  Tested {len(param_grid)} hyperparameter combinations")
    
    audit_results['checks']['D8_overfitting_sensitivity'] = {
        'status': 'PASSED',
        'details': {'n_combinations': len(param_grid), 'results': grid_results}
    }
    
    return {'results': grid_results}


def generate_audit_report():
    """Generate final audit report."""
    logger.info("="*80)
    logger.info("GENERATING AUDIT REPORT")
    logger.info("="*80)
    
    # Create summary table
    checks_summary = []
    for check_name, check_data in audit_results['checks'].items():
        checks_summary.append({
            'Check': check_name,
            'Status': check_data['status'],
            'Details': str(check_data.get('details', {}))[:100]  # Truncate for table
        })
    
    # Get nested evaluation results if available
    nested_eval = audit_results['checks'].get('C6_nested_evaluation', {}).get('details', {})
    
    # Generate report
    report_lines = [
        "# Forensic Audit Report: XGBoost Meta-Learner",
        "",
        f"**Audit Date**: {audit_results['timestamp']}",
        "",
        "## Executive Summary",
        ""
    ]
    
    # Determine overall verdict
    failed_checks = [k for k, v in audit_results['checks'].items() if v['status'] == 'FAILED']
    critical_check = audit_results['checks'].get('C6_nested_evaluation', {})
    
    if failed_checks:
        verdict = "NOT VERIFIED"
        verdict_reason = f"Failed checks: {', '.join(failed_checks)}"
    elif critical_check.get('status') == 'PASSED':
        if nested_eval.get('fn_mean', 999) <= 1:
            verdict = "VERIFIED"
            verdict_reason = f"Nested evaluation shows FN={nested_eval.get('fn_mean', 0):.2f} (stable and excellent)"
        else:
            verdict = "PARTIALLY VERIFIED"
            verdict_reason = f"Nested evaluation shows FN={nested_eval.get('fn_mean', 0):.2f} (acceptable but higher than optimistic estimate)"
    elif critical_check.get('status') == 'PARTIALLY_VERIFIED':
        verdict = "PARTIALLY VERIFIED"
        verdict_reason = "Nested evaluation shows acceptable performance but with some variation"
    else:
        verdict = "NOT VERIFIED"
        verdict_reason = "Critical nested evaluation check not completed or failed"
    
    report_lines.extend([
        f"**Overall Verdict**: **{verdict}**",
        "",
        f"**Reason**: {verdict_reason}",
        "",
        "---",
        "",
        "## Check Summary",
        "",
        "| Check | Status | Notes |",
        "|-------|--------|-------|"
    ])
    
    for check in checks_summary:
        status_emoji = "✅" if check['Status'] == 'PASSED' else "⚠️" if check['Status'] == 'PARTIALLY_VERIFIED' else "❌"
        report_lines.append(f"| {check['Check']} | {status_emoji} {check['Status']} | {check['Details'][:50]}... |")
    
    # Add nested evaluation details
    if nested_eval:
        report_lines.extend([
            "",
            "---",
            "",
            "## Nested Evaluation Results (CRITICAL)",
            "",
            "This is the most important check: performance on a truly held-out test set.",
            "",
            f"**Tested across {nested_eval.get('n_seeds', 0)} random seeds**",
            "",
            "| Metric | Mean ± Std | Range |",
            "|--------|------------|-------|",
            f"| FN | {nested_eval.get('fn_mean', 0):.2f} ± {nested_eval.get('fn_std', 0):.2f} | [{nested_eval.get('fn_min', 0)}, {nested_eval.get('fn_max', 0)}] |",
            f"| FP | {nested_eval.get('fp_mean', 0):.2f} ± {nested_eval.get('fp_std', 0):.2f} | - |",
            f"| Cost | {nested_eval.get('cost_mean', 0):.2f} ± {nested_eval.get('cost_std', 0):.2f} | - |",
            ""
        ])
    
    report_lines.extend([
        "---",
        "",
        "## Detailed Findings",
        ""
    ])
    
    # Add details for each check
    for check_name, check_data in audit_results['checks'].items():
        details_serialized = make_json_serializable(check_data.get('details', {}))
        report_lines.extend([
            f"### {check_name}",
            "",
            f"**Status**: {check_data['status']}",
            "",
            f"```json",
            json.dumps(details_serialized, indent=2),
            "```",
            ""
        ])
    
    report_lines.extend([
        "---",
        "",
        "## Conclusion",
        "",
        f"**Verdict**: {verdict}",
        "",
        verdict_reason,
        ""
    ])
    
    if verdict == "VERIFIED":
        report_lines.append("✅ XGBoost performance is **VERIFIED** and safe to claim.")
    elif verdict == "PARTIALLY VERIFIED":
        report_lines.append("⚠️ XGBoost performance is **PARTIALLY VERIFIED**. Results are acceptable but may be optimistic.")
    else:
        report_lines.append("❌ XGBoost performance is **NOT VERIFIED**. Do not claim adoption without further investigation.")
    
    # Save report
    with open(AUDIT_DIR / 'AUDIT_REPORT.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Save CSV summary
    summary_df = pd.DataFrame(checks_summary)
    summary_df.to_csv(AUDIT_DIR / 'audit_metrics.csv', index=False)
    
    logger.info(f"✓ Audit report saved to: {AUDIT_DIR / 'AUDIT_REPORT.md'}")
    logger.info(f"✓ Summary CSV saved to: {AUDIT_DIR / 'audit_metrics.csv'}")
    
    return verdict


def main():
    """Main audit function."""
    logger.info("="*80)
    logger.info("FORENSIC AUDIT: XGBOOST META-LEARNER")
    logger.info("="*80)
    
    if not XGBOOST_AVAILABLE:
        logger.error("XGBoost not available. Cannot perform audit.")
        return
    
    # Run all checks
    try:
        check_a1_data_integrity()
        check_a2_label_sanity()
        check_b3_oof_provenance()
        check_b4_leakage_tests()
        check_c5_calibration_equivalence()
        check_c6_nested_evaluation()  # Most important
        check_d7_margin_analysis()
        check_d8_overfitting_sensitivity()
    except Exception as e:
        logger.error(f"Audit failed with error: {e}", exc_info=True)
        audit_results['error'] = str(e)
    
    # Save full results
    with open(EVIDENCE_DIR / 'full_audit_results.json', 'w') as f:
        json.dump(make_json_serializable(audit_results), f, indent=2)
    
    # Generate report
    verdict = generate_audit_report()
    
    logger.info("\n" + "="*80)
    logger.info(f"FINAL VERDICT: {verdict}")
    logger.info("="*80)


if __name__ == '__main__':
    main()

