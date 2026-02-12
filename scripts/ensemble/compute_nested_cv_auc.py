#!/usr/bin/env python3
"""
Compute AUC-ROC for nested CV ensemble with meta-features.

This script re-runs the nested CV evaluation to compute AUC-ROC
which was not originally computed in the nested_cv_meta_features.py script.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
MERGED_OOF_FILE = Path('ensemble/oof_predictions/merged_oof_predictions.csv')
OUTPUT_FILE = Path('ensemble/results/nested_cv_meta_features/auc_roc_computed.json')
RESULTS_FILE = Path('ensemble/results/nested_cv_meta_features/meta_features_results_20260209_005859.json')

# Configuration (must match nested_cv_meta_features.py)
OUTER_CV_FOLDS = 5
RANDOM_SEED = 42
BASE_PROB_COLS = ['hgg_prob_resnet', 'hgg_prob_swin', 'mil_prob']
TARGET_COLUMN = 'label'
PATIENT_ID_COLUMN = 'patient_id'

def engineer_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer meta-features (same as nested_cv_meta_features.py)."""
    base_probs = df[BASE_PROB_COLS].values
    
    features = {}
    features['p_resnet'] = base_probs[:, 0]
    features['p_swin'] = base_probs[:, 1]
    features['p_mil'] = base_probs[:, 2]
    features['prob_mean'] = np.mean(base_probs, axis=1)
    features['prob_std'] = np.std(base_probs, axis=1)
    features['prob_max'] = np.max(base_probs, axis=1)
    features['prob_min'] = np.min(base_probs, axis=1)
    features['prob_range'] = features['prob_max'] - features['prob_min']
    features['margin_mean'] = np.abs(features['prob_mean'] - 0.5)
    features['margin_max'] = np.max(np.abs(base_probs - 0.5), axis=1)
    
    prob_mean = features['prob_mean']
    prob_mean_clipped = np.clip(prob_mean, 1e-7, 1 - 1e-7)
    features['entropy_mean'] = -(prob_mean_clipped * np.log2(prob_mean_clipped) + 
                                 (1 - prob_mean_clipped) * np.log2(1 - prob_mean_clipped))
    
    argmax_idx = np.argmax(base_probs, axis=1)
    features['argmax_resnet'] = (argmax_idx == 0).astype(float)
    features['argmax_swin'] = (argmax_idx == 1).astype(float)
    features['argmax_mil'] = (argmax_idx == 2).astype(float)
    
    meta_features_df = pd.DataFrame(features)
    meta_features_df[PATIENT_ID_COLUMN] = df[PATIENT_ID_COLUMN].values
    meta_features_df[TARGET_COLUMN] = df[TARGET_COLUMN].values
    
    return meta_features_df

def apply_platt_calibration(meta_learner, X_cal, y_cal, X_eval):
    """Apply Platt calibration (simplified version)."""
    from sklearn.linear_model import LogisticRegression as PlattScaling
    
    y_proba_cal_uncal = meta_learner.predict_proba(X_cal)[:, 1]
    y_proba_eval_uncal = meta_learner.predict_proba(X_eval)[:, 1]
    
    y_proba_cal_clipped = np.clip(y_proba_cal_uncal, 1e-7, 1 - 1e-7)
    log_odds_cal = np.log(y_proba_cal_clipped / (1 - y_proba_cal_clipped))
    
    platt_model = PlattScaling()
    platt_model.fit(log_odds_cal.reshape(-1, 1), y_cal)
    
    y_proba_eval_clipped = np.clip(y_proba_eval_uncal, 1e-7, 1 - 1e-7)
    log_odds_eval = np.log(y_proba_eval_clipped / (1 - y_proba_eval_clipped))
    y_proba_eval_cal = platt_model.predict_proba(log_odds_eval.reshape(-1, 1))[:, 1]
    
    return y_proba_eval_cal

def main():
    logger.info("="*80)
    logger.info("COMPUTING AUC-ROC FOR NESTED CV ENSEMBLE WITH META-FEATURES")
    logger.info("="*80)
    
    # Load data
    df = pd.read_csv(MERGED_OOF_FILE)
    logger.info(f"Loaded {len(df)} patients")
    
    # Engineer meta-features
    df_meta = engineer_meta_features(df)
    feature_cols = [col for col in df_meta.columns 
                    if col not in [PATIENT_ID_COLUMN, TARGET_COLUMN]]
    X = df_meta[feature_cols].values
    y = df_meta[TARGET_COLUMN].values
    
    logger.info(f"Features: {len(feature_cols)}")
    
    # Nested CV
    outer_cv = StratifiedKFold(n_splits=OUTER_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    fold_aucs = []
    all_y_true = []
    all_y_proba = []
    
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X, y)):
        logger.info(f"\nProcessing fold {fold_idx + 1}/{OUTER_CV_FOLDS}...")
        
        # Split
        X_outer_train = X[outer_train_idx]
        y_outer_train = y[outer_train_idx]
        X_outer_test = X[outer_test_idx]
        y_outer_test = y[outer_test_idx]
        
        # Train meta-learner
        meta_learner = LogisticRegression(
            class_weight='balanced',
            solver='lbfgs',
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_SEED
        )
        meta_learner.fit(X_outer_train, y_outer_train)
        
        # Apply calibration (use 70% of outer_train for calibration)
        from sklearn.model_selection import train_test_split
        X_cal, X_thr, y_cal, y_thr = train_test_split(
            X_outer_train, y_outer_train,
            test_size=0.3, random_state=RANDOM_SEED, stratify=y_outer_train
        )
        
        # Calibrate test probabilities
        y_proba_test_cal = apply_platt_calibration(
            meta_learner, X_cal, y_cal, X_outer_test
        )
        
        # Compute AUC for this fold
        fold_auc = roc_auc_score(y_outer_test, y_proba_test_cal)
        fold_aucs.append(fold_auc)
        all_y_true.extend(y_outer_test)
        all_y_proba.extend(y_proba_test_cal)
        
        logger.info(f"  Fold {fold_idx + 1} AUC: {fold_auc:.4f}")
    
    # Compute overall AUC
    overall_auc = roc_auc_score(all_y_true, all_y_proba)
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    
    results = {
        'overall_auc_roc': float(overall_auc),
        'mean_fold_auc_roc': float(mean_auc),
        'std_fold_auc_roc': float(std_auc),
        'per_fold_auc_roc': [float(auc) for auc in fold_aucs],
        'n_folds': len(fold_aucs),
        'note': 'AUC computed from nested CV with meta-features and Platt calibration'
    }
    
    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80)
    logger.info(f"Overall AUC-ROC: {overall_auc:.4f}")
    logger.info(f"Mean fold AUC-ROC: {mean_auc:.4f} ± {std_auc:.4f}")
    logger.info(f"Per-fold AUC-ROC: {[f'{auc:.4f}' for auc in fold_aucs]}")
    
    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Results saved to: {OUTPUT_FILE}")
    
    return results

if __name__ == '__main__':
    main()

