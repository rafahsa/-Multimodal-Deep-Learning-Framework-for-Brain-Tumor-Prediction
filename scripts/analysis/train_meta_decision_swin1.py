#!/usr/bin/env python3
"""
Train Lightweight Meta-Decision Layer for Swin-1

This script trains a Logistic Regression meta-decision layer on top of Swin-1
to reduce FN while keeping FP under control.

NO DEEP LEARNING - strictly lightweight post-hoc meta-decision.
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
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FEATURES_FILE = PROJECT_ROOT / 'ensemble' / 'results' / 'meta_decision' / 'meta_features.csv'
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'results' / 'meta_decision'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target evaluation
FN_EXCELLENT = 25
FN_VERY_STRONG = 15
FN_RESEARCH_LEVEL = 10


def convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    This ensures all numpy scalars (int64, float64, etc.) are converted to
    native Python int/float before json.dump, preventing serialization errors.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare features for training.
    
    Features:
    - hgg_prob_swin (from Swin-1)
    - prediction_entropy
    - Tumor volume proxy (T1ce, FLAIR)
    - Intensity variance (T1ce, FLAIR)
    - GLCM texture stats (T1ce, FLAIR)
    """
    # Base features
    feature_cols = ['hgg_prob_swin', 'prediction_entropy']
    
    # Add modality-specific features
    for modality in ['t1ce', 'flair']:
        feature_cols.extend([
            f'{modality}_volume_proxy',
            f'{modality}_intensity_variance',
            f'{modality}_glcm_contrast',
            f'{modality}_glcm_entropy',
            f'{modality}_glcm_homogeneity'
        ])
    
    # Check which features are available
    available_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Missing features: {missing_cols}")
    
    X = df[available_cols].values
    y = df['label'].values
    
    return X, y, available_cols


def train_meta_decision_model(df: pd.DataFrame) -> Dict:
    """
    Train Logistic Regression meta-decision model using nested CV.
    
    For each fold:
    - Train on all other folds
    - Predict on current fold
    """
    folds = sorted(df['fold'].unique())
    
    all_predictions = []
    all_labels = []
    all_patient_ids = []
    all_folds = []
    
    fold_results = []
    
    logger.info("Training meta-decision model using nested CV...")
    
    for fold in folds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold}")
        logger.info(f"{'='*60}")
        
        # Split: train on all other folds, test on current fold
        train_mask = df['fold'] != fold
        test_mask = df['fold'] == fold
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        # Prepare features
        X_train, y_train, feature_names = prepare_features(train_df)
        X_test, y_test, _ = prepare_features(test_df)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Logistic Regression
        model = LogisticRegression(
            class_weight='balanced',  # Handle class imbalance
            max_iter=1000,
            random_state=42,
            C=1.0  # Regularization
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        fold_results.append({
            'fold': int(fold),  # Ensure fold is native Python int (not np.int64)
            'fn': int(fn),
            'fp': int(fp),
            'tn': int(tn),
            'tp': int(tp),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc)
        })
        
        logger.info(f"FN: {fn}, FP: {fp}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # Store predictions
        all_predictions.extend(y_proba)
        all_labels.extend(y_test)
        all_patient_ids.extend(test_df['patient_id'].tolist())
        all_folds.extend([fold] * len(y_test))
    
    # Overall metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_pred_binary = (all_predictions >= 0.5).astype(int)
    
    cm_overall = confusion_matrix(all_labels, all_pred_binary)
    tn, fp, fn, tp = cm_overall.ravel()
    
    precision_overall = precision_score(all_labels, all_pred_binary, zero_division=0)
    recall_overall = recall_score(all_labels, all_pred_binary, zero_division=0)
    f1_overall = f1_score(all_labels, all_pred_binary, zero_division=0)
    auc_overall = roc_auc_score(all_labels, all_predictions)
    
    overall_results = {
        'fn': int(fn),
        'fp': int(fp),
        'tn': int(tn),
        'tp': int(tp),
        'precision': float(precision_overall),
        'recall': float(recall_overall),
        'f1': float(f1_overall),
        'auc': float(auc_overall)
    }
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'patient_id': all_patient_ids,
        'fold': all_folds,
        'label': all_labels,
        'hgg_prob_swin': df.loc[df['patient_id'].isin(all_patient_ids), 'hgg_prob_swin'].values,
        'meta_prob': all_predictions,
        'meta_pred': all_pred_binary
    })
    
    predictions_file = OUTPUT_DIR / 'meta_decision_predictions.csv'
    predictions_df.to_csv(predictions_file, index=False)
    logger.info(f"\n✓ Saved predictions to: {predictions_file}")
    
    return {
        'fold_results': fold_results,
        'overall_results': overall_results,
        'feature_names': feature_names
    }


def main():
    logger.info("="*80)
    logger.info("TRAIN META-DECISION LAYER FOR SWIN-1")
    logger.info("="*80)
    
    # Load features
    logger.info(f"\nLoading features from: {FEATURES_FILE}")
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}. Run extract_meta_features_swin1.py first.")
    
    df = pd.read_csv(FEATURES_FILE)
    logger.info(f"Loaded {len(df)} patients with features")
    
    # Load fold information from OOF file
    oof_file = PROJECT_ROOT / 'ensemble' / 'oof_predictions' / 'merged_oof_predictions.csv'
    oof_df = pd.read_csv(oof_file)
    df = df.merge(oof_df[['patient_id', 'fold']], on='patient_id', how='inner')
    
    # Train model
    results = train_meta_decision_model(df)
    
    # Save results
    # Convert numpy types to native Python types for JSON serialization
    results_serializable = convert_numpy_types(results)
    results_file = OUTPUT_DIR / 'meta_decision_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    logger.info(f"\n✓ Saved results to: {results_file}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("OVERALL RESULTS")
    logger.info("="*80)
    overall = results['overall_results']
    logger.info(f"FN: {overall['fn']} (target: <{FN_RESEARCH_LEVEL} research-level, <{FN_VERY_STRONG} very strong, <{FN_EXCELLENT} excellent)")
    logger.info(f"FP: {overall['fp']}")
    logger.info(f"Precision: {overall['precision']:.4f}")
    logger.info(f"Recall: {overall['recall']:.4f}")
    logger.info(f"F1: {overall['f1']:.4f}")
    logger.info(f"AUC: {overall['auc']:.4f}")
    
    # Evaluate targets
    if overall['fn'] < FN_RESEARCH_LEVEL:
        logger.info(f"\n✅ RESEARCH-LEVEL SUCCESS: FN < {FN_RESEARCH_LEVEL}")
    elif overall['fn'] < FN_VERY_STRONG:
        logger.info(f"\n✅ VERY STRONG: FN < {FN_VERY_STRONG}")
    elif overall['fn'] < FN_EXCELLENT:
        logger.info(f"\n✅ EXCELLENT: FN < {FN_EXCELLENT}")
    else:
        logger.info(f"\n⚠️ FN reduction insufficient: FN = {overall['fn']} (target: <{FN_EXCELLENT})")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()

