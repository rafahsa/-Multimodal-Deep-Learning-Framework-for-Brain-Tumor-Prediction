#!/usr/bin/env python3
"""
Train Meta-Decision Model on Uncertain Samples Only

This script trains a Logistic Regression meta-decision model ONLY on uncertain samples,
preserving strict OOF evaluation by fold.

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
    roc_auc_score
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

UNCERTAIN_SAMPLES_FILE = PROJECT_ROOT / 'ensemble' / 'results' / 'hybrid_safety_net' / 'uncertain_samples.csv'
META_FEATURES_FILE = PROJECT_ROOT / 'ensemble' / 'results' / 'meta_decision' / 'meta_features.csv'
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'results' / 'hybrid_safety_net'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
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
    """Prepare features for training (same as before)."""
    feature_cols = ['hgg_prob_swin', 'prediction_entropy']
    
    for modality in ['t1ce', 'flair']:
        feature_cols.extend([
            f'{modality}_volume_proxy',
            f'{modality}_intensity_variance',
            f'{modality}_glcm_contrast',
            f'{modality}_glcm_entropy',
            f'{modality}_glcm_homogeneity'
        ])
    
    available_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Missing features: {missing_cols}")
    
    X = df[available_cols].values
    y = df['label'].values
    
    return X, y, available_cols


def train_uncertain_meta_decision(df: pd.DataFrame) -> Dict:
    """
    Train meta-decision model ONLY on uncertain samples using nested CV.
    
    For each fold:
    - Filter to uncertain samples only
    - Train on uncertain samples from all other folds
    - Predict on uncertain samples from current fold
    """
    folds = sorted(df['fold'].unique())
    
    all_predictions = {}  # patient_id -> meta_prob
    all_labels = {}  # patient_id -> label
    fold_results = []
    
    logger.info("Training meta-decision model on uncertain samples only...")
    
    for fold in folds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold}")
        logger.info(f"{'='*60}")
        
        # Split: train on uncertain samples from all other folds, test on uncertain samples from current fold
        train_mask = (df['fold'] != fold) & (df['uncertainty_status'] == 'uncertain')
        test_mask = (df['fold'] == fold) & (df['uncertainty_status'] == 'uncertain')
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        if len(train_df) == 0:
            logger.warning(f"No uncertain samples in training folds for fold {fold}, skipping")
            continue
        
        if len(test_df) == 0:
            logger.warning(f"No uncertain samples in test fold {fold}, skipping")
            continue
        
        logger.info(f"Training on {len(train_df)} uncertain samples, testing on {len(test_df)} uncertain samples")
        
        # Prepare features
        X_train, y_train, feature_names = prepare_features(train_df)
        X_test, y_test, _ = prepare_features(test_df)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Logistic Regression
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            C=1.0
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Store predictions
        for idx, patient_id in enumerate(test_df['patient_id']):
            all_predictions[patient_id] = float(y_proba[idx])
            all_labels[patient_id] = int(y_test[idx])
        
        # Evaluate on uncertain samples only
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        fold_results.append({
            'fold': int(fold),
            'uncertain_samples': len(test_df),
            'fn': int(fn),
            'fp': int(fp),
            'tn': int(tn),
            'tp': int(tp),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc)
        })
        
        logger.info(f"Uncertain samples only - FN: {fn}, FP: {fp}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    return {
        'fold_results': fold_results,
        'meta_predictions': all_predictions,
        'feature_names': feature_names
    }


def main():
    logger.info("="*80)
    logger.info("TRAIN META-DECISION ON UNCERTAIN SAMPLES ONLY")
    logger.info("="*80)
    
    # Load uncertain samples
    logger.info(f"\nLoading uncertain samples from: {UNCERTAIN_SAMPLES_FILE}")
    if not UNCERTAIN_SAMPLES_FILE.exists():
        raise FileNotFoundError(f"Uncertain samples file not found: {UNCERTAIN_SAMPLES_FILE}. Run define_uncertain_samples_swin1.py first.")
    
    uncertain_df = pd.read_csv(UNCERTAIN_SAMPLES_FILE)
    logger.info(f"Loaded {len(uncertain_df)} patients")
    
    # Load meta-features
    logger.info(f"Loading meta-features from: {META_FEATURES_FILE}")
    if not META_FEATURES_FILE.exists():
        raise FileNotFoundError(f"Meta-features file not found: {META_FEATURES_FILE}. Run extract_meta_features_swin1.py first.")
    
    features_df = pd.read_csv(META_FEATURES_FILE)
    
    # Merge
    df = uncertain_df.merge(features_df, on=['patient_id', 'label'], how='inner')
    
    # Train model on uncertain samples only
    results = train_uncertain_meta_decision(df)
    
    # Save results
    results_serializable = convert_numpy_types(results)
    results_file = OUTPUT_DIR / 'uncertain_meta_decision_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    logger.info(f"\n✓ Saved results to: {results_file}")
    
    # Save meta-predictions
    predictions_df = pd.DataFrame({
        'patient_id': list(results['meta_predictions'].keys()),
        'meta_prob': list(results['meta_predictions'].values())
    })
    predictions_file = OUTPUT_DIR / 'uncertain_meta_predictions.csv'
    predictions_df.to_csv(predictions_file, index=False)
    logger.info(f"✓ Saved meta-predictions to: {predictions_file}")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()

