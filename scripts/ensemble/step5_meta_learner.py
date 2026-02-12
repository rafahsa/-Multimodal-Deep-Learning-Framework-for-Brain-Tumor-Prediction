"""
Step 5: Meta-Learner Retraining

Retrains meta-learner with calibrated probabilities and non-DL features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available, skipping XGBoost model")

logger = logging.getLogger(__name__)


def retrain_meta_learner_with_features(df: pd.DataFrame, output_dir: Path) -> Dict:
    """Retrain meta-learner with new features."""
    logger.info("Retraining meta-learner...")
    
    # Base probability features
    prob_cols = ['swin_prob_cal', 'resnet_prob_cal', 'mil_prob_cal']
    
    # Non-DL features (exclude patient_id, fold, label)
    feature_cols = [col for col in df.columns 
                    if col not in ['patient_id', 'fold', 'label', 'hgg_prob_resnet', 
                                   'hgg_prob_swin', 'mil_prob', 'swin_prob_tta', 'resnet_prob_tta']]
    
    # Combine features
    all_feature_cols = prob_cols + feature_cols
    
    # Remove any missing columns
    available_cols = [col for col in all_feature_cols if col in df.columns]
    
    X = df[available_cols].values
    y = df['label'].values
    
    # Nested-CV evaluation
    results = {}
    
    models_to_try = [('LogisticRegression', LogisticRegression)]
    if XGBOOST_AVAILABLE:
        models_to_try.append(('XGBoost', xgb.XGBClassifier))
    
    for model_name, model_class in models_to_try:
        logger.info(f"\nEvaluating {model_name}...")
        
        fold_results = []
        
        for test_fold in range(5):
            train_mask = df['fold'] != test_fold
            test_mask = df['fold'] == test_fold
            
            X_train = X[train_mask]
            y_train = y[train_mask]
            X_test = X[test_mask]
            y_test = y[test_mask]
            
            # Train model
            if model_name == 'LogisticRegression':
                model = model_class(class_weight='balanced', max_iter=1000, random_state=42)
            else:
                model = model_class(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    eval_metric='logloss'
                )
            
            model.fit(X_train, y_train)
            
            # Predict
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            
            # Metrics
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            fold_results.append({
                'fold': test_fold,
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'f1': float(f1_score(y_test, y_pred, zero_division=0)),
                'auc': float(roc_auc_score(y_test, y_proba))
            })
        
        # Aggregate
        results[model_name] = {
            'fold_results': fold_results,
            'fn_mean': np.mean([r['fn'] for r in fold_results]),
            'fn_std': np.std([r['fn'] for r in fold_results]),
            'fp_mean': np.mean([r['fp'] for r in fold_results]),
            'fp_std': np.std([r['fp'] for r in fold_results]),
            'recall_mean': np.mean([r['recall'] for r in fold_results]),
            'recall_std': np.std([r['recall'] for r in fold_results]),
            'precision_mean': np.mean([r['precision'] for r in fold_results]),
            'precision_std': np.std([r['precision'] for r in fold_results]),
            'auc_mean': np.mean([r['auc'] for r in fold_results]),
            'auc_std': np.std([r['auc'] for r in fold_results])
        }
        
        logger.info(f"  FN: {results[model_name]['fn_mean']:.2f} ± {results[model_name]['fn_std']:.2f}")
        logger.info(f"  FP: {results[model_name]['fp_mean']:.2f} ± {results[model_name]['fp_std']:.2f}")
        logger.info(f"  Recall: {results[model_name]['recall_mean']:.4f} ± {results[model_name]['recall_std']:.4f}")
        logger.info(f"  Precision: {results[model_name]['precision_mean']:.4f} ± {results[model_name]['precision_std']:.4f}")
        logger.info(f"  AUC: {results[model_name]['auc_mean']:.4f} ± {results[model_name]['auc_std']:.4f}")
    
    # Save results
    import json
    output_file = output_dir / 'meta_learner_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Saved meta-learner results to: {output_file}")
    
    return results

