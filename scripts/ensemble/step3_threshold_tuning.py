"""
Step 3: Threshold Tuning for Ensemble

Performs threshold sweep on OOF predictions to optimize for recall.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
)

logger = logging.getLogger(__name__)


def tune_ensemble_thresholds(df: pd.DataFrame, output_dir: Path) -> Dict:
    """Tune ensemble thresholds for recall targets."""
    logger.info("Tuning ensemble thresholds...")
    
    # Train meta-learner on calibrated probabilities
    feature_cols = ['swin_prob_cal', 'resnet_prob_cal', 'mil_prob_cal']
    if not all(col in df.columns for col in feature_cols):
        logger.error("Missing required probability columns")
        return {}
    
    X = df[feature_cols].values
    y = df['label'].values
    
    # Nested-CV threshold tuning
    fold_results = []
    
    for test_fold in range(5):
        # Train on other folds
        train_mask = df['fold'] != test_fold
        test_mask = df['fold'] == test_fold
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        # Train meta-learner
        meta_learner = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        meta_learner.fit(X_train, y_train)
        
        # Get probabilities
        y_proba = meta_learner.predict_proba(X_test)[:, 1]
        
        # Threshold sweep
        thresholds = np.arange(0.01, 0.99, 0.01)
        best_threshold_85 = None
        best_threshold_90 = None
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            recall = recall_score(y_test, y_pred, zero_division=0)
            
            if recall >= 0.85 and best_threshold_85 is None:
                best_threshold_85 = threshold
            if recall >= 0.90 and best_threshold_90 is None:
                best_threshold_90 = threshold
            
            if best_threshold_85 and best_threshold_90:
                break
        
        # Evaluate at both thresholds
        for recall_target, best_threshold in [(0.85, best_threshold_85), (0.90, best_threshold_90)]:
            if best_threshold is None:
                continue
            
            y_pred = (y_proba >= best_threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            fold_results.append({
                'fold': test_fold,
                'recall_target': recall_target,
                'threshold': best_threshold,
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'f1': float(f1_score(y_test, y_pred, zero_division=0)),
                'auc': float(roc_auc_score(y_test, y_proba))
            })
    
    # Save results
    results_df = pd.DataFrame(fold_results)
    output_file = output_dir / 'threshold_tuning_results.csv'
    results_df.to_csv(output_file, index=False)
    logger.info(f"✓ Saved threshold tuning results to: {output_file}")
    
    # Summary
    logger.info("\nThreshold Tuning Summary:")
    for recall_target in [0.85, 0.90]:
        target_results = results_df[results_df['recall_target'] == recall_target]
        if len(target_results) > 0:
            logger.info(f"\nRecall >= {recall_target}:")
            logger.info(f"  Mean threshold: {target_results['threshold'].mean():.3f} ± {target_results['threshold'].std():.3f}")
            logger.info(f"  Mean FN: {target_results['fn'].mean():.2f} ± {target_results['fn'].std():.2f}")
            logger.info(f"  Mean FP: {target_results['fp'].mean():.2f} ± {target_results['fp'].std():.2f}")
            logger.info(f"  Mean Recall: {target_results['recall'].mean():.4f} ± {target_results['recall'].std():.4f}")
            logger.info(f"  Mean Precision: {target_results['precision'].mean():.4f} ± {target_results['precision'].std():.4f}")
    
    return {'results_df': results_df, 'fold_results': fold_results}


