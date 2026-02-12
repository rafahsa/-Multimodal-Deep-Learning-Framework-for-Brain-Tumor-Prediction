"""
Step 6: Ablation Study

Reports metrics for all configurations to determine what helped.
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

logger = logging.getLogger(__name__)


def run_ablation_study(
    df_baseline: pd.DataFrame,
    df_tta: pd.DataFrame,
    df_cal: pd.DataFrame,
    df_features: pd.DataFrame,
    threshold_results: Dict,
    meta_learner_results: Dict,
    output_dir: Path
) -> Dict:
    """Run comprehensive ablation study."""
    logger.info("Running ablation study...")
    
    results = {}
    
    # Configuration 1: Baseline ensemble
    logger.info("\n1. Baseline Ensemble")
    results['baseline'] = evaluate_configuration(
        df_baseline,
        prob_cols=['hgg_prob_resnet', 'hgg_prob_swin', 'mil_prob']
    )
    
    # Configuration 2: + TTA
    logger.info("\n2. + TTA")
    results['tta'] = evaluate_configuration(
        df_tta,
        prob_cols=['resnet_prob_tta', 'swin_prob_tta', 'mil_prob']
    )
    
    # Configuration 3: + TTA + Calibration
    logger.info("\n3. + TTA + Calibration")
    results['tta_cal'] = evaluate_configuration(
        df_cal,
        prob_cols=['resnet_prob_cal', 'swin_prob_cal', 'mil_prob_cal']
    )
    
    # Configuration 4: + TTA + Calibration + Threshold tuning
    logger.info("\n4. + TTA + Calibration + Threshold Tuning")
    # Use threshold results from step 3
    if threshold_results:
        results['tta_cal_thresh'] = aggregate_threshold_results(threshold_results)
    
    # Configuration 5: + All above + Non-DL features
    logger.info("\n5. + All above + Non-DL Features")
    if meta_learner_results:
        # Use best model from step 5
        best_model = min(meta_learner_results.items(), 
                        key=lambda x: x[1]['fn_mean'])
        results['all_features'] = {
            'model': best_model[0],
            'fn_mean': best_model[1]['fn_mean'],
            'fn_std': best_model[1]['fn_std'],
            'fp_mean': best_model[1]['fp_mean'],
            'fp_std': best_model[1]['fp_std'],
            'recall_mean': best_model[1]['recall_mean'],
            'recall_std': best_model[1]['recall_std'],
            'precision_mean': best_model[1]['precision_mean'],
            'precision_std': best_model[1]['precision_std'],
            'auc_mean': best_model[1]['auc_mean'],
            'auc_std': best_model[1]['auc_std']
        }
    
    # Save results
    import json
    output_file = output_dir / 'ablation_study_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Saved ablation study to: {output_file}")
    
    # Print summary table
    logger.info("\n" + "="*80)
    logger.info("ABLATION STUDY SUMMARY")
    logger.info("="*80)
    logger.info(f"{'Configuration':<30} {'FN':<12} {'FP':<12} {'Recall':<12} {'Precision':<12} {'AUC':<12}")
    logger.info("-"*80)
    
    for config_name, metrics in results.items():
        fn_str = f"{metrics.get('fn_mean', 0):.2f}±{metrics.get('fn_std', 0):.2f}"
        fp_str = f"{metrics.get('fp_mean', 0):.2f}±{metrics.get('fp_std', 0):.2f}"
        recall_str = f"{metrics.get('recall_mean', 0):.4f}±{metrics.get('recall_std', 0):.4f}"
        precision_str = f"{metrics.get('precision_mean', 0):.4f}±{metrics.get('precision_std', 0):.4f}"
        auc_str = f"{metrics.get('auc_mean', 0):.4f}±{metrics.get('auc_std', 0):.4f}"
        logger.info(f"{config_name:<30} {fn_str:<12} {fp_str:<12} {recall_str:<12} {precision_str:<12} {auc_str:<12}")
    
    return results


def evaluate_configuration(df: pd.DataFrame, prob_cols: list) -> Dict:
    """Evaluate a configuration using nested-CV."""
    if not all(col in df.columns for col in prob_cols):
        return {}
    
    X = df[prob_cols].values
    y = df['label'].values
    
    fold_results = []
    
    for test_fold in range(5):
        train_mask = df['fold'] != test_fold
        test_mask = df['fold'] == test_fold
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        # Train meta-learner
        model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        
        # Metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        fold_results.append({
            'fn': int(fn), 'fp': int(fp),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'auc': float(roc_auc_score(y_test, y_proba))
        })
    
    return {
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


def aggregate_threshold_results(threshold_results: Dict) -> Dict:
    """Aggregate threshold tuning results."""
    if 'results_df' not in threshold_results:
        return {}
    
    results_df = threshold_results['results_df']
    
    # Use recall >= 0.85 results
    target_results = results_df[results_df['recall_target'] == 0.85]
    
    if len(target_results) == 0:
        return {}
    
    return {
        'fn_mean': float(target_results['fn'].mean()),
        'fn_std': float(target_results['fn'].std()),
        'fp_mean': float(target_results['fp'].mean()),
        'fp_std': float(target_results['fp'].std()),
        'recall_mean': float(target_results['recall'].mean()),
        'recall_std': float(target_results['recall'].std()),
        'precision_mean': float(target_results['precision'].mean()),
        'precision_std': float(target_results['precision'].std()),
        'auc_mean': float(target_results['auc'].mean()),
        'auc_std': float(target_results['auc'].std())
    }


