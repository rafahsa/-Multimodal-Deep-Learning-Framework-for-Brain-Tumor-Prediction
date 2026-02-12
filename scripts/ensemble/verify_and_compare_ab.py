#!/usr/bin/env python3
"""
Verify Fairness and Perform A/B Comparison: Baseline vs ROI-MIL Ensemble

This script:
1. Verifies that baseline and ROI-MIL ensembles use identical data (except mil_prob)
2. Performs clean A/B comparison at threshold 0.22
3. Reports per-fold results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Paths
BASELINE_OOF = Path('ensemble/oof_predictions/merged_oof_predictions.csv')
ROI_MIL_OOF = Path('ensemble/oof_predictions/merged_oof_predictions_roi_mil.csv')
OUTPUT_DIR = Path('ensemble/results/ab_comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.22

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load both OOF prediction files."""
    baseline = pd.read_csv(BASELINE_OOF)
    roi_mil = pd.read_csv(ROI_MIL_OOF)
    
    print(f"Baseline: {len(baseline)} samples, {len(baseline.columns)} columns")
    print(f"ROI MIL: {len(roi_mil)} samples, {len(roi_mil.columns)} columns")
    
    return baseline, roi_mil

def verify_fairness(baseline: pd.DataFrame, roi_mil: pd.DataFrame) -> Dict:
    """Verify that A and B are fair for comparison."""
    print("\n" + "="*80)
    print("FAIRNESS VERIFICATION")
    print("="*80)
    
    results = {}
    
    # 1. Same patient_id set
    baseline_ids = set(baseline['patient_id'].values)
    roi_mil_ids = set(roi_mil['patient_id'].values)
    same_patients = baseline_ids == roi_mil_ids
    results['same_patients'] = same_patients
    print(f"✓ Same patient IDs: {same_patients} ({len(baseline_ids)} patients)")
    
    if not same_patients:
        missing_in_roi = baseline_ids - roi_mil_ids
        missing_in_baseline = roi_mil_ids - baseline_ids
        if missing_in_roi:
            print(f"  Missing in ROI: {missing_in_roi}")
        if missing_in_baseline:
            print(f"  Missing in baseline: {missing_in_baseline}")
    
    # 2. Same labels
    baseline_sorted = baseline.sort_values('patient_id')
    roi_mil_sorted = roi_mil.sort_values('patient_id')
    
    labels_match = np.array_equal(
        baseline_sorted['label'].values,
        roi_mil_sorted['label'].values
    )
    results['same_labels'] = labels_match
    print(f"✓ Same labels: {labels_match}")
    
    if not labels_match:
        diff_mask = baseline_sorted['label'].values != roi_mil_sorted['label'].values
        print(f"  Mismatched labels: {diff_mask.sum()} patients")
    
    # 3. Same fold assignments
    folds_match = np.array_equal(
        baseline_sorted['fold'].values,
        roi_mil_sorted['fold'].values
    )
    results['same_folds'] = folds_match
    print(f"✓ Same fold assignments: {folds_match}")
    
    # 4. Same ResNet probabilities
    resnet_diff = np.abs(
        baseline_sorted['hgg_prob_resnet'].values - 
        roi_mil_sorted['hgg_prob_resnet'].values
    )
    max_resnet_diff = resnet_diff.max()
    resnet_match = max_resnet_diff < 1e-10
    results['same_resnet'] = resnet_match
    results['max_resnet_diff'] = max_resnet_diff
    print(f"✓ Same ResNet probabilities: {resnet_match} (max diff: {max_resnet_diff:.2e})")
    
    # 5. Same Swin probabilities
    swin_diff = np.abs(
        baseline_sorted['hgg_prob_swin'].values - 
        roi_mil_sorted['hgg_prob_swin'].values
    )
    max_swin_diff = swin_diff.max()
    swin_match = max_swin_diff < 1e-10
    results['same_swin'] = swin_match
    results['max_swin_diff'] = max_swin_diff
    print(f"✓ Same Swin probabilities: {swin_match} (max diff: {max_swin_diff:.2e})")
    
    # 6. Different MIL probabilities (this is expected)
    mil_diff = np.abs(
        baseline_sorted['mil_prob'].values - 
        roi_mil_sorted['mil_prob'].values
    )
    max_mil_diff = mil_diff.max()
    mean_mil_diff = mil_diff.mean()
    mil_different = max_mil_diff > 1e-6
    results['mil_different'] = mil_different
    results['max_mil_diff'] = max_mil_diff
    results['mean_mil_diff'] = mean_mil_diff
    print(f"✓ MIL probabilities differ (expected): {mil_different}")
    print(f"  Max MIL diff: {max_mil_diff:.6f}, Mean MIL diff: {mean_mil_diff:.6f}")
    
    # Summary table
    print("\n" + "-"*80)
    print("VERIFICATION SUMMARY")
    print("-"*80)
    verification_table = pd.DataFrame([
        ['Same patient IDs', 'PASS' if same_patients else 'FAIL', '-'],
        ['Same labels', 'PASS' if labels_match else 'FAIL', '-'],
        ['Same fold assignments', 'PASS' if folds_match else 'FAIL', '-'],
        ['Same ResNet probs', 'PASS' if resnet_match else 'FAIL', f'{max_resnet_diff:.2e}'],
        ['Same Swin probs', 'PASS' if swin_match else 'FAIL', f'{max_swin_diff:.2e}'],
        ['MIL probs differ', 'PASS' if mil_different else 'FAIL', f'{max_mil_diff:.6f}'],
    ], columns=['Check', 'Status', 'Max Absolute Difference'])
    print(verification_table.to_string(index=False))
    
    all_pass = all([
        same_patients, labels_match, folds_match, 
        resnet_match, swin_match, mil_different
    ])
    results['all_checks_pass'] = all_pass
    
    print(f"\n{'='*80}")
    print(f"OVERALL VERIFICATION: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*80}\n")
    
    return results, baseline_sorted, roi_mil_sorted

def compute_metrics(y_true, y_pred, y_proba):
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'auc_roc': auc,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'fn_rate': fn / (fn + tp) if (fn + tp) > 0 else 0.0
    }

def evaluate_ensemble(df: pd.DataFrame, threshold: float) -> Dict:
    """Evaluate ensemble predictions at given threshold."""
    y_true = df['label'].values
    
    # Load trained meta-learner
    # For now, we'll use the OOF predictions directly
    # In production, we'd load the trained model and predict
    # But for A/B comparison, we can use the OOF predictions as proxy
    
    # For this comparison, we need the actual ensemble predictions
    # Let's check if there are saved predictions or we need to load models
    
    # Actually, we should load the trained models and predict
    # But for now, let's assume we have the ensemble probabilities
    # We'll need to load the models and predict
    
    # For A/B comparison, we need to:
    # 1. Load baseline meta-learner model
    # 2. Load ROI-MIL meta-learner model  
    # 3. Predict on the same data
    # 4. Compare at threshold
    
    # Since we're comparing OOF predictions, the models were trained on these
    # So we can't use OOF predictions directly for evaluation
    # We need to load the actual trained models
    
    # For now, let's return metrics per fold
    fold_results = {}
    for fold in sorted(df['fold'].unique()):
        fold_df = df[df['fold'] == fold]
        y_true_fold = fold_df['label'].values
        
        # We need ensemble predictions, not base model predictions
        # This requires loading the trained meta-learner models
        # For now, let's structure the code to load models
        
    return fold_results

def load_models_and_predict(baseline_df: pd.DataFrame, roi_mil_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Load trained meta-learner models and predict."""
    from sklearn.linear_model import LogisticRegression
    import joblib
    
    # Load baseline model
    baseline_model_path = Path('ensemble/models/meta_learner_logistic_regression.joblib')
    if baseline_model_path.exists():
        baseline_model = joblib.load(baseline_model_path)
    else:
        # Train on the fly if model doesn't exist
        print("Baseline model not found, training...")
        X_baseline = baseline_df[['hgg_prob_resnet', 'hgg_prob_swin', 'mil_prob']].values
        y = baseline_df['label'].values
        baseline_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        baseline_model.fit(X_baseline, y)
    
    # Load ROI-MIL model
    roi_mil_model_path = Path('ensemble/models/roi_mil/meta_learner_logistic_regression_roi_mil.joblib')
    if roi_mil_model_path.exists():
        roi_mil_model = joblib.load(roi_mil_model_path)
    else:
        # Train on the fly if model doesn't exist
        print("ROI-MIL model not found, training...")
        X_roi_mil = roi_mil_df[['hgg_prob_resnet', 'hgg_prob_swin', 'mil_prob']].values
        y = roi_mil_df['label'].values
        roi_mil_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        roi_mil_model.fit(X_roi_mil, y)
    
    # Predict
    X_baseline = baseline_df[['hgg_prob_resnet', 'hgg_prob_swin', 'mil_prob']].values
    X_roi_mil = roi_mil_df[['hgg_prob_resnet', 'hgg_prob_swin', 'mil_prob']].values
    
    baseline_proba = baseline_model.predict_proba(X_baseline)[:, 1]
    roi_mil_proba = roi_mil_model.predict_proba(X_roi_mil)[:, 1]
    
    return baseline_proba, roi_mil_proba

def compare_ab(baseline_df: pd.DataFrame, roi_mil_df: pd.DataFrame, threshold: float) -> Dict:
    """Perform A/B comparison."""
    print("\n" + "="*80)
    print("A/B COMPARISON")
    print("="*80)
    print(f"Threshold: {threshold}")
    
    # Load models and predict
    baseline_proba, roi_mil_proba = load_models_and_predict(baseline_df, roi_mil_df)
    
    y_true = baseline_df['label'].values
    
    # Predictions at threshold
    baseline_pred = (baseline_proba >= threshold).astype(int)
    roi_mil_pred = (roi_mil_proba >= threshold).astype(int)
    
    # Overall metrics
    baseline_metrics = compute_metrics(y_true, baseline_pred, baseline_proba)
    roi_mil_metrics = compute_metrics(y_true, roi_mil_pred, roi_mil_proba)
    
    print("\n" + "-"*80)
    print("OVERALL METRICS")
    print("-"*80)
    comparison_df = pd.DataFrame({
        'Metric': ['AUC-ROC', 'HGG Recall', 'FN Count', 'FN Rate', 'Precision', 'Accuracy'],
        'Baseline': [
            baseline_metrics['auc_roc'],
            baseline_metrics['recall'],
            baseline_metrics['fn'],
            baseline_metrics['fn_rate'],
            baseline_metrics['precision'],
            baseline_metrics['accuracy']
        ],
        'ROI-MIL': [
            roi_mil_metrics['auc_roc'],
            roi_mil_metrics['recall'],
            roi_mil_metrics['fn'],
            roi_mil_metrics['fn_rate'],
            roi_mil_metrics['precision'],
            roi_mil_metrics['accuracy']
        ],
        'Difference': [
            roi_mil_metrics['auc_roc'] - baseline_metrics['auc_roc'],
            roi_mil_metrics['recall'] - baseline_metrics['recall'],
            roi_mil_metrics['fn'] - baseline_metrics['fn'],
            roi_mil_metrics['fn_rate'] - baseline_metrics['fn_rate'],
            roi_mil_metrics['precision'] - baseline_metrics['precision'],
            roi_mil_metrics['accuracy'] - baseline_metrics['accuracy']
        ]
    })
    print(comparison_df.to_string(index=False))
    
    # Per-fold metrics
    print("\n" + "-"*80)
    print("PER-FOLD METRICS")
    print("-"*80)
    
    fold_results = {}
    for fold in sorted(baseline_df['fold'].unique()):
        fold_mask = baseline_df['fold'] == fold
        y_true_fold = y_true[fold_mask]
        baseline_proba_fold = baseline_proba[fold_mask]
        roi_mil_proba_fold = roi_mil_proba[fold_mask]
        
        baseline_pred_fold = (baseline_proba_fold >= threshold).astype(int)
        roi_mil_pred_fold = (roi_mil_proba_fold >= threshold).astype(int)
        
        baseline_metrics_fold = compute_metrics(y_true_fold, baseline_pred_fold, baseline_proba_fold)
        roi_mil_metrics_fold = compute_metrics(y_true_fold, roi_mil_pred_fold, roi_mil_proba_fold)
        
        fold_results[fold] = {
            'baseline': baseline_metrics_fold,
            'roi_mil': roi_mil_metrics_fold
        }
        
        print(f"\nFold {fold}:")
        print(f"  Baseline: AUC={baseline_metrics_fold['auc_roc']:.4f}, "
              f"Recall={baseline_metrics_fold['recall']:.4f}, "
              f"FN={baseline_metrics_fold['fn']}, "
              f"FN Rate={baseline_metrics_fold['fn_rate']:.4f}")
        print(f"  ROI-MIL:  AUC={roi_mil_metrics_fold['auc_roc']:.4f}, "
              f"Recall={roi_mil_metrics_fold['recall']:.4f}, "
              f"FN={roi_mil_metrics_fold['fn']}, "
              f"FN Rate={roi_mil_metrics_fold['fn_rate']:.4f}")
        print(f"  Δ:         AUC={roi_mil_metrics_fold['auc_roc'] - baseline_metrics_fold['auc_roc']:+.4f}, "
              f"Recall={roi_mil_metrics_fold['recall'] - baseline_metrics_fold['recall']:+.4f}, "
              f"FN={roi_mil_metrics_fold['fn'] - baseline_metrics_fold['fn']:+d}")
    
    results = {
        'threshold': threshold,
        'overall': {
            'baseline': baseline_metrics,
            'roi_mil': roi_mil_metrics
        },
        'per_fold': fold_results
    }
    
    return results

def main():
    """Main function."""
    print("="*80)
    print("A/B COMPARISON: BASELINE vs ROI-MIL ENSEMBLE")
    print("="*80)
    
    # Load data
    baseline_df, roi_mil_df = load_data()
    
    # Verify fairness
    verification_results, baseline_sorted, roi_mil_sorted = verify_fairness(baseline_df, roi_mil_df)
    
    if not verification_results['all_checks_pass']:
        print("WARNING: Fairness checks failed. Proceeding with comparison anyway...")
    
    # A/B comparison
    comparison_results = compare_ab(baseline_sorted, roi_mil_sorted, THRESHOLD)
    
    # Save results (convert int64 keys to strings)
    output_file = OUTPUT_DIR / 'ab_comparison_results.json'
    comparison_results_serializable = {
        'threshold': comparison_results['threshold'],
        'overall': comparison_results['overall'],
        'per_fold': {str(k): v for k, v in comparison_results['per_fold'].items()}
    }
    with open(output_file, 'w') as f:
        json.dump({
            'verification': verification_results,
            'comparison': comparison_results_serializable
        }, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    # Final conclusion
    print("\n" + "="*80)
    print("FINAL CONCLUSION")
    print("="*80)
    
    baseline_fn = comparison_results['overall']['baseline']['fn']
    roi_mil_fn = comparison_results['overall']['roi_mil']['fn']
    baseline_recall = comparison_results['overall']['baseline']['recall']
    roi_mil_recall = comparison_results['overall']['roi_mil']['recall']
    
    if roi_mil_fn > baseline_fn or roi_mil_recall < baseline_recall:
        print("❌ DO NOT REPLACE: ROI-MIL ensemble shows degradation")
        print(f"   Evidence: FN increased from {baseline_fn} to {roi_mil_fn} (+{roi_mil_fn - baseline_fn})")
        print(f"            Recall decreased from {baseline_recall:.4f} to {roi_mil_recall:.4f} ({roi_mil_recall - baseline_recall:+.4f})")
        
        # Find worst fold
        worst_fold = None
        worst_fn_diff = 0
        for fold, fold_data in comparison_results['per_fold'].items():
            fn_diff = fold_data['roi_mil']['fn'] - fold_data['baseline']['fn']
            if fn_diff > worst_fn_diff:
                worst_fn_diff = fn_diff
                worst_fold = fold
        
        if worst_fold is not None:
            print(f"   Worst regression: Fold {worst_fold} (FN +{worst_fn_diff})")
    else:
        print("✅ REPLACE: ROI-MIL ensemble shows improvement or no degradation")
        print(f"   FN: {baseline_fn} → {roi_mil_fn} ({roi_mil_fn - baseline_fn:+d})")
        print(f"   Recall: {baseline_recall:.4f} → {roi_mil_recall:.4f} ({roi_mil_recall - baseline_recall:+.4f})")

if __name__ == '__main__':
    main()

