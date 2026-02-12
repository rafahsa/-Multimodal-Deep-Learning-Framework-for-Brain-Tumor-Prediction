#!/usr/bin/env python3
"""
Extract AUC-ROC for nested CV ensemble with meta-features.

This script computes AUC-ROC from the nested CV meta-features experiment
by re-running the evaluation to get probabilities, or by checking if
probabilities were saved.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score
import sys

# Paths
MERGED_OOF_FILE = Path('ensemble/oof_predictions/merged_oof_predictions.csv')
RESULTS_FILE = Path('ensemble/results/nested_cv_meta_features/meta_features_results_20260209_005859.json')
NESTED_CV_SCRIPT = Path('scripts/ensemble/nested_cv_meta_features.py')

def check_if_auc_exists():
    """Check if AUC is already computed in results file."""
    if not RESULTS_FILE.exists():
        print(f"❌ Results file not found: {RESULTS_FILE}")
        return None
    
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)
    
    # Check for AUC at different levels
    if 'auc_mean' in results:
        return {
            'source': 'results_file',
            'key': 'auc_mean',
            'value': results['auc_mean'],
            'type': 'mean_fold_auc'
        }
    
    if 'auc_roc' in results:
        return {
            'source': 'results_file',
            'key': 'auc_roc',
            'value': results['auc_roc'],
            'type': 'overall_auc'
        }
    
    # Check per-fold results
    if 'fold_results' in results:
        fold_aucs = []
        for fold_result in results['fold_results']:
            if 'auc' in fold_result or 'auc_roc' in fold_result:
                auc_key = 'auc' if 'auc' in fold_result else 'auc_roc'
                fold_aucs.append(fold_result[auc_key])
        
        if fold_aucs:
            return {
                'source': 'results_file',
                'key': 'fold_results[].auc',
                'value': np.mean(fold_aucs),
                'std': np.std(fold_aucs),
                'per_fold': fold_aucs,
                'type': 'mean_fold_auc'
            }
    
    return None

def compute_auc_from_nested_cv():
    """
    Compute AUC by re-running nested CV evaluation.
    This requires the script to save probabilities or we need to modify it.
    """
    print("⚠️  AUC not found in results file.")
    print("⚠️  The nested_cv_meta_features.py script does not compute AUC.")
    print("⚠️  Need to either:")
    print("    1. Modify script to compute and save AUC")
    print("    2. Check if probabilities were saved elsewhere")
    print("    3. Use baseline ensemble AUC if it matches the protocol")
    return None

def check_baseline_ensemble_auc():
    """Check baseline ensemble AUC for comparison."""
    baseline_file = Path('ensemble/results/meta_learner_metrics.json')
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        
        if 'auc_roc' in baseline:
            return {
                'source': 'baseline_ensemble',
                'file': str(baseline_file),
                'key': 'auc_roc',
                'value': baseline['auc_roc'],
                'type': 'baseline_ensemble_auc',
                'note': 'This is baseline ensemble (no nested CV structure, no meta-features)'
            }
    
    return None

def main():
    print("="*80)
    print("EXTRACTING AUC-ROC FOR NESTED CV ENSEMBLE WITH META-FEATURES")
    print("="*80)
    
    # Check if AUC exists in results
    print("\n1. Checking if AUC exists in results file...")
    auc_result = check_if_auc_exists()
    
    if auc_result:
        print(f"✅ Found AUC in results file!")
        print(f"   Source: {auc_result['source']}")
        print(f"   Key: {auc_result['key']}")
        print(f"   Value: {auc_result['value']}")
        print(f"   Type: {auc_result['type']}")
        if 'std' in auc_result:
            print(f"   Std: {auc_result['std']}")
        if 'per_fold' in auc_result:
            print(f"   Per-fold: {auc_result['per_fold']}")
        return auc_result
    
    # Check baseline for comparison
    print("\n2. Checking baseline ensemble AUC for reference...")
    baseline_auc = check_baseline_ensemble_auc()
    if baseline_auc:
        print(f"⚠️  Found baseline AUC (NOT the nested CV ensemble):")
        print(f"   File: {baseline_auc['file']}")
        print(f"   Key: {baseline_auc['key']}")
        print(f"   Value: {baseline_auc['value']}")
        print(f"   Note: {baseline_auc['note']}")
    
    # Try to compute
    print("\n3. Attempting to compute AUC...")
    computed_auc = compute_auc_from_nested_cv()
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("❌ AUC-ROC is NOT computed in the nested CV meta-features results.")
    print("   The script focuses on threshold-based metrics (FN, FP, recall, precision).")
    print("\n📋 RECOMMENDATION:")
    print("   Option 1: Modify nested_cv_meta_features.py to compute AUC per fold")
    print("   Option 2: Use baseline ensemble AUC (0.9074) if protocol matches")
    print("   Option 3: Compute AUC from saved probabilities if they exist")
    
    return None

if __name__ == '__main__':
    main()

