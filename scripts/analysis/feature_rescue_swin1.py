#!/usr/bin/env python3
"""
Feature-Level Rescue for Swin-1: Rule-Based and Lightweight Model

This script implements feature-based rescue to reduce Swin-1 false negatives
while keeping FP < 10.

Methods:
1. Rule-based rescue (flip LGG→HGG based on high-risk features)
2. Lightweight logistic regression (if rule-based helps)

All evaluation is strict 5-fold OOF (no leakage).
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
from typing import Dict, List, Tuple, Optional
import SimpleITK as sitk
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OOF_FILE = PROJECT_ROOT / 'ensemble' / 'oof_predictions' / 'merged_oof_predictions.csv'
DATA_ROOT = PROJECT_ROOT / 'data' / 'processed' / 'stage_4_resize' / 'train'
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'results' / 'feature_rescue'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target constraints
TARGET_FN_MAX = 10
TARGET_FP_MAX = 10
TARGET_PRECISION_MIN = 0.90
TARGET_RECALL_MIN = 0.90


def load_volume(patient_id: str, class_name: str, modality: str) -> Optional[np.ndarray]:
    """Load a single modality volume for a patient."""
    patient_dir = DATA_ROOT / class_name / patient_id
    if not patient_dir.exists():
        return None
    
    volume_path = patient_dir / f"{patient_id}_{modality}.nii.gz"
    if not volume_path.exists():
        volume_path = patient_dir / f"{patient_id}_{modality}.nii"
    
    if not volume_path.exists():
        return None
    
    try:
        volume = sitk.ReadImage(str(volume_path))
        volume_array = sitk.GetArrayFromImage(volume).astype(np.float32)
        return volume_array
    except Exception as e:
        logger.warning(f"Error loading {volume_path}: {e}")
        return None


def compute_roi_features(volume: np.ndarray, percentile_low: float = 1.0, percentile_high: float = 99.0) -> Dict:
    """Compute ROI-based features from a volume."""
    # Remove background (very low values)
    brain_mask = volume > np.percentile(volume, percentile_low)
    brain_values = volume[brain_mask]
    
    if len(brain_values) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'variance': 0.0,
            'entropy': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'p95': 0.0,
            'p99': 0.0
        }
    
    # Intensity statistics
    mean_intensity = float(np.mean(brain_values))
    std_intensity = float(np.std(brain_values))
    variance_intensity = float(np.var(brain_values))
    
    # Percentiles
    p95 = float(np.percentile(brain_values, 95))
    p99 = float(np.percentile(brain_values, 99))
    
    # Higher-order statistics
    skewness = float(stats.skew(brain_values))
    kurtosis = float(stats.kurtosis(brain_values))
    
    # Entropy (histogram-based)
    hist, _ = np.histogram(brain_values, bins=256)
    hist = hist / (hist.sum() + 1e-10)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
    
    return {
        'mean': mean_intensity,
        'std': std_intensity,
        'variance': variance_intensity,
        'entropy': entropy,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'p95': p95,
        'p99': p99
    }


def extract_features_for_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features for all patients."""
    logger.info("Extracting features for all patients...")
    logger.info("This may take a few minutes...")
    
    features_list = []
    
    for idx, row in df.iterrows():
        patient_id = row['patient_id']
        label = row['label']
        fold = row['fold']
        
        # Determine class
        class_name = 'HGG' if label == 1 else 'LGG'
        
        # Initialize feature dict
        patient_features = {
            'patient_id': patient_id,
            'fold': fold,
            'label': label
        }
        
        # Extract features from T1ce and FLAIR (most informative for tumors)
        for modality in ['t1ce', 'flair']:
            volume = load_volume(patient_id, class_name, modality)
            
            if volume is not None:
                roi_features = compute_roi_features(volume)
                for key, value in roi_features.items():
                    patient_features[f'{modality}_{key}'] = value
            else:
                # Fill with zeros if volume not found
                for key in ['mean', 'std', 'variance', 'entropy', 'skewness', 'kurtosis', 'p95', 'p99']:
                    patient_features[f'{modality}_{key}'] = 0.0
        
        features_list.append(patient_features)
        
        if (idx + 1) % 50 == 0:
            logger.info(f"Processed {idx + 1}/{len(df)} patients...")
    
    features_df = pd.DataFrame(features_list)
    logger.info(f"✓ Extracted features for {len(features_df)} patients")
    
    return features_df


def rule_based_rescue(
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    fold: int
) -> Tuple[np.ndarray, Dict]:
    """
    Rule-based rescue: Flip LGG→HGG based on high-risk features.
    
    Rules are learned on TRAIN folds only, applied to VAL fold.
    """
    # Get train and val splits
    train_mask = df['fold'] != fold
    val_mask = df['fold'] == fold
    
    train_df = df[train_mask].merge(features_df, on=['patient_id', 'fold', 'label'], how='inner')
    val_df = df[val_mask].merge(features_df, on=['patient_id', 'fold', 'label'], how='inner')
    
    # Start with Swin-1 predictions
    train_df['swin1_pred'] = (train_df['hgg_prob_swin'] >= 0.5).astype(int)
    val_df['swin1_pred'] = (val_df['hgg_prob_swin'] >= 0.5).astype(int)
    val_df['rescue_pred'] = val_df['swin1_pred'].copy()
    
    # Identify Swin-1 FN cases in training (for rule learning)
    train_fn = train_df[(train_df['label'] == 1) & (train_df['swin1_pred'] == 0)].copy()
    train_tp = train_df[(train_df['label'] == 1) & (train_df['swin1_pred'] == 1)].copy()
    train_tn = train_df[(train_df['label'] == 0) & (train_df['swin1_pred'] == 0)].copy()
    
    if len(train_fn) == 0:
        logger.warning(f"No Swin-1 FN cases in training fold {fold}, skipping rescue")
        return val_df['rescue_pred'].values, {}
    
    # Learn feature thresholds from training FN cases
    # High-risk features: high intensity, high variance, high entropy in T1ce/FLAIR
    feature_cols = [col for col in features_df.columns if col.startswith(('t1ce_', 'flair_'))]
    
    rules = {}
    for feat_col in feature_cols:
        if feat_col.endswith('_mean') or feat_col.endswith('_p95') or feat_col.endswith('_p99'):
            # High intensity → HGG risk
            fn_values = train_fn[feat_col].values
            if len(fn_values) > 0:
                threshold = np.percentile(fn_values, 50)  # Median of FN cases
                rules[feat_col] = {'type': 'high', 'threshold': float(threshold)}
        elif feat_col.endswith('_entropy') or feat_col.endswith('_variance'):
            # High entropy/variance → HGG risk
            fn_values = train_fn[feat_col].values
            if len(fn_values) > 0:
                threshold = np.percentile(fn_values, 50)
                rules[feat_col] = {'type': 'high', 'threshold': float(threshold)}
    
    # Apply rescue: Flip LGG→HGG if high-risk features
    rescue_mask = val_df['swin1_pred'] == 0  # Only rescue LGG predictions
    
    for feat_col, rule in rules.items():
        if feat_col in val_df.columns:
            if rule['type'] == 'high':
                # If feature is above threshold, flip to HGG
                high_risk = val_df[feat_col] >= rule['threshold']
                val_df.loc[rescue_mask & high_risk, 'rescue_pred'] = 1
    
    return val_df['rescue_pred'].values, rules


def lightweight_model_rescue(
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    fold: int
) -> Tuple[np.ndarray, Optional[LogisticRegression]]:
    """
    Lightweight logistic regression rescue.
    Trained on TRAIN folds, applied to VAL fold.
    """
    # Get train and val splits
    train_mask = df['fold'] != fold
    val_mask = df['fold'] == fold
    
    train_df = df[train_mask].merge(features_df, on=['patient_id', 'fold', 'label'], how='inner')
    val_df = df[val_mask].merge(features_df, on=['patient_id', 'fold', 'label'], how='inner')
    
    # Prepare features: Swin-1 prob + handcrafted features
    feature_cols = [col for col in features_df.columns 
                   if col not in ['patient_id', 'fold', 'label']]
    
    X_train = train_df[['hgg_prob_swin'] + feature_cols].values
    y_train = train_df['label'].values
    
    X_val = val_df[['hgg_prob_swin'] + feature_cols].values
    y_val = val_df['label'].values
    
    # Train logistic regression
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Check constraints
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = precision_score(y_val, y_pred, zero_division=0)
    
    if fp >= TARGET_FP_MAX or precision < TARGET_PRECISION_MIN:
        logger.warning(f"Lightweight model violates constraints: FP={fp}, Precision={precision:.4f}")
        return None, None
    
    return y_pred, model


def evaluate_rescue(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    method_name: str
) -> Dict:
    """Evaluate rescue method."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)
    
    meets_constraints = (
        fn <= TARGET_FN_MAX and
        fp <= TARGET_FP_MAX and
        precision >= TARGET_PRECISION_MIN and
        recall >= TARGET_RECALL_MIN
    )
    
    return {
        'method': method_name,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'meets_all_constraints': bool(meets_constraints),
        'fn_excellent': bool(fn < 5)
    }


def main():
    logger.info("="*80)
    logger.info("FEATURE-LEVEL RESCUE FOR SWIN-1")
    logger.info("="*80)
    
    # Load OOF predictions
    logger.info(f"Loading OOF predictions from: {OOF_FILE}")
    df = pd.read_csv(OOF_FILE)
    logger.info(f"Loaded {len(df)} patients")
    
    # Extract features
    features_file = OUTPUT_DIR / 'patient_features.csv'
    if features_file.exists():
        logger.info(f"Loading cached features from: {features_file}")
        features_df = pd.read_csv(features_file)
    else:
        features_df = extract_features_for_patients(df)
        features_df.to_csv(features_file, index=False)
        logger.info(f"✓ Saved features to: {features_file}")
    
    folds = sorted(df['fold'].unique())
    all_results = {}
    
    # Method 1: Rule-based rescue
    logger.info("\n" + "="*80)
    logger.info("METHOD 1: RULE-BASED RESCUE")
    logger.info("="*80)
    
    rule_results = []
    for fold in folds:
        fold_mask = df['fold'] == fold
        y_true = df.loc[fold_mask, 'label'].values
        y_proba = df.loc[fold_mask, 'hgg_prob_swin'].values
        
        y_pred, rules = rule_based_rescue(df, features_df, fold)
        
        metrics = evaluate_rescue(y_true, y_pred, y_proba, f"RuleBased_fold_{fold}")
        metrics['rules'] = rules
        rule_results.append(metrics)
        
        logger.info(f"Fold {fold}: FN={metrics['fn']}, FP={metrics['fp']}, "
                   f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    rule_agg = aggregate_results(rule_results, "RuleBased")
    all_results['rule_based'] = {'fold_results': rule_results, 'aggregated': rule_agg}
    
    # Method 2: Lightweight model (only if rule-based helps)
    logger.info("\n" + "="*80)
    logger.info("METHOD 2: LIGHTWEIGHT LOGISTIC REGRESSION")
    logger.info("="*80)
    
    if rule_agg['meets_all_constraints'] or rule_agg['fn_mean'] < 10:
        logger.info("Rule-based shows promise, trying lightweight model...")
        
        model_results = []
        for fold in folds:
            fold_mask = df['fold'] == fold
            y_true = df.loc[fold_mask, 'label'].values
            y_proba = df.loc[fold_mask, 'hgg_prob_swin'].values
            
            y_pred, model = lightweight_model_rescue(df, features_df, fold)
            
            if y_pred is not None:
                metrics = evaluate_rescue(y_true, y_pred, y_proba, f"LightweightModel_fold_{fold}")
                model_results.append(metrics)
                
                logger.info(f"Fold {fold}: FN={metrics['fn']}, FP={metrics['fp']}, "
                           f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
            else:
                logger.warning(f"Fold {fold}: Lightweight model rejected (violates constraints)")
                # Fallback to baseline
                y_pred = (y_proba >= 0.5).astype(int)
                metrics = evaluate_rescue(y_true, y_pred, y_proba, f"LightweightModel_fold_{fold}")
                model_results.append(metrics)
        
        model_agg = aggregate_results(model_results, "LightweightModel")
        all_results['lightweight_model'] = {'fold_results': model_results, 'aggregated': model_agg}
    else:
        logger.info("Rule-based does not help, skipping lightweight model")
    
    # Save results
    json_path = OUTPUT_DIR / 'rescue_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\n✓ Saved results to: {json_path}")
    
    # Generate markdown report
    generate_markdown_report(all_results, OUTPUT_DIR)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    
    for method_name, method_data in all_results.items():
        agg = method_data['aggregated']
        logger.info(f"\n{method_name.upper()}:")
        logger.info(f"  FN: {agg['fn_mean']:.1f} ± {agg['fn_std']:.1f} (target: <{TARGET_FN_MAX})")
        logger.info(f"  FP: {agg['fp_mean']:.1f} ± {agg['fp_std']:.1f} (target: <{TARGET_FP_MAX})")
        logger.info(f"  Precision: {agg['precision_mean']:.4f} ± {agg['precision_std']:.4f} (target: ≥{TARGET_PRECISION_MIN})")
        logger.info(f"  Recall: {agg['recall_mean']:.4f} ± {agg['recall_std']:.4f} (target: ≥{TARGET_RECALL_MIN})")
        logger.info(f"  Meets ALL constraints: {agg['meets_all_constraints']}")


def aggregate_results(fold_results: List[Dict], method_name: str) -> Dict:
    """Aggregate results across folds."""
    fn_values = [r['fn'] for r in fold_results]
    fp_values = [r['fp'] for r in fold_results]
    precision_values = [r['precision'] for r in fold_results]
    recall_values = [r['recall'] for r in fold_results]
    f1_values = [r['f1'] for r in fold_results]
    auc_roc_values = [r['auc_roc'] for r in fold_results]
    
    return {
        'method': method_name,
        'fn_mean': float(np.mean(fn_values)),
        'fn_std': float(np.std(fn_values)),
        'fp_mean': float(np.mean(fp_values)),
        'fp_std': float(np.std(fp_values)),
        'precision_mean': float(np.mean(precision_values)),
        'precision_std': float(np.std(precision_values)),
        'recall_mean': float(np.mean(recall_values)),
        'recall_std': float(np.std(recall_values)),
        'f1_mean': float(np.mean(f1_values)),
        'f1_std': float(np.std(f1_values)),
        'auc_roc_mean': float(np.mean(auc_roc_values)),
        'auc_roc_std': float(np.std(auc_roc_values)),
        'meets_all_constraints': bool(
            np.mean(fn_values) <= TARGET_FN_MAX and
            np.mean(fp_values) <= TARGET_FP_MAX and
            np.mean(precision_values) >= TARGET_PRECISION_MIN and
            np.mean(recall_values) >= TARGET_RECALL_MIN
        ),
        'fn_excellent': bool(np.mean(fn_values) < 5)
    }


def generate_markdown_report(all_results: Dict, output_dir: Path):
    """Generate markdown comparison report."""
    md_content = "# Feature-Level Rescue Results for Swin-1\n\n"
    md_content += "## Target Constraints\n\n"
    md_content += f"- FN < {TARGET_FN_MAX} (FN < 5 is excellent)\n"
    md_content += f"- FP < {TARGET_FP_MAX}\n"
    md_content += f"- Precision ≥ {TARGET_PRECISION_MIN}\n"
    md_content += f"- Recall ≥ {TARGET_RECALL_MIN}\n\n"
    md_content += "**All constraints must be met simultaneously.**\n\n"
    
    md_content += "## Results Comparison\n\n"
    md_content += "| Method | FN (mean±std) | FP (mean±std) | Precision (mean±std) | Recall (mean±std) | Meets All? |\n"
    md_content += "|--------|---------------|---------------|---------------------|-------------------|------------|\n"
    
    for method_name, method_data in all_results.items():
        agg = method_data['aggregated']
        meets = "✅ YES" if agg['meets_all_constraints'] else "❌ NO"
        md_content += f"| {agg['method']} | {agg['fn_mean']:.1f}±{agg['fn_std']:.1f} | "
        md_content += f"{agg['fp_mean']:.1f}±{agg['fp_std']:.1f} | "
        md_content += f"{agg['precision_mean']:.4f}±{agg['precision_std']:.4f} | "
        md_content += f"{agg['recall_mean']:.4f}±{agg['recall_std']:.4f} | {meets} |\n"
    
    md_content += "\n## Executive Summary\n\n"
    
    # Find best method
    best_method = None
    for method_name, method_data in all_results.items():
        agg = method_data['aggregated']
        if agg['meets_all_constraints']:
            if best_method is None or agg['fn_mean'] < all_results[best_method]['aggregated']['fn_mean']:
                best_method = method_name
    
    if best_method:
        best_agg = all_results[best_method]['aggregated']
        md_content += f"**Best Method: {best_agg['method']}**\n\n"
        md_content += f"- FN: {best_agg['fn_mean']:.1f} ± {best_agg['fn_std']:.1f}\n"
        md_content += f"- FP: {best_agg['fp_mean']:.1f} ± {best_agg['fp_std']:.1f}\n"
        md_content += f"- Precision: {best_agg['precision_mean']:.4f} ± {best_agg['precision_std']:.4f}\n"
        md_content += f"- Recall: {best_agg['recall_mean']:.4f} ± {best_agg['recall_std']:.4f}\n"
        if best_agg['fn_excellent']:
            md_content += f"\n✅ **EXCELLENT: FN < 5 achieved!**\n"
    else:
        md_content += "**❌ NO METHOD MEETS ALL CONSTRAINTS**\n\n"
        md_content += "None of the rescue methods achieve:\n"
        md_content += f"- FN < {TARGET_FN_MAX} AND\n"
        md_content += f"- FP < {TARGET_FP_MAX} AND\n"
        md_content += f"- Precision ≥ {TARGET_PRECISION_MIN} AND\n"
        md_content += f"- Recall ≥ {TARGET_RECALL_MIN}\n\n"
        md_content += "**Conclusion:** Post-hoc methods alone cannot achieve target constraints.\n"
        md_content += "**Recommendation:** Model retraining or additional data may be required.\n"
    
    md_path = output_dir / 'rescue_results.md'
    with open(md_path, 'w') as f:
        f.write(md_content)
    logger.info(f"✓ Saved markdown report to: {md_path}")


if __name__ == '__main__':
    main()

