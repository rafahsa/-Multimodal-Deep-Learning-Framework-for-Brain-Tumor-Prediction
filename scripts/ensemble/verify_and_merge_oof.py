"""
Verify and Merge Out-of-Fold (OOF) Predictions for Ensemble Stacking

This script performs comprehensive verification of OOF predictions and merges them
from all three models into a single CSV file ready for meta-learner training.

Usage:
    python scripts/ensemble/verify_and_merge_oof.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ENSEMBLE_DIR = Path('ensemble/oof_predictions')
SPLITS_DIR = Path('splits')
NUM_FOLDS = 5

MODEL_FILES = {
    'ResNet50-3D': 'resnet50_3d_oof.csv',
    'SwinUNETR-3D': 'swinunetr_3d_oof.csv',
    'DualStreamMIL-3D': 'dualstream_mil_3d_oof.csv'
}

VERIFICATION_REPORT = []


def log_verification(check_name: str, passed: bool, message: str = ""):
    """Log verification result and add to report."""
    status = "✓ PASS" if passed else "✗ FAIL"
    logger.info(f"{status}: {check_name} - {message}")
    VERIFICATION_REPORT.append({
        'check': check_name,
        'passed': passed,
        'message': message
    })


def verify_individual_model_oof(model_name: str, file_path: Path) -> Tuple[bool, pd.DataFrame]:
    """
    Verify OOF predictions for a single model.
    
    Returns:
        (success: bool, df: DataFrame)
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Verifying {model_name}")
    logger.info(f"{'='*80}")
    
    if not file_path.exists():
        log_verification(f"{model_name} - File exists", False, f"File not found: {file_path}")
        return False, None
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        log_verification(f"{model_name} - CSV readable", False, f"Error reading CSV: {e}")
        return False, None
    
    log_verification(f"{model_name} - CSV readable", True, f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Check required columns
    required_cols = ['patient_id', 'fold', 'hgg_prob', 'label']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        log_verification(f"{model_name} - Required columns", False, f"Missing columns: {missing_cols}")
        return False, df
    log_verification(f"{model_name} - Required columns", True, f"All required columns present: {required_cols}")
    
    # Check uniqueness of patient_id
    n_unique = df['patient_id'].nunique()
    n_total = len(df)
    if n_unique != n_total:
        duplicates = df[df.duplicated(subset=['patient_id'], keep=False)]
        log_verification(f"{model_name} - Patient ID uniqueness", False, 
                        f"Found {n_total - n_unique} duplicate patient IDs. Examples: {duplicates['patient_id'].head(5).tolist()}")
        return False, df
    log_verification(f"{model_name} - Patient ID uniqueness", True, f"All {n_unique} patient IDs are unique")
    
    # Check fold values
    unique_folds = sorted(df['fold'].unique())
    expected_folds = list(range(NUM_FOLDS))
    if set(unique_folds) != set(expected_folds):
        log_verification(f"{model_name} - Fold values", False, 
                        f"Unexpected folds. Expected {expected_folds}, got {unique_folds}")
        return False, df
    log_verification(f"{model_name} - Fold values", True, f"All folds present: {unique_folds}")
    
    # Check fold distribution
    fold_counts = df['fold'].value_counts().sort_index()
    log_verification(f"{model_name} - Fold distribution", True, 
                    f"Fold counts: {fold_counts.to_dict()}")
    
    # Check hgg_prob range
    invalid_probs = df[(df['hgg_prob'] < 0) | (df['hgg_prob'] > 1)]
    if len(invalid_probs) > 0:
        log_verification(f"{model_name} - Probability range", False, 
                        f"Found {len(invalid_probs)} probabilities outside [0, 1]")
        return False, df
    log_verification(f"{model_name} - Probability range", True, 
                    f"All probabilities in [0, 1]. Range: [{df['hgg_prob'].min():.4f}, {df['hgg_prob'].max():.4f}]")
    
    # Check label values
    invalid_labels = df[~df['label'].isin([0, 1])]
    if len(invalid_labels) > 0:
        log_verification(f"{model_name} - Label values", False, 
                        f"Found {len(invalid_labels)} invalid labels (not 0 or 1)")
        return False, df
    
    label_counts = df['label'].value_counts().sort_index()
    log_verification(f"{model_name} - Label values", True, 
                    f"Label distribution: {label_counts.to_dict()} (0=LGG, 1=HGG)")
    
    # Check for missing values
    missing_values = df[required_cols].isnull().sum()
    if missing_values.sum() > 0:
        log_verification(f"{model_name} - Missing values", False, 
                        f"Found missing values:\n{missing_values[missing_values > 0]}")
        return False, df
    log_verification(f"{model_name} - Missing values", True, "No missing values found")
    
    # Verify against validation split files
    all_patients_in_splits = set()
    for fold in range(NUM_FOLDS):
        split_file = SPLITS_DIR / f'fold_{fold}_val.csv'
        if split_file.exists():
            split_df = pd.read_csv(split_file)
            fold_patients = set(split_df['patient_id'].unique())
            all_patients_in_splits.update(fold_patients)
            
            # Check patients in this fold match
            df_fold = df[df['fold'] == fold]
            df_fold_patients = set(df_fold['patient_id'].unique())
            
            if df_fold_patients != fold_patients:
                missing_in_df = fold_patients - df_fold_patients
                extra_in_df = df_fold_patients - fold_patients
                log_verification(f"{model_name} - Fold {fold} patient match", False,
                               f"Patient mismatch. Missing in OOF: {list(missing_in_df)[:5]}, Extra in OOF: {list(extra_in_df)[:5]}")
                return False, df
            else:
                log_verification(f"{model_name} - Fold {fold} patient match", True,
                               f"All {len(fold_patients)} patients from validation split present")
    
    # Verify total patients match
    oof_patients = set(df['patient_id'].unique())
    if oof_patients != all_patients_in_splits:
        missing = all_patients_in_splits - oof_patients
        extra = oof_patients - all_patients_in_splits
        log_verification(f"{model_name} - Total patient match", False,
                        f"Patient mismatch. Missing: {len(missing)}, Extra: {len(extra)}")
        if len(missing) <= 10:
            log_verification(f"{model_name} - Missing patients", False, f"Missing: {list(missing)}")
        if len(extra) <= 10:
            log_verification(f"{model_name} - Extra patients", False, f"Extra: {list(extra)}")
        return False, df
    
    log_verification(f"{model_name} - Total patient match", True,
                    f"All {len(all_patients_in_splits)} patients from validation splits present")
    
    logger.info(f"✓ All verification checks passed for {model_name}")
    return True, df


def verify_no_data_leakage(all_dfs: Dict[str, pd.DataFrame]) -> bool:
    """Verify that there is no data leakage across folds."""
    logger.info(f"\n{'='*80}")
    logger.info("Verifying No Data Leakage")
    logger.info(f"{'='*80}")
    
    # Check that each patient appears exactly once per model
    for model_name, df in all_dfs.items():
        patient_counts = df['patient_id'].value_counts()
        duplicates = patient_counts[patient_counts > 1]
        if len(duplicates) > 0:
            log_verification(f"{model_name} - No duplicate patients", False,
                           f"Found {len(duplicates)} patients appearing multiple times")
            return False
        log_verification(f"{model_name} - No duplicate patients", True,
                        f"Each patient appears exactly once")
    
    # Check that patients are mutually exclusive across folds
    all_patients_by_fold = {}
    for fold in range(NUM_FOLDS):
        fold_patients = set()
        for model_name, df in all_dfs.items():
            fold_patients.update(df[df['fold'] == fold]['patient_id'].unique())
        all_patients_by_fold[fold] = fold_patients
    
    # Check for overlap between folds
    for fold1 in range(NUM_FOLDS):
        for fold2 in range(fold1 + 1, NUM_FOLDS):
            overlap = all_patients_by_fold[fold1] & all_patients_by_fold[fold2]
            if overlap:
                log_verification(f"Folds {fold1} and {fold2} - Mutually exclusive", False,
                               f"Found {len(overlap)} overlapping patients: {list(overlap)[:5]}")
                return False
    
    log_verification("Folds mutually exclusive", True,
                    f"All folds are mutually exclusive (no patient overlap)")
    
    logger.info("✓ No data leakage detected")
    return True


def merge_oof_predictions(all_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge OOF predictions from all models.
    
    Returns:
        Merged DataFrame with columns: patient_id, fold, hgg_prob_resnet, hgg_prob_swin, hgg_prob_mil, label
    """
    logger.info(f"\n{'='*80}")
    logger.info("Merging OOF Predictions")
    logger.info(f"{'='*80}")
    
    # Start with first model
    merged = all_dfs['ResNet50-3D'][['patient_id', 'fold', 'hgg_prob', 'label']].copy()
    merged.rename(columns={'hgg_prob': 'hgg_prob_resnet'}, inplace=True)
    
    # Merge with SwinUNETR-3D
    swin_df = all_dfs['SwinUNETR-3D'][['patient_id', 'hgg_prob']].copy()
    swin_df.rename(columns={'hgg_prob': 'hgg_prob_swin'}, inplace=True)
    merged = merged.merge(swin_df, on='patient_id', how='inner', validate='1:1')
    
    # Merge with DualStreamMIL-3D
    mil_df = all_dfs['DualStreamMIL-3D'][['patient_id', 'hgg_prob']].copy()
    mil_df.rename(columns={'hgg_prob': 'hgg_prob_mil'}, inplace=True)
    merged = merged.merge(mil_df, on='patient_id', how='inner', validate='1:1')
    
    # Verify merge results
    if len(merged) != len(all_dfs['ResNet50-3D']):
        logger.error(f"Merge resulted in {len(merged)} rows, expected {len(all_dfs['ResNet50-3D'])}")
        return None
    
    # Verify all models have predictions for all patients
    missing_resnet = set(all_dfs['ResNet50-3D']['patient_id']) - set(merged['patient_id'])
    missing_swin = set(all_dfs['SwinUNETR-3D']['patient_id']) - set(merged['patient_id'])
    missing_mil = set(all_dfs['DualStreamMIL-3D']['patient_id']) - set(merged['patient_id'])
    
    if missing_resnet or missing_swin or missing_mil:
        logger.error(f"Missing patients after merge: ResNet={len(missing_resnet)}, Swin={len(missing_swin)}, MIL={len(missing_mil)}")
        return None
    
    # Sort by patient_id for consistency
    merged = merged.sort_values('patient_id').reset_index(drop=True)
    
    # Reorder columns
    merged = merged[['patient_id', 'fold', 'hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil', 'label']]
    
    logger.info(f"✓ Successfully merged predictions from all three models")
    logger.info(f"  Total patients: {len(merged)}")
    logger.info(f"  Columns: {list(merged.columns)}")
    logger.info(f"  Fold distribution: {merged['fold'].value_counts().sort_index().to_dict()}")
    
    # Verify label consistency across models (should be same for all)
    if not (merged['label'].nunique() == 1 or 
            (all_dfs['ResNet50-3D'].set_index('patient_id')['label'] == 
             all_dfs['SwinUNETR-3D'].set_index('patient_id')['label']).all()):
        logger.warning("Labels may not be consistent across models - this is unexpected")
    else:
        log_verification("Label consistency", True, "Labels are consistent across all models")
    
    return merged


def generate_verification_report(merged_df: pd.DataFrame = None) -> str:
    """Generate a comprehensive verification report."""
    report_lines = [
        "=" * 80,
        "OOF Predictions Verification Report",
        "=" * 80,
        "",
        f"Generated: {pd.Timestamp.now()}",
        "",
        "VERIFICATION CHECKS:",
        "-" * 80,
    ]
    
    for item in VERIFICATION_REPORT:
        status = "✓ PASS" if item['passed'] else "✗ FAIL"
        report_lines.append(f"{status} - {item['check']}")
        if item['message']:
            report_lines.append(f"  {item['message']}")
        report_lines.append("")
    
    report_lines.append("-" * 80)
    
    # Summary statistics
    passed_checks = sum(1 for item in VERIFICATION_REPORT if item['passed'])
    total_checks = len(VERIFICATION_REPORT)
    report_lines.append(f"Summary: {passed_checks}/{total_checks} checks passed")
    report_lines.append("")
    
    if merged_df is not None:
        report_lines.extend([
            "MERGED DATASET STATISTICS:",
            "-" * 80,
            f"Total patients: {len(merged_df)}",
            f"Columns: {', '.join(merged_df.columns)}",
            "",
            "Fold distribution:",
        ])
        
        fold_counts = merged_df['fold'].value_counts().sort_index()
        for fold, count in fold_counts.items():
            report_lines.append(f"  Fold {fold}: {count} patients")
        
        report_lines.extend([
            "",
            "Label distribution:",
        ])
        label_counts = merged_df['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            label_name = "LGG" if label == 0 else "HGG"
            report_lines.append(f"  {label_name} (label={label}): {count} patients")
        
        report_lines.extend([
            "",
            "Probability statistics:",
            f"  ResNet50-3D: mean={merged_df['hgg_prob_resnet'].mean():.4f}, std={merged_df['hgg_prob_resnet'].std():.4f}, "
            f"min={merged_df['hgg_prob_resnet'].min():.4f}, max={merged_df['hgg_prob_resnet'].max():.4f}",
            f"  SwinUNETR-3D: mean={merged_df['hgg_prob_swin'].mean():.4f}, std={merged_df['hgg_prob_swin'].std():.4f}, "
            f"min={merged_df['hgg_prob_swin'].min():.4f}, max={merged_df['hgg_prob_swin'].max():.4f}",
            f"  DualStreamMIL-3D: mean={merged_df['hgg_prob_mil'].mean():.4f}, std={merged_df['hgg_prob_mil'].std():.4f}, "
            f"min={merged_df['hgg_prob_mil'].min():.4f}, max={merged_df['hgg_prob_mil'].max():.4f}",
        ])
    
    report_lines.extend([
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80,
    ])
    
    return "\n".join(report_lines)


def main():
    """Main function to verify and merge OOF predictions."""
    logger.info("=" * 80)
    logger.info("OOF Predictions Verification and Merging")
    logger.info("=" * 80)
    
    # Verify each model's OOF predictions
    all_dfs = {}
    all_passed = True
    
    for model_name, filename in MODEL_FILES.items():
        file_path = ENSEMBLE_DIR / filename
        success, df = verify_individual_model_oof(model_name, file_path)
        
        if not success:
            all_passed = False
            logger.error(f"Verification failed for {model_name}")
            continue
        
        all_dfs[model_name] = df
    
    if not all_passed:
        logger.error("Verification failed for one or more models. Cannot proceed with merging.")
        return
    
    # Verify no data leakage
    if not verify_no_data_leakage(all_dfs):
        logger.error("Data leakage detected. Cannot proceed with merging.")
        return
    
    # Merge predictions
    merged_df = merge_oof_predictions(all_dfs)
    
    if merged_df is None:
        logger.error("Failed to merge predictions")
        return
    
    # Save merged predictions
    output_file = ENSEMBLE_DIR / 'merged_oof_predictions.csv'
    merged_df.to_csv(output_file, index=False)
    logger.info(f"\n✓ Saved merged OOF predictions to: {output_file}")
    
    # Generate and save verification report
    report = generate_verification_report(merged_df)
    report_file = ENSEMBLE_DIR / 'verification_report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"✓ Saved verification report to: {report_file}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Verification and Merging Complete")
    logger.info("=" * 80)
    logger.info(f"Merged file: {output_file}")
    logger.info(f"Report file: {report_file}")
    logger.info(f"Total patients: {len(merged_df)}")
    logger.info(f"Ready for meta-learner training: ✓")


if __name__ == '__main__':
    main()

