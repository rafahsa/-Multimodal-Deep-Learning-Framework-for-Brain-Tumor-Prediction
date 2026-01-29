"""
Verify Merged OOF Predictions for Meta-Learner Training Readiness

This script performs comprehensive verification of the merged OOF predictions
to ensure they are ready for Logistic Regression meta-learner training.

Usage:
    python scripts/ensemble/verify_merged_oof_for_training.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MERGED_FILE = Path('ensemble/oof_predictions/merged_oof_predictions.csv')
SPLITS_DIR = Path('splits')
NUM_FOLDS = 5
EXPECTED_PATIENTS_PER_FOLD = 57
EXPECTED_TOTAL_PATIENTS = 285

REQUIRED_COLUMNS = [
    'patient_id', 
    'fold', 
    'hgg_prob_resnet', 
    'hgg_prob_swin', 
    'hgg_prob_mil', 
    'label'
]

VERIFICATION_RESULTS = []


def log_check(check_name: str, passed: bool, message: str = "", details: Dict = None):
    """Log verification check result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    logger.info(f"{status}: {check_name}")
    if message:
        logger.info(f"  {message}")
    if details:
        for key, value in details.items():
            logger.info(f"    {key}: {value}")
    
    VERIFICATION_RESULTS.append({
        'check': check_name,
        'passed': passed,
        'message': message,
        'details': details or {}
    })


def verify_file_exists() -> Tuple[bool, pd.DataFrame]:
    """Check if merged file exists and is readable."""
    if not MERGED_FILE.exists():
        log_check("File exists", False, f"Merged file not found: {MERGED_FILE}")
        return False, None
    
    try:
        df = pd.read_csv(MERGED_FILE)
        log_check("File exists and readable", True, 
                 f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
        return True, df
    except Exception as e:
        log_check("File readable", False, f"Error reading CSV: {e}")
        return False, None


def verify_columns(df: pd.DataFrame) -> bool:
    """Verify required columns are present."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in REQUIRED_COLUMNS]
    
    if missing_cols:
        log_check("Required columns", False, 
                 f"Missing columns: {missing_cols}")
        return False
    
    if extra_cols:
        log_check("Required columns", True, 
                 f"All required columns present. Extra columns (will be ignored): {extra_cols}")
    else:
        log_check("Required columns", True, 
                 f"All required columns present: {REQUIRED_COLUMNS}")
    
    return True


def verify_patient_id(df: pd.DataFrame) -> bool:
    """Verify patient_id column."""
    # Check for missing values
    missing = df['patient_id'].isnull().sum()
    if missing > 0:
        log_check("Patient ID - No missing values", False, 
                 f"Found {missing} missing patient IDs")
        return False
    
    log_check("Patient ID - No missing values", True, "No missing patient IDs")
    
    # Check uniqueness
    n_unique = df['patient_id'].nunique()
    n_total = len(df)
    if n_unique != n_total:
        duplicates = df[df.duplicated(subset=['patient_id'], keep=False)]
        log_check("Patient ID - Uniqueness", False,
                 f"Found {n_total - n_unique} duplicate patient IDs",
                 {'duplicate_examples': duplicates['patient_id'].head(10).tolist()})
        return False
    
    log_check("Patient ID - Uniqueness", True, 
             f"All {n_unique} patient IDs are unique")
    
    # Check format (should be string, non-empty)
    invalid_format = df['patient_id'].astype(str).str.strip().eq('').sum()
    if invalid_format > 0:
        log_check("Patient ID - Format", False,
                 f"Found {invalid_format} empty or whitespace-only patient IDs")
        return False
    
    log_check("Patient ID - Format", True, "All patient IDs are non-empty strings")
    
    return True


def verify_fold(df: pd.DataFrame) -> bool:
    """Verify fold column."""
    # Check for missing values
    missing = df['fold'].isnull().sum()
    if missing > 0:
        log_check("Fold - No missing values", False,
                 f"Found {missing} missing fold values")
        return False
    
    log_check("Fold - No missing values", True, "No missing fold values")
    
    # Check range
    valid_folds = set(range(NUM_FOLDS))
    actual_folds = set(df['fold'].unique())
    invalid_folds = actual_folds - valid_folds
    
    if invalid_folds:
        log_check("Fold - Valid range", False,
                 f"Found invalid fold values: {sorted(invalid_folds)}. Expected: {sorted(valid_folds)}")
        return False
    
    log_check("Fold - Valid range", True,
             f"All fold values in valid range [0, {NUM_FOLDS-1}]. Folds present: {sorted(actual_folds)}")
    
    # Check completeness - each fold should have exactly EXPECTED_PATIENTS_PER_FOLD patients
    fold_counts = df['fold'].value_counts().sort_index()
    incomplete_folds = []
    for fold in range(NUM_FOLDS):
        if fold not in fold_counts.index:
            incomplete_folds.append(f"Fold {fold}: missing")
        elif fold_counts[fold] != EXPECTED_PATIENTS_PER_FOLD:
            incomplete_folds.append(f"Fold {fold}: {fold_counts[fold]} patients (expected {EXPECTED_PATIENTS_PER_FOLD})")
    
    if incomplete_folds:
        log_check("Fold - Completeness", False,
                 "Fold distribution issues",
                 {'fold_counts': fold_counts.to_dict(), 'issues': incomplete_folds})
        return False
    
    log_check("Fold - Completeness", True,
             f"All {NUM_FOLDS} folds have exactly {EXPECTED_PATIENTS_PER_FOLD} patients each",
             {'fold_distribution': fold_counts.to_dict()})
    
    # Check that each patient appears in exactly one fold
    patient_fold_counts = df.groupby('patient_id')['fold'].nunique()
    patients_in_multiple_folds = patient_fold_counts[patient_fold_counts > 1]
    
    if len(patients_in_multiple_folds) > 0:
        log_check("Fold - Patient exclusivity", False,
                 f"Found {len(patients_in_multiple_folds)} patients appearing in multiple folds",
                 {'examples': patients_in_multiple_folds.head(10).to_dict()})
        return False
    
    log_check("Fold - Patient exclusivity", True,
             "Each patient appears in exactly one fold")
    
    return True


def verify_probabilities(df: pd.DataFrame) -> bool:
    """Verify probability columns."""
    prob_columns = ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']
    all_passed = True
    
    for col in prob_columns:
        if col not in df.columns:
            log_check(f"{col} - Column exists", False, f"Column not found: {col}")
            all_passed = False
            continue
        
        # Check for missing values
        missing = df[col].isnull().sum()
        if missing > 0:
            log_check(f"{col} - No missing values", False,
                     f"Found {missing} missing values")
            all_passed = False
            continue
        
        log_check(f"{col} - No missing values", True, "No missing values")
        
        # Check range [0, 1]
        invalid_range = df[(df[col] < 0) | (df[col] > 1)]
        if len(invalid_range) > 0:
            log_check(f"{col} - Valid range [0, 1]", False,
                     f"Found {len(invalid_range)} values outside [0, 1]",
                     {'min': df[col].min(), 'max': df[col].max(),
                      'invalid_examples': invalid_range[[col]].head(10).to_dict('records')})
            all_passed = False
            continue
        
        # Check data type
        if not pd.api.types.is_numeric_dtype(df[col]):
            log_check(f"{col} - Numeric type", False,
                     f"Column is not numeric: {df[col].dtype}")
            all_passed = False
            continue
        
        log_check(f"{col} - Valid range and type", True,
                 f"All values in [0, 1]. Range: [{df[col].min():.6f}, {df[col].max():.6f}], "
                 f"Mean: {df[col].mean():.6f}, Std: {df[col].std():.6f}")
    
    return all_passed


def verify_labels(df: pd.DataFrame) -> bool:
    """Verify label column."""
    # Check for missing values
    missing = df['label'].isnull().sum()
    if missing > 0:
        log_check("Label - No missing values", False,
                 f"Found {missing} missing labels")
        return False
    
    log_check("Label - No missing values", True, "No missing labels")
    
    # Check valid values (0 or 1 only)
    valid_labels = {0, 1}
    actual_labels = set(df['label'].unique())
    invalid_labels = actual_labels - valid_labels
    
    if invalid_labels:
        log_check("Label - Valid values", False,
                 f"Found invalid label values: {sorted(invalid_labels)}. Expected: {sorted(valid_labels)}")
        return False
    
    log_check("Label - Valid values", True,
             f"All labels are valid (0=LGG, 1=HGG). Values present: {sorted(actual_labels)}")
    
    # Check data type
    if not pd.api.types.is_integer_dtype(df['label']):
        log_check("Label - Integer type", False,
                 f"Label column is not integer: {df['label'].dtype}")
        return False
    
    log_check("Label - Integer type", True, "Label column is integer type")
    
    # Check distribution
    label_counts = df['label'].value_counts().sort_index()
    log_check("Label - Distribution", True,
             f"Label distribution: {label_counts.to_dict()}",
             {'LGG (0)': int(label_counts.get(0, 0)), 
              'HGG (1)': int(label_counts.get(1, 0)),
              'Total': len(df)})
    
    return True


def verify_validation_split_matching(df: pd.DataFrame) -> bool:
    """Verify that patient IDs match validation split files."""
    all_patients_in_splits = set()
    all_folds_valid = True
    
    for fold in range(NUM_FOLDS):
        split_file = SPLITS_DIR / f'fold_{fold}_val.csv'
        if not split_file.exists():
            log_check(f"Validation split file - Fold {fold}", False,
                     f"Validation split file not found: {split_file}")
            all_folds_valid = False
            continue
        
        try:
            split_df = pd.read_csv(split_file)
            split_patients = set(split_df['patient_id'].unique())
            all_patients_in_splits.update(split_patients)
            
            # Check patients in OOF data for this fold
            oof_fold_patients = set(df[df['fold'] == fold]['patient_id'].unique())
            
            if oof_fold_patients != split_patients:
                missing_in_oof = split_patients - oof_fold_patients
                extra_in_oof = oof_fold_patients - split_patients
                
                log_check(f"Patient-fold matching - Fold {fold}", False,
                         f"Patient mismatch",
                         {'missing_in_oof': list(missing_in_oof)[:10] if missing_in_oof else [],
                          'extra_in_oof': list(extra_in_oof)[:10] if extra_in_oof else [],
                          'split_count': len(split_patients),
                          'oof_count': len(oof_fold_patients)})
                all_folds_valid = False
            else:
                log_check(f"Patient-fold matching - Fold {fold}", True,
                         f"All {len(split_patients)} patients from validation split match")
        
        except Exception as e:
            log_check(f"Validation split file - Fold {fold}", False,
                     f"Error reading validation split file: {e}")
            all_folds_valid = False
    
    # Check total patient count
    oof_patients = set(df['patient_id'].unique())
    if oof_patients != all_patients_in_splits:
        missing = all_patients_in_splits - oof_patients
        extra = oof_patients - all_patients_in_splits
        log_check("Total patient matching", False,
                 f"Total patient mismatch",
                 {'missing_in_oof': len(missing), 'extra_in_oof': len(extra),
                  'expected_total': len(all_patients_in_splits),
                  'actual_total': len(oof_patients)})
        if len(missing) <= 20:
            log_check("Missing patients", False, f"Missing: {list(missing)}")
        if len(extra) <= 20:
            log_check("Extra patients", False, f"Extra: {list(extra)}")
        return False
    
    log_check("Total patient matching", True,
             f"All {len(all_patients_in_splits)} patients from validation splits are present")
    
    return all_folds_valid


def verify_training_readiness(df: pd.DataFrame) -> bool:
    """Verify that data is ready for Logistic Regression training."""
    logger.info(f"\n{'='*80}")
    logger.info("Training Readiness Check")
    logger.info(f"{'='*80}")
    
    # Check that we have the required features for training
    feature_cols = ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']
    missing_features = [col for col in feature_cols if col not in df.columns]
    
    if missing_features:
        log_check("Training readiness - Feature columns", False,
                 f"Missing feature columns: {missing_features}")
        return False
    
    log_check("Training readiness - Feature columns", True,
             f"All feature columns present: {feature_cols}")
    
    # Check that we have target variable
    if 'label' not in df.columns:
        log_check("Training readiness - Target variable", False,
                 "Target variable 'label' not found")
        return False
    
    log_check("Training readiness - Target variable", True,
             "Target variable 'label' present")
    
    # Check minimum sample size (should have at least some samples from each class)
    label_counts = df['label'].value_counts()
    if len(label_counts) < 2:
        log_check("Training readiness - Class balance", False,
                 f"Only {len(label_counts)} class(es) present. Need at least 2 for binary classification")
        return False
    
    min_class_count = label_counts.min()
    log_check("Training readiness - Class balance", True,
             f"Both classes present. Min class size: {min_class_count}",
             {'class_distribution': label_counts.to_dict()})
    
    # Check that all features are numeric and ready
    all_numeric = all(pd.api.types.is_numeric_dtype(df[col]) for col in feature_cols)
    if not all_numeric:
        non_numeric = [col for col in feature_cols if not pd.api.types.is_numeric_dtype(df[col])]
        log_check("Training readiness - Feature types", False,
                 f"Non-numeric feature columns: {non_numeric}")
        return False
    
    log_check("Training readiness - Feature types", True,
             "All features are numeric")
    
    # Check for sufficient samples (285 is sufficient for logistic regression)
    if len(df) < 10:
        log_check("Training readiness - Sample size", False,
                 f"Insufficient samples: {len(df)} (minimum 10 recommended)")
        return False
    
    log_check("Training readiness - Sample size", True,
             f"Sufficient samples for training: {len(df)}")
    
    logger.info("✓ Data is ready for Logistic Regression training")
    return True


def generate_report() -> str:
    """Generate comprehensive verification report."""
    passed = sum(1 for r in VERIFICATION_RESULTS if r['passed'])
    total = len(VERIFICATION_RESULTS)
    
    report_lines = [
        "=" * 80,
        "MERGED OOF PREDICTIONS VERIFICATION REPORT",
        "Meta-Learner Training Readiness Check",
        "=" * 80,
        "",
        f"Generated: {pd.Timestamp.now()}",
        f"File: {MERGED_FILE}",
        "",
        "SUMMARY:",
        "-" * 80,
        f"Total checks: {total}",
        f"Passed: {passed}",
        f"Failed: {total - passed}",
        f"Success rate: {passed/total*100:.1f}%",
        "",
        "VERIFICATION RESULTS:",
        "-" * 80,
    ]
    
    for result in VERIFICATION_RESULTS:
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        report_lines.append(f"{status} - {result['check']}")
        if result['message']:
            report_lines.append(f"  {result['message']}")
        if result['details']:
            for key, value in result['details'].items():
                if isinstance(value, (dict, list)):
                    value_str = str(value)[:200]  # Truncate long values
                else:
                    value_str = str(value)
                report_lines.append(f"    {key}: {value_str}")
        report_lines.append("")
    
    report_lines.extend([
        "-" * 80,
        "",
        "CONCLUSION:",
        "-" * 80,
    ])
    
    if passed == total:
        report_lines.append("✅ ALL VERIFICATION CHECKS PASSED")
        report_lines.append("")
        report_lines.append("The merged OOF predictions are:")
        report_lines.append("  - Properly formatted and structured")
        report_lines.append("  - Complete with no missing values")
        report_lines.append("  - Correctly matched to validation splits")
        report_lines.append("  - Ready for Logistic Regression meta-learner training")
    else:
        report_lines.append("❌ VERIFICATION FAILED")
        report_lines.append("")
        report_lines.append(f"{total - passed} check(s) failed. Please review the issues above.")
        report_lines.append("Do not proceed with meta-learner training until all issues are resolved.")
    
    report_lines.extend([
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80,
    ])
    
    return "\n".join(report_lines)


def main():
    """Main verification function."""
    logger.info("=" * 80)
    logger.info("Merged OOF Predictions Verification for Meta-Learner Training")
    logger.info("=" * 80)
    
    # Step 1: Verify file exists and is readable
    success, df = verify_file_exists()
    if not success:
        logger.error("Cannot proceed: File does not exist or cannot be read")
        return
    
    # Step 2: Verify columns
    if not verify_columns(df):
        logger.error("Column verification failed")
        return
    
    # Step 3: Verify patient_id
    if not verify_patient_id(df):
        logger.error("Patient ID verification failed")
        return
    
    # Step 4: Verify fold
    if not verify_fold(df):
        logger.error("Fold verification failed")
        return
    
    # Step 5: Verify probabilities
    if not verify_probabilities(df):
        logger.error("Probability verification failed")
        return
    
    # Step 6: Verify labels
    if not verify_labels(df):
        logger.error("Label verification failed")
        return
    
    # Step 7: Verify validation split matching
    if not verify_validation_split_matching(df):
        logger.error("Validation split matching verification failed")
        return
    
    # Step 8: Verify training readiness
    if not verify_training_readiness(df):
        logger.error("Training readiness check failed")
        return
    
    # Generate and save report
    report = generate_report()
    report_file = Path('ensemble/oof_predictions/training_readiness_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"\n{'='*80}")
    logger.info("VERIFICATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Report saved to: {report_file}")
    
    # Print summary
    passed = sum(1 for r in VERIFICATION_RESULTS if r['passed'])
    total = len(VERIFICATION_RESULTS)
    
    if passed == total:
        logger.info(f"\n✅ ALL {total} CHECKS PASSED")
        logger.info("✓ Data is ready for Logistic Regression meta-learner training")
        logger.info("\nNEXT STEP: Train Logistic Regression meta-learner")
    else:
        logger.error(f"\n❌ VERIFICATION FAILED: {total - passed} check(s) failed")
        logger.error("Do not proceed with training until all issues are resolved")


if __name__ == '__main__':
    main()

