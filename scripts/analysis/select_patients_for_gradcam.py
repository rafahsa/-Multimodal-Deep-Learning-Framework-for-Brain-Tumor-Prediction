#!/usr/bin/env python3
"""
Select Representative Patients for Grad-CAM Visualization

This script selects 12 patients (3 per category: TP, TN, FP, FN) from ResNet50-3D
OOF predictions for Grad-CAM visualization. Selection criteria:
- TP: High-confidence correct HGG predictions
- TN: High-confidence correct LGG predictions
- FP: Borderline or high-impact false positives (LGG predicted as HGG)
- FN: Borderline or high-impact false negatives (HGG predicted as LGG)

Author: Medical Imaging Pipeline
"""

import sys
from pathlib import Path

# Set project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple
import SimpleITK as sitk

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
OOF_PREDICTIONS_FILE = PROJECT_ROOT / 'ensemble' / 'oof_predictions' / 'resnet50_3d_oof.csv'
DATA_ROOT = PROJECT_ROOT / 'data' / 'processed' / 'stage_4_resize' / 'train'
OUTPUT_FILE = PROJECT_ROOT / 'data' / 'selected_patients.txt'
OUTPUT_SUMMARY = PROJECT_ROOT / 'data' / 'selected_patients_summary.csv'
THRESHOLD = 0.5  # Classification threshold


def load_predictions() -> pd.DataFrame:
    """Load ResNet50-3D OOF predictions."""
    logger.info(f"Loading predictions from: {OOF_PREDICTIONS_FILE}")
    df = pd.read_csv(OOF_PREDICTIONS_FILE)
    logger.info(f"Loaded {len(df)} predictions")
    return df


def compute_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute predicted labels and categories."""
    df = df.copy()
    
    # Compute predicted label (threshold = 0.5)
    df['predicted_label'] = (df['hgg_prob'] >= THRESHOLD).astype(int)
    
    # Categorize: TP, TN, FP, FN
    # TP: label=1, predicted=1 (HGG correctly predicted as HGG)
    # TN: label=0, predicted=0 (LGG correctly predicted as LGG)
    # FP: label=0, predicted=1 (LGG incorrectly predicted as HGG)
    # FN: label=1, predicted=0 (HGG incorrectly predicted as LGG)
    
    conditions = [
        (df['label'] == 1) & (df['predicted_label'] == 1),  # TP
        (df['label'] == 0) & (df['predicted_label'] == 0),  # TN
        (df['label'] == 0) & (df['predicted_label'] == 1),  # FP
        (df['label'] == 1) & (df['predicted_label'] == 0),  # FN
    ]
    choices = ['TP', 'TN', 'FP', 'FN']
    df['category'] = np.select(conditions, choices, default='UNKNOWN')
    
    # Compute confidence (distance from threshold)
    df['confidence'] = np.abs(df['hgg_prob'] - THRESHOLD)
    
    return df


def verify_patient_exists(patient_id: str) -> Tuple[bool, str]:
    """
    Verify patient volume exists in data directory.
    
    Returns:
        (exists, class_name) tuple
    """
    for class_name in ['LGG', 'HGG']:
        patient_dir = DATA_ROOT / class_name / patient_id
        if patient_dir.exists():
            # Check if all modalities exist
            modalities = ['t1', 't1ce', 't2', 'flair']
            all_exist = True
            for mod in modalities:
                mod_path = patient_dir / f"{patient_id}_{mod}.nii.gz"
                if not mod_path.exists():
                    mod_path = patient_dir / f"{patient_id}_{mod}.nii"
                    if not mod_path.exists():
                        all_exist = False
                        break
            if all_exist:
                return True, class_name
    return False, None


def select_patients_by_category(
    df: pd.DataFrame,
    category: str,
    n: int = 3,
    prefer_high_confidence: bool = True
) -> List[Dict]:
    """
    Select n patients from a category.
    
    Args:
        df: DataFrame with predictions
        category: 'TP', 'TN', 'FP', or 'FN'
        n: Number of patients to select
        prefer_high_confidence: If True, prefer high confidence for TP/TN, 
                                prefer borderline for FP/FN
    
    Returns:
        List of patient dicts with metadata
    """
    category_df = df[df['category'] == category].copy()
    
    if len(category_df) == 0:
        logger.warning(f"No patients found in category {category}")
        return []
    
    # Selection strategy:
    # - TP/TN: Prefer high confidence (large |prob - 0.5|)
    # - FP/FN: Prefer borderline (small |prob - 0.5|) for high-impact errors
    if category in ['TP', 'TN']:
        # High confidence is good
        category_df = category_df.sort_values('confidence', ascending=False)
    else:  # FP, FN
        # Borderline errors are more interesting (but not too close to threshold)
        # Prefer errors with confidence between 0.1 and 0.3
        category_df['error_interest'] = category_df['confidence'].apply(
            lambda x: 0.1 <= x <= 0.3
        )
        # Sort: interesting errors first, then by confidence
        category_df = category_df.sort_values(
            ['error_interest', 'confidence'],
            ascending=[False, True]
        )
    
    selected = []
    for _, row in category_df.iterrows():
        patient_id = row['patient_id']
        
        # Verify patient exists
        exists, class_name = verify_patient_exists(patient_id)
        if not exists:
            logger.warning(f"Patient {patient_id} not found in data directory, skipping")
            continue
        
        selected.append({
            'patient_id': patient_id,
            'true_label': int(row['label']),
            'predicted_label': int(row['predicted_label']),
            'hgg_prob': float(row['hgg_prob']),
            'category': category,
            'confidence': float(row['confidence']),
            'class_name': class_name,
            'fold': int(row['fold'])
        })
        
        if len(selected) >= n:
            break
    
    if len(selected) < n:
        logger.warning(f"Only found {len(selected)}/{n} patients for category {category}")
    
    return selected


def select_all_patients(df: pd.DataFrame) -> List[Dict]:
    """Select 3 patients from each category."""
    all_selected = []
    
    for category in ['TP', 'TN', 'FP', 'FN']:
        logger.info(f"\nSelecting patients for {category}...")
        selected = select_patients_by_category(df, category, n=3)
        
        # Special handling for FN: if we don't have enough, use low-confidence HGG cases
        if category == 'FN' and len(selected) < 3:
            logger.warning(f"Only {len(selected)} actual FN found. Looking for low-confidence HGG cases...")
            # Find HGG patients with low confidence (prob < 0.7) as "near-miss" cases
            # These are HGG cases that were correctly predicted but with low confidence
            low_conf_hgg = df[
                (df['label'] == 1) & 
                (df['hgg_prob'] < 0.7) &
                (df['hgg_prob'] >= 0.5) &  # Still correctly predicted as HGG
                (~df['patient_id'].isin([s['patient_id'] for s in selected]))
            ].copy()
            low_conf_hgg = low_conf_hgg.sort_values('hgg_prob', ascending=True)  # Lowest prob first
            
            for _, row in low_conf_hgg.iterrows():
                if len(selected) >= 3:
                    break
                patient_id = row['patient_id']
                exists, class_name = verify_patient_exists(patient_id)
                if exists:
                    # Mark as "near-miss FN" (low confidence HGG)
                    selected.append({
                        'patient_id': patient_id,
                        'true_label': int(row['label']),
                        'predicted_label': int(row['predicted_label']),
                        'hgg_prob': float(row['hgg_prob']),
                        'category': 'FN_nearmiss',  # Mark as near-miss
                        'confidence': float(row['confidence']),
                        'class_name': class_name,
                        'fold': int(row['fold'])
                    })
                    logger.info(f"    Added near-miss FN: {patient_id} (prob={row['hgg_prob']:.3f})")
        
        # For other categories, try to fill with any available
        if len(selected) < 3 and category != 'FN':
            logger.warning(f"Only found {len(selected)}/{3} patients for {category}. Trying to fill...")
            category_df = df[df['category'] == category].copy()
            for _, row in category_df.iterrows():
                if len(selected) >= 3:
                    break
                patient_id = row['patient_id']
                if patient_id not in [s['patient_id'] for s in selected]:
                    exists, class_name = verify_patient_exists(patient_id)
                    if exists:
                        selected.append({
                            'patient_id': patient_id,
                            'true_label': int(row['label']),
                            'predicted_label': int(row['predicted_label']),
                            'hgg_prob': float(row['hgg_prob']),
                            'category': category,
                            'confidence': float(row['confidence']),
                            'class_name': class_name,
                            'fold': int(row['fold'])
                        })
        
        all_selected.extend(selected)
        logger.info(f"  Selected {len(selected)} patients for {category}")
        for p in selected:
            category_label = p['category']
            logger.info(f"    - {p['patient_id']}: prob={p['hgg_prob']:.3f}, "
                       f"true={p['true_label']}, pred={p['predicted_label']}, "
                       f"conf={p['confidence']:.3f}, cat={category_label}")
    
    return all_selected


def save_selection(selected: List[Dict]):
    """Save selected patients to files."""
    # Save patient IDs to text file
    patient_ids = [p['patient_id'] for p in selected]
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        for patient_id in patient_ids:
            f.write(f"{patient_id}\n")
    
    logger.info(f"\n✓ Saved {len(patient_ids)} patient IDs to: {OUTPUT_FILE}")
    
    # Save summary CSV
    summary_df = pd.DataFrame(selected)
    summary_df = summary_df[[
        'patient_id', 'category', 'true_label', 'predicted_label',
        'hgg_prob', 'confidence', 'class_name', 'fold'
    ]]
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)
    logger.info(f"✓ Saved summary to: {OUTPUT_SUMMARY}")
    
    # Print summary table
    logger.info("\n" + "="*80)
    logger.info("SELECTED PATIENTS SUMMARY")
    logger.info("="*80)
    logger.info(f"\n{summary_df.to_string(index=False)}")
    
    # Print category counts
    logger.info("\n" + "="*80)
    logger.info("CATEGORY COUNTS")
    logger.info("="*80)
    for category in ['TP', 'TN', 'FP', 'FN']:
        count = len([p for p in selected if p['category'] == category])
        logger.info(f"  {category}: {count}")
    fn_nearmiss_count = len([p for p in selected if p['category'] == 'FN_nearmiss'])
    if fn_nearmiss_count > 0:
        logger.info(f"  FN_nearmiss: {fn_nearmiss_count} (low-confidence HGG cases used to fill FN category)")
        logger.info(f"  Total FN-like: {count + fn_nearmiss_count}")


def validate_selection(selected: List[Dict]) -> bool:
    """Validate that selection meets requirements."""
    errors = []
    warnings = []
    
    # Check count
    if len(selected) != 12:
        errors.append(f"Expected 12 patients, got {len(selected)}")
    
    # Check category balance (allow FN_nearmiss to count as FN)
    fn_count = len([p for p in selected if p['category'] in ['FN', 'FN_nearmiss']])
    for category in ['TP', 'TN', 'FP']:
        count = len([p for p in selected if p['category'] == category])
        if count != 3:
            errors.append(f"Expected 3 patients for {category}, got {count}")
    
    if fn_count != 3:
        warnings.append(f"Expected 3 patients for FN (including near-miss), got {fn_count}")
        # This is a warning, not an error, since we may have limited FNs
    
    # Check for duplicates
    patient_ids = [p['patient_id'] for p in selected]
    if len(patient_ids) != len(set(patient_ids)):
        errors.append("Duplicate patient IDs found")
    
    # Check all patients exist
    for p in selected:
        exists, _ = verify_patient_exists(p['patient_id'])
        if not exists:
            errors.append(f"Patient {p['patient_id']} does not exist in data directory")
    
    if warnings:
        logger.warning("Validation warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    if errors:
        logger.error("Validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("✓ Validation passed: All requirements met")
    if warnings:
        logger.info("  (Some warnings noted above)")
    return True


def main():
    logger.info("="*80)
    logger.info("SELECTING PATIENTS FOR GRAD-CAM VISUALIZATION")
    logger.info("="*80)
    
    # Load predictions
    df = load_predictions()
    
    # Compute predictions and categories
    df = compute_predictions(df)
    
    # Print category distribution
    logger.info("\nCategory distribution in OOF predictions:")
    for category in ['TP', 'TN', 'FP', 'FN']:
        count = len(df[df['category'] == category])
        logger.info(f"  {category}: {count}")
    
    # Select patients
    selected = select_all_patients(df)
    
    # Validate
    if not validate_selection(selected):
        logger.error("Selection validation failed. Please review and fix.")
        return 1
    
    # Save
    save_selection(selected)
    
    logger.info("\n" + "="*80)
    logger.info("SELECTION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nTo use these patients with Grad-CAM:")
    logger.info(f"  python scripts/analysis/generate_cnn_gradcam_3d.py \\")
    logger.info(f"    --checkpoint \"AUTO\" \\")
    logger.info(f"    --patient_ids_file {OUTPUT_FILE} \\")
    logger.info(f"    --fold 0")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

