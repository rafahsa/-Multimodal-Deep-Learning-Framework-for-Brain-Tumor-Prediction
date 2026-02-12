#!/usr/bin/env python3
"""
Integrate New MIL Model into Ensemble with Calibration

This script:
1. Loads new MIL OOF predictions from exp_1_1_entropy
2. Applies nested-CV-safe probability calibration (Platt scaling)
3. Replaces old MIL with new calibrated MIL in merged OOF predictions
4. Updates ensemble to use mil_prob (calibrated) instead of hgg_prob_mil

Usage:
    python scripts/ensemble/integrate_new_mil.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
OLD_MERGED_OOF = Path('ensemble/oof_predictions/merged_oof_predictions.csv')
NEW_MIL_OOF = Path('ensemble/results/mil_improvements/exp_1_1_entropy/oof_predictions.csv')
OUTPUT_MERGED_OOF = Path('ensemble/oof_predictions/merged_oof_predictions.csv')
CALIBRATORS_DIR = Path('ensemble/calibrators')
CALIBRATORS_DIR.mkdir(parents=True, exist_ok=True)

# Calibration settings
CALIBRATION_METHOD = 'platt'  # 'platt' or 'isotonic'
NUM_FOLDS = 5


def calibrate_mil_probabilities_nested_cv(
    df_mil: pd.DataFrame,
    method: str = 'platt'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Calibrate MIL probabilities using nested-CV-safe approach.
    
    For each fold:
    - Fit calibrator on all OTHER folds (inner/train)
    - Apply calibration to THIS fold (outer/validation)
    
    This ensures no data leakage.
    
    Args:
        df_mil: DataFrame with columns ['patient_id', 'fold', 'hgg_prob_mil', 'label']
        method: 'platt' (LogisticRegression) or 'isotonic' (IsotonicRegression)
    
    Returns:
        df_calibrated: DataFrame with calibrated probabilities in 'mil_prob' column
        calibrators: Dictionary of calibrators per fold
    """
    logger.info("="*80)
    logger.info("Calibrating MIL Probabilities (Nested-CV Safe)")
    logger.info("="*80)
    logger.info(f"Method: {method}")
    logger.info(f"Total samples: {len(df_mil)}")
    
    # Ensure we have required columns
    required_cols = ['patient_id', 'fold', 'hgg_prob_mil', 'label']
    missing_cols = [col for col in required_cols if col not in df_mil.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create output DataFrame
    df_calibrated = df_mil.copy()
    df_calibrated['mil_prob'] = np.nan  # Will be filled with calibrated probabilities
    
    calibrators = {}
    
    # For each fold, fit calibrator on other folds and apply to this fold
    for test_fold in range(NUM_FOLDS):
        logger.info(f"\nProcessing fold {test_fold}...")
        
        # Split: inner (train) = all folds except test_fold
        inner_mask = df_mil['fold'] != test_fold
        outer_mask = df_mil['fold'] == test_fold
        
        if not outer_mask.any():
            logger.warning(f"No samples in fold {test_fold}, skipping")
            continue
        
        # Inner (train) data for calibration
        X_inner = df_mil.loc[inner_mask, 'hgg_prob_mil'].values.reshape(-1, 1)
        y_inner = df_mil.loc[inner_mask, 'label'].values
        
        # Outer (test) data to calibrate
        X_outer = df_mil.loc[outer_mask, 'hgg_prob_mil'].values.reshape(-1, 1)
        y_outer = df_mil.loc[outer_mask, 'label'].values
        
        logger.info(f"  Inner (train) samples: {len(X_inner)}")
        logger.info(f"  Outer (test) samples: {len(X_outer)}")
        
        # Fit calibrator on inner data
        if method == 'platt':
            # Platt scaling: LogisticRegression on probabilities
            calibrator = LogisticRegression()
            calibrator.fit(X_inner, y_inner)
            
            # Apply calibration to outer data
            mil_prob_cal = calibrator.predict_proba(X_outer)[:, 1]
        elif method == 'isotonic':
            # Isotonic regression
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(X_inner.flatten(), y_inner)
            
            # Apply calibration to outer data
            mil_prob_cal = calibrator.predict(X_outer.flatten())
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        # Store calibrated probabilities
        df_calibrated.loc[outer_mask, 'mil_prob'] = mil_prob_cal
        
        # Store calibrator for this fold
        calibrators[f'fold_{test_fold}'] = {
            'method': method,
            'calibrator': calibrator,
            'n_train': len(X_inner),
            'n_test': len(X_outer)
        }
        
        # Log calibration statistics
        logger.info(f"  Uncalibrated range: [{X_outer.min():.4f}, {X_outer.max():.4f}]")
        logger.info(f"  Calibrated range: [{mil_prob_cal.min():.4f}, {mil_prob_cal.max():.4f}]")
        logger.info(f"  Calibrated mean: {mil_prob_cal.mean():.4f}")
    
    # Verify no NaN values
    if df_calibrated['mil_prob'].isna().any():
        raise ValueError("Found NaN values in calibrated probabilities")
    
    logger.info("\n✓ Calibration complete")
    logger.info(f"  Calibrated probability range: [{df_calibrated['mil_prob'].min():.4f}, {df_calibrated['mil_prob'].max():.4f}]")
    logger.info(f"  Calibrated probability mean: {df_calibrated['mil_prob'].mean():.4f}")
    
    return df_calibrated, calibrators


def load_and_merge_with_new_mil() -> pd.DataFrame:
    """
    Load existing merged OOF predictions and replace old MIL with new calibrated MIL.
    
    Returns:
        Updated merged DataFrame with mil_prob (calibrated) instead of hgg_prob_mil
    """
    logger.info("="*80)
    logger.info("Loading and Merging OOF Predictions with New MIL")
    logger.info("="*80)
    
    # Load existing merged OOF (ResNet + Swin + old MIL)
    if not OLD_MERGED_OOF.exists():
        raise FileNotFoundError(f"Existing merged OOF file not found: {OLD_MERGED_OOF}")
    
    logger.info(f"Loading existing merged OOF from: {OLD_MERGED_OOF}")
    df_merged = pd.read_csv(OLD_MERGED_OOF)
    logger.info(f"  Loaded {len(df_merged)} samples")
    logger.info(f"  Columns: {list(df_merged.columns)}")
    
    # Verify old MIL column exists
    if 'hgg_prob_mil' not in df_merged.columns:
        raise ValueError("Old MIL column 'hgg_prob_mil' not found in merged OOF")
    
    # Load new MIL OOF predictions
    if not NEW_MIL_OOF.exists():
        raise FileNotFoundError(f"New MIL OOF file not found: {NEW_MIL_OOF}")
    
    logger.info(f"\nLoading new MIL OOF from: {NEW_MIL_OOF}")
    df_new_mil = pd.read_csv(NEW_MIL_OOF)
    logger.info(f"  Loaded {len(df_new_mil)} samples")
    logger.info(f"  Columns: {list(df_new_mil.columns)}")
    
    # Verify new MIL has required columns
    if 'hgg_prob_mil' not in df_new_mil.columns:
        raise ValueError("New MIL OOF missing 'hgg_prob_mil' column")
    if 'fold' not in df_new_mil.columns:
        raise ValueError("New MIL OOF missing 'fold' column")
    if 'label' not in df_new_mil.columns:
        raise ValueError("New MIL OOF missing 'label' column")
    
    # Verify patient IDs match
    old_patients = set(df_merged['patient_id'])
    new_patients = set(df_new_mil['patient_id'])
    
    if old_patients != new_patients:
        missing = old_patients - new_patients
        extra = new_patients - old_patients
        logger.warning(f"Patient ID mismatch:")
        logger.warning(f"  Missing in new MIL: {len(missing)} patients")
        logger.warning(f"  Extra in new MIL: {len(extra)} patients")
        if len(missing) <= 10:
            logger.warning(f"  Missing: {list(missing)}")
        if len(extra) <= 10:
            logger.warning(f"  Extra: {list(extra)}")
        
        # Use intersection
        common_patients = old_patients & new_patients
        logger.info(f"  Using intersection: {len(common_patients)} patients")
        df_merged = df_merged[df_merged['patient_id'].isin(common_patients)].copy()
        df_new_mil = df_new_mil[df_new_mil['patient_id'].isin(common_patients)].copy()
    
    # Calibrate new MIL probabilities
    logger.info("\n" + "="*80)
    logger.info("Step 1: Calibrating New MIL Probabilities")
    logger.info("="*80)
    df_calibrated, calibrators = calibrate_mil_probabilities_nested_cv(
        df_new_mil,
        method=CALIBRATION_METHOD
    )
    
    # Save calibrators
    calibrator_file = CALIBRATORS_DIR / f'mil_calibrator_{CALIBRATION_METHOD}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
    # Note: sklearn calibrators may not be directly serializable, so we'll save metadata
    calibrator_metadata = {
        'method': CALIBRATION_METHOD,
        'folds': list(calibrators.keys()),
        'n_samples': len(df_calibrated),
        'calibration_date': datetime.now().isoformat()
    }
    with open(CALIBRATORS_DIR / f'mil_calibrator_metadata_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(calibrator_metadata, f, indent=2)
    logger.info(f"✓ Saved calibrator metadata to {CALIBRATORS_DIR}")
    
    # Merge: Remove old MIL, add new calibrated MIL
    logger.info("\n" + "="*80)
    logger.info("Step 2: Merging with New Calibrated MIL")
    logger.info("="*80)
    
    # Remove old MIL column
    df_merged = df_merged.drop(columns=['hgg_prob_mil'])
    logger.info("  ✓ Removed old MIL column (hgg_prob_mil)")
    
    # Merge new calibrated MIL
    mil_to_merge = df_calibrated[['patient_id', 'mil_prob']].copy()
    df_merged = df_merged.merge(mil_to_merge, on='patient_id', how='inner', validate='1:1')
    logger.info("  ✓ Added new calibrated MIL column (mil_prob)")
    
    # Verify merge
    if len(df_merged) != len(df_calibrated):
        raise ValueError(f"Merge failed: {len(df_merged)} rows after merge, expected {len(df_calibrated)}")
    
    # Reorder columns: patient_id, fold, hgg_prob_resnet, hgg_prob_swin, mil_prob, label
    df_merged = df_merged[['patient_id', 'fold', 'hgg_prob_resnet', 'hgg_prob_swin', 'mil_prob', 'label']]
    
    logger.info(f"\n✓ Merge complete")
    logger.info(f"  Final columns: {list(df_merged.columns)}")
    logger.info(f"  Total samples: {len(df_merged)}")
    logger.info(f"  MIL probability range: [{df_merged['mil_prob'].min():.4f}, {df_merged['mil_prob'].max():.4f}]")
    logger.info(f"  MIL probability mean: {df_merged['mil_prob'].mean():.4f}")
    
    return df_merged


def main():
    """Main function to integrate new MIL into ensemble."""
    logger.info("="*80)
    logger.info("INTEGRATING NEW MIL INTO ENSEMBLE")
    logger.info("="*80)
    logger.info("  - Removing old MIL (hgg_prob_mil)")
    logger.info("  - Adding new MIL with calibration (mil_prob)")
    logger.info("  - Using nested-CV-safe calibration")
    logger.info("="*80)
    
    try:
        # Load and merge with new calibrated MIL
        df_merged = load_and_merge_with_new_mil()
        
        # Save updated merged OOF predictions
        logger.info("\n" + "="*80)
        logger.info("Step 3: Saving Updated Merged OOF Predictions")
        logger.info("="*80)
        
        # Backup old file
        if OLD_MERGED_OOF.exists():
            backup_file = OLD_MERGED_OOF.parent / f"merged_oof_predictions_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            import shutil
            shutil.copy2(OLD_MERGED_OOF, backup_file)
            logger.info(f"  ✓ Backed up old merged OOF to: {backup_file}")
        
        # Save new merged file
        df_merged.to_csv(OUTPUT_MERGED_OOF, index=False)
        logger.info(f"  ✓ Saved updated merged OOF to: {OUTPUT_MERGED_OOF}")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("INTEGRATION COMPLETE")
        logger.info("="*80)
        logger.info("Changes made:")
        logger.info("  ✓ Removed: hgg_prob_mil (old MIL)")
        logger.info("  ✓ Added: mil_prob (new MIL, calibrated)")
        logger.info(f"  ✓ Updated file: {OUTPUT_MERGED_OOF}")
        logger.info("\nNext steps:")
        logger.info("  1. Run: python scripts/ensemble/train_meta_learner.py")
        logger.info("  2. Verify ensemble performance with new MIL")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Integration failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()

