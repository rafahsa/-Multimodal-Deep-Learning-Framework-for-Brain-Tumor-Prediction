"""
Step 2: Nested-CV Safe Calibration

Applies probability calibration to Swin, ResNet, and MIL probabilities.
Uses nested-CV safe approach: for each fold, fit calibrator on other folds only.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import joblib

logger = logging.getLogger(__name__)

NUM_FOLDS = 5
CALIBRATION_METHOD = 'platt'  # 'platt' or 'isotonic'


def apply_nested_cv_calibration(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Apply nested-CV safe calibration to probabilities.
    
    For each fold:
    - Fit calibrator on all other folds (inner/train)
    - Apply calibration to this fold (outer/validation)
    """
    logger.info("Applying nested-CV safe calibration...")
    
    df_cal = df.copy()
    df_cal['swin_prob_cal'] = np.nan
    df_cal['resnet_prob_cal'] = np.nan
    df_cal['mil_prob_cal'] = df_cal['mil_prob'].copy()  # Already calibrated, keep as is
    
    calibrators = {}
    
    # Calibrate Swin and ResNet probabilities
    for prob_col, cal_col in [('swin_prob_tta', 'swin_prob_cal'), 
                               ('resnet_prob_tta', 'resnet_prob_cal')]:
        if prob_col not in df_cal.columns:
            logger.warning(f"Column {prob_col} not found, skipping calibration")
            continue
        
        logger.info(f"\nCalibrating {prob_col}...")
        
        for test_fold in range(NUM_FOLDS):
            # Inner (train): all folds except test_fold
            inner_mask = df_cal['fold'] != test_fold
            outer_mask = df_cal['fold'] == test_fold
            
            if not outer_mask.any():
                continue
            
            # Get probabilities and labels
            X_inner = df_cal.loc[inner_mask, prob_col].values.reshape(-1, 1)
            y_inner = df_cal.loc[inner_mask, 'label'].values
            X_outer = df_cal.loc[outer_mask, prob_col].values.reshape(-1, 1)
            
            # Fit calibrator
            if CALIBRATION_METHOD == 'platt':
                calibrator = LogisticRegression()
                calibrator.fit(X_inner, y_inner)
                cal_probs = calibrator.predict_proba(X_outer)[:, 1]
            elif CALIBRATION_METHOD == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(X_inner.flatten(), y_inner)
                cal_probs = calibrator.predict(X_outer.flatten())
            else:
                raise ValueError(f"Unknown calibration method: {CALIBRATION_METHOD}")
            
            # Store calibrated probabilities
            df_cal.loc[outer_mask, cal_col] = cal_probs
            
            # Store calibrator
            calibrators[f'{cal_col}_fold_{test_fold}'] = calibrator
        
        logger.info(f"  ✓ Calibrated {prob_col}")
    
    # Save calibrators
    calibrator_file = output_dir / 'calibrators.joblib'
    joblib.dump(calibrators, calibrator_file)
    logger.info(f"\n✓ Saved calibrators to: {calibrator_file}")
    
    # Save calibrated predictions
    output_file = output_dir / 'oof_predictions_with_calibration.csv'
    df_cal.to_csv(output_file, index=False)
    logger.info(f"✓ Saved calibrated predictions to: {output_file}")
    
    # Statistics
    logger.info("\nCalibration Statistics:")
    for prob_col, cal_col in [('swin_prob_tta', 'swin_prob_cal'), 
                               ('resnet_prob_tta', 'resnet_prob_cal')]:
        if prob_col in df_cal.columns and cal_col in df_cal.columns:
            logger.info(f"{cal_col}:")
            logger.info(f"  Uncalibrated range: [{df_cal[prob_col].min():.4f}, {df_cal[prob_col].max():.4f}]")
            logger.info(f"  Calibrated range: [{df_cal[cal_col].min():.4f}, {df_cal[cal_col].max():.4f}]")
    
    return df_cal


