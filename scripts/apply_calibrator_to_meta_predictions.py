#!/usr/bin/env python3
"""
Apply saved Platt calibrator to meta-learner predictions.

This script:
1. Loads the saved Platt calibrator
2. Loads meta_decision_predictions.csv
3. Applies calibration to meta_prob column
4. Saves calibrated predictions to a new CSV file
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
CALIBRATOR_PATH = Path('ensemble/results/calibration/2026-02-07_22-29-29_platt_seed42/calibrator_platt.joblib')
INPUT_CSV = Path('ensemble/results/meta_decision/meta_decision_predictions.csv')
OUTPUT_CSV = Path('ensemble/results/meta_decision/meta_decision_predictions_calibrated.csv')

def apply_platt_calibrator(calibrator: dict, y_proba_uncal: np.ndarray) -> np.ndarray:
    """
    Apply Platt calibrator to uncalibrated probabilities.
    
    Args:
        calibrator: Dictionary with 'type' and 'model' keys
        y_proba_uncal: Uncalibrated probabilities (1D array)
    
    Returns:
        Calibrated probabilities (1D array)
    """
    if calibrator['type'] != 'platt':
        raise ValueError(f"Expected Platt calibrator, got {calibrator['type']}")
    
    platt_model = calibrator['model']
    
    # Clip probabilities to avoid log(0) and log(1)
    y_proba_clipped = np.clip(y_proba_uncal, 1e-7, 1 - 1e-7)
    
    # Transform to log-odds
    log_odds = np.log(y_proba_clipped / (1 - y_proba_clipped))
    
    # Apply Platt scaling (model expects log-odds as input)
    y_proba_cal = platt_model.predict_proba(log_odds.reshape(-1, 1))[:, 1]
    
    return y_proba_cal


def main():
    logger.info("=" * 80)
    logger.info("Applying Platt Calibrator to Meta-Learner Predictions")
    logger.info("=" * 80)
    
    # Load calibrator
    logger.info(f"Loading calibrator from: {CALIBRATOR_PATH}")
    if not CALIBRATOR_PATH.exists():
        raise FileNotFoundError(f"Calibrator not found: {CALIBRATOR_PATH}")
    
    calibrator = joblib.load(CALIBRATOR_PATH)
    logger.info(f"✓ Loaded calibrator: type={calibrator['type']}")
    
    # Load predictions
    logger.info(f"Loading predictions from: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    
    df = pd.read_csv(INPUT_CSV)
    logger.info(f"✓ Loaded {len(df)} samples")
    logger.info(f"  Columns: {list(df.columns)}")
    
    # Verify required columns
    if 'meta_prob' not in df.columns:
        raise ValueError("Column 'meta_prob' not found in input CSV")
    
    # Extract uncalibrated probabilities
    meta_prob_uncal = df['meta_prob'].values
    logger.info(f"  Uncalibrated prob range: [{meta_prob_uncal.min():.4f}, {meta_prob_uncal.max():.4f}]")
    logger.info(f"  Uncalibrated prob mean: {meta_prob_uncal.mean():.4f}")
    
    # Apply calibration
    logger.info("Applying Platt calibration...")
    meta_prob_cal = apply_platt_calibrator(calibrator, meta_prob_uncal)
    logger.info(f"  Calibrated prob range: [{meta_prob_cal.min():.4f}, {meta_prob_cal.max():.4f}]")
    logger.info(f"  Calibrated prob mean: {meta_prob_cal.mean():.4f}")
    
    # Create output dataframe
    df_output = pd.DataFrame({
        'patient_id': df['patient_id'],
        'fold': df['fold'],
        'label': df['label'],
        'meta_prob': meta_prob_uncal,
        'meta_prob_calibrated': meta_prob_cal
    })
    
    # Verify no rows dropped or reordered
    assert len(df_output) == len(df), f"Row count mismatch: {len(df_output)} != {len(df)}"
    assert (df_output['patient_id'] == df['patient_id']).all(), "Patient IDs do not match"
    assert (df_output['fold'] == df['fold']).all(), "Folds do not match"
    assert (df_output['label'] == df['label']).all(), "Labels do not match"
    
    # Save output
    logger.info(f"Saving calibrated predictions to: {OUTPUT_CSV}")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(OUTPUT_CSV, index=False)
    
    logger.info("=" * 80)
    logger.info("Calibration Complete!")
    logger.info(f"  Input samples: {len(df)}")
    logger.info(f"  Output samples: {len(df_output)}")
    logger.info(f"  Output file: {OUTPUT_CSV}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

