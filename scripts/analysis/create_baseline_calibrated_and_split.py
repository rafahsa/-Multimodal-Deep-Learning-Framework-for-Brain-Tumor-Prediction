#!/usr/bin/env python3
"""
STEP 1: Create calibrated baseline ensemble probabilities.
STEP 2: Reproduce 70/30 split (seed=42) and save held-out patient list.

Uses:
- reports/figures/data/baseline_ensemble_oof.csv (uncalibrated)
- ensemble/results/calibration/2026-02-07_22-29-29_platt_seed42/calibrator_platt.joblib
- ensemble/oof_predictions/merged_oof_predictions.csv (for split row order)
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT = Path(__file__).resolve().parents[2]
BASELINE_CSV = PROJECT / 'reports/figures/data/baseline_ensemble_oof.csv'
CALIBRATOR_PATH = PROJECT / 'ensemble/results/calibration/2026-02-07_22-29-29_platt_seed42/calibrator_platt.joblib'
MERGED_OOF = PROJECT / 'ensemble/oof_predictions/merged_oof_predictions.csv'
OUT_CALIBRATED = PROJECT / 'reports/figures/data/baseline_ensemble_oof_calibrated.csv'
OUT_HELDOUT = PROJECT / 'reports/figures/data/threshold_selection_set_seed42.csv'


def apply_platt(calibrator, y_proba):
    if calibrator['type'] != 'platt':
        raise ValueError(f"Expected platt, got {calibrator['type']}")
    clipped = np.clip(y_proba, 1e-7, 1 - 1e-7)
    log_odds = np.log(clipped / (1 - clipped))
    return calibrator['model'].predict_proba(log_odds.reshape(-1, 1))[:, 1]


def main():
    # Load baseline
    df = pd.read_csv(BASELINE_CSV)
    assert len(df) == 285
    p_uncal = df['ensemble_prob_baseline'].values

    # Load calibrator
    calibrator = joblib.load(CALIBRATOR_PATH)
    p_cal = apply_platt(calibrator, p_uncal)

    # Stats
    print("STEP 1: Calibrated baseline ensemble")
    print(f"  Uncalibrated: min={p_uncal.min():.4f}, max={p_uncal.max():.4f}, mean={p_uncal.mean():.4f}")
    print(f"  Calibrated:   min={p_cal.min():.4f}, max={p_cal.max():.4f}, mean={p_cal.mean():.4f}")

    out = pd.DataFrame({
        'patient_id': df['patient_id'],
        'label': df['label'],
        'ensemble_prob_baseline': p_uncal,
        'ensemble_prob_baseline_calibrated': p_cal,
    })
    out.to_csv(OUT_CALIBRATED, index=False)
    print(f"  Saved {OUT_CALIBRATED}, n={len(out)}")

    # STEP 2: Reproduce split using merged_oof row order (same as calibration script)
    merged = pd.read_csv(MERGED_OOF)
    assert len(merged) == 285
    y = merged['label'].values
    indices = np.arange(len(merged))
    _, thr_idx = train_test_split(
        indices, test_size=0.30, stratify=y, random_state=42
    )
    thr_patients = merged.iloc[thr_idx]['patient_id'].tolist()
    assert len(thr_patients) == 86

    heldout = pd.DataFrame({'patient_id': thr_patients})
    heldout.to_csv(OUT_HELDOUT, index=False)
    print(f"\nSTEP 2: Threshold selection set")
    print(f"  Saved {OUT_HELDOUT}, n={len(heldout)}")


if __name__ == '__main__':
    main()
