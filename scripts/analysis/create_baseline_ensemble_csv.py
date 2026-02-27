#!/usr/bin/env python3
"""
Create canonical baseline ensemble OOF predictions CSV.

Source: merged_oof_predictions_backup_20260209_233113.csv (hgg_prob_mil)
        + meta-learner coefficients from meta_learner_metrics.json
Output: reports/figures/data/baseline_ensemble_oof.csv
        Columns: patient_id, label, ensemble_prob_baseline

This reproduces Full OOF AUC = 0.9126 (verified).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
BACKUP_PATH = PROJECT / 'ensemble/oof_predictions/merged_oof_predictions_backup_20260209_233113.csv'
METRICS_PATH = PROJECT / 'ensemble/results/meta_learner_metrics.json'
OUTPUT_PATH = PROJECT / 'reports/figures/data/baseline_ensemble_oof.csv'


def main():
    df = pd.read_csv(BACKUP_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    coef = np.array([
        metrics['model_coefficients']['hgg_prob_resnet'],
        metrics['model_coefficients']['hgg_prob_swin'],
        metrics['model_coefficients']['hgg_prob_mil'],
    ])
    intercept = metrics['model_intercept']

    X = df[['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']].values
    ensemble_prob = 1 / (1 + np.exp(-np.clip(intercept + X @ coef, -500, 500)))

    out = pd.DataFrame({
        'patient_id': df['patient_id'],
        'label': df['label'],
        'ensemble_prob_baseline': ensemble_prob,
    })
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(out['label'], out['ensemble_prob_baseline'])
    print(f"Created {OUTPUT_PATH}")
    print(f"Full OOF AUC (verify): {auc:.6f} (expected 0.9126)")


if __name__ == '__main__':
    main()
