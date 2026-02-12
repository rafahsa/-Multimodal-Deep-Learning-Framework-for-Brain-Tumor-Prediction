# Final Calibration Summary - Ready for Submission

## Finalized Configuration

**Selected Run**: `2026-02-07_22-29-29_platt_seed42`

- **Calibration Method**: Platt scaling (sigmoid)
- **Split Seed**: 42
- **Calibration Fraction**: 0.70 (30% held out for threshold selection)
- **Evaluation Set**: 86 samples (held-out from OOF predictions)

## Final Operating Thresholds (Calibrated Probabilities)

| Threshold | Precision | Recall | F1 | Accuracy | FN | FP | Use Case |
|-----------|-----------|--------|----|----------|----|----|----------|
| **0.41** | 0.9365 | 0.9365 | 0.9365 | 0.9070 | 4 | 4 | **Balanced (max F1)** |
| **0.38** | 0.9091 | 0.9524 | 0.9302 | 0.8953 | 3 | 6 | **High-sensitivity (Recall ≥ 0.94)** |

## Calibration Impact

**Probability Reliability Metrics** (computed on held-out threshold selection set):
- **Brier Score**: 0.119 → 0.099 (improvement: 0.021)
- **Expected Calibration Error (ECE)**: 0.119 → 0.087 (improvement: 0.032)

**Classification Performance**: Maintained strong performance with improved probability reliability.

## Final Figures Ready for Inclusion

### (A) Reliability Diagram (Calibration Curve)
- **File**: `reliability_diagram_platt.png`
- **Location**: `ensemble/results/calibration/2026-02-07_22-29-29_platt_seed42/`
- **Content**: Shows uncalibrated vs Platt-calibrated probabilities with perfect calibration line
- **Purpose**: Demonstrates improved probability reliability

### (B) Threshold Comparison Table
- **File**: `threshold_comparison_calibrated.md`
- **Location**: `ensemble/results/calibration/`
- **Content**: Performance metrics for both operating points on calibrated probabilities
- **Purpose**: Documents final operating thresholds and their performance

## Model Definition (Final)

The final system consists of:
1. **Logistic Regression ensemble meta-learner** (unchanged)
2. **Post-hoc Platt probability calibration** (applied at inference time, optional)
3. **Threshold-based decision policy** (0.41 balanced, 0.38 high-sensitivity)

## Documentation Text (Paper/README Ready)

> We applied post-hoc Platt probability calibration to improve the reliability of ensemble predictions. Calibration significantly reduced Brier score (0.119 → 0.099, improvement: 0.021) and Expected Calibration Error (ECE: 0.119 → 0.087, improvement: 0.032) **without degrading classification performance**. Operating thresholds were re-selected on calibrated probabilities using a held-out validation set (30% of OOF predictions, seed=42) to prevent data leakage. The final operating points are: balanced threshold = 0.41 (Precision=0.9365, Recall=0.9365, F1=0.9365) and high-sensitivity threshold = 0.38 (Precision=0.9091, Recall=0.9524, F1=0.9302).

## Verification Checklist

✅ Selected run confirmed: `platt` / `seed=42` / `calibration_fraction=0.7`
✅ Reliability diagram exists and shows before/after calibration
✅ Threshold comparison table created with exact metrics from selected run
✅ Documentation text prepared (no invented numbers)
✅ No new experiments introduced
✅ Model definition unchanged (Logistic Regression + post-hoc Platt calibration)
✅ Operating thresholds match selected run (0.41 balanced, 0.38 high-sensitivity)

## Files Not Modified (As Requested)

- ROC curves (calibration does not affect ranking)
- Precision-Recall curves
- Full confusion matrix sweeps
- Model training or meta-learner architecture
- Existing ROC and confusion matrix figures

## Inference Usage

**With Calibration (Recommended)**:
```bash
# Balanced
python scripts/ensemble/test_ensemble_on_new_patients.py --calibration-mode platt --threshold 0.41

# High-sensitivity
python scripts/ensemble/test_ensemble_on_new_patients.py --calibration-mode platt --threshold 0.38
```

**Without Calibration (Backward Compatible)**:
```bash
python scripts/ensemble/test_ensemble_on_new_patients.py --threshold 0.22
```

## Status: ✅ READY FOR SUBMISSION

All final figures and documentation are prepared. No inconsistencies found. System is ready for finalization.


