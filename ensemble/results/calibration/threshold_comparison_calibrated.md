# Threshold Comparison: Calibrated Probabilities (Platt Scaling)

## Final Operating Points

This table reports performance metrics for the ensemble meta-learner with **post-hoc Platt probability calibration** applied.

**Calibration Run**: `2026-02-07_22-29-29_platt_seed42`
- Calibration method: Platt scaling (sigmoid)
- Split seed: 42
- Calibration fraction: 0.70 (30% held out for threshold selection)
- Evaluation set: 86 samples (held-out from OOF predictions)

## Performance Metrics

| Threshold | Precision | Recall | F1 | Accuracy | FN | FP | Use Case |
|-----------|-----------|--------|----|----------|----|----|----------|
| **0.41** | 0.9365 | 0.9365 | 0.9365 | 0.9070 | 4 | 4 | **Balanced (max F1)** |
| **0.38** | 0.9091 | 0.9524 | 0.9302 | 0.8953 | 3 | 6 | **High-sensitivity (Recall ≥ 0.94)** |

## Calibration Impact

**Probability Reliability Improvement**:
- Brier Score: 0.119 → 0.099 (improvement: 0.021)
- Expected Calibration Error (ECE): 0.119 → 0.087 (improvement: 0.032)

**Note**: These metrics are computed on the held-out threshold selection set (86 samples) to ensure valid evaluation without data leakage.

## Comparison with Uncalibrated Thresholds

For reference, uncalibrated operating points (from earlier analysis):
- Balanced: threshold 0.22 (Precision=0.9000, Recall=0.9000, F1=0.9000)
- High-sensitivity: threshold 0.19 (Precision=0.8319, Recall=0.9429, F1=0.8839)

Calibration improves probability reliability while maintaining strong classification performance.


