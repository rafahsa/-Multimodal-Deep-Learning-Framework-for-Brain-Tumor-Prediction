# MIL-Only Calibration: Nested CV Evaluation Report

**Date**: 2026-02-09 01:59:10

## Executive Summary

This report evaluates the impact of calibrating **ONLY the MIL model probabilities**
on ensemble performance, using strict nested cross-validation.

**Key Findings**:

---

## Results Comparison

| Metric | Baseline (No Cal) | MIL-Platt | MIL-Isotonic |
|--------|------------------|-----------|---------------|
| FN (mean ± std) | 2.20 ± 1.17 | 2.80 ± 1.47 | 3.20 ± 2.04 |
| FP (mean ± std) | 9.60 ± 3.01 | 8.40 ± 2.73 | 9.40 ± 1.74 |
| Cost (mean ± std) | 14.00 ± 1.41 | 14.00 ± 2.45 | 15.80 ± 3.71 |
| Recall (mean ± std) | 0.9476 ± 0.0278 | 0.9333 ± 0.0350 | 0.9238 ± 0.0486 |
| Precision (mean ± std) | 0.8097 ± 0.0499 | 0.8269 ± 0.0458 | 0.8061 ± 0.0260 |

### MIL Brier Score Improvement

- **Platt**: 0.2432 → 0.1756 (improvement: +0.0676)
- **Isotonic**: 0.2432 → 0.1724 (improvement: +0.0709)

---

## Verdict

### Does MIL calibration reduce FN?

❌ **Platt**: FN increased by 0.60 (2.20 → 2.80)
❌ **Isotonic**: FN increased by 1.00 (2.20 → 3.20)

### Does it improve ensemble recall?

❌ **Platt**: Recall decreased by 0.0143 (0.9476 → 0.9333)
❌ **Isotonic**: Recall decreased by 0.0238 (0.9476 → 0.9238)

### Is the improvement stable across folds?

⚠️ **Platt**: Less stable (FN std: 1.47 vs baseline 1.17)
⚠️ **Isotonic**: Less stable (FN std: 2.04 vs baseline 1.17)

---

## Final Recommendation

Based on the nested CV evaluation:

❌ **MIL calibration does NOT improve ensemble performance**.

**Conclusion**: The limitation is architectural. MIL model probabilities
are not the bottleneck. Consider:
- Improving MIL model architecture
- Improving MIL training procedure
- Replacing MIL with a better base model

---

## Methodology

### Calibration Protocol

- **Method**: Platt scaling and Isotonic regression
- **Scope**: MIL probabilities only (ResNet and Swin unchanged)
- **Fitting**: Only on outer-train data within each fold
- **Evaluation**: Only on outer-test data (never seen during calibration)

### Nested CV Structure

- **Outer folds**: 5-fold patient-level StratifiedKFold
- **Inner split**: 70% calibration/threshold selection, 30% meta-learner training
- **Threshold selection**: Cost-sensitive (minimize 2×FN + FP)
- **Meta-learner**: Logistic Regression (class_weight='balanced')

### Metrics

- All metrics computed on outer-test folds only
- Aggregated as mean ± std across folds
- Brier score computed for MIL probabilities (before vs after calibration)
