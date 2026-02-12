# Post-Hoc Thresholding Results for Swin-1

## Target Constraints

- FN < 10 (FN < 5 is excellent)
- FP < 10
- Precision ≥ 0.9
- Recall ≥ 0.9

**All constraints must be met simultaneously.**

## Results Comparison

| Policy | FN (mean±std) | FP (mean±std) | Precision (mean±std) | Recall (mean±std) | Meets All? |
|--------|---------------|---------------|---------------------|-------------------|------------|
| Baseline | 10.6±3.8 | 0.4±0.5 | 0.9881±0.0146 | 0.7476±0.0911 | ❌ NO |
| RejectBand | 9.6±3.8 | 1.2±1.2 | 0.9671±0.0315 | 0.7714±0.0898 | ❌ NO |
| ConfidenceAware | 9.2±4.0 | 1.8±1.5 | 0.9528±0.0367 | 0.7810±0.0957 | ❌ NO |
| FoldCalibrated | 10.6±3.8 | 0.4±0.5 | 0.9881±0.0146 | 0.7476±0.0911 | ❌ NO |

## Executive Summary

**❌ NO POLICY MEETS ALL CONSTRAINTS**

None of the thresholding policies achieve:
- FN < 10 AND
- FP < 10 AND
- Precision ≥ 0.9 AND
- Recall ≥ 0.9

**Next Step:** Proceed to Part B (Feature-level rescue)
