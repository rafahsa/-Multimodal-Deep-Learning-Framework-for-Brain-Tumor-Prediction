# MIL Improvement Evaluation: Final Decision Report

**Date**: 2026-02-09 21:39:09

## Executive Summary

This report presents the results of a systematic evaluation of limited
improvements to the Dual-Stream MIL model, using strict nested cross-validation.

---

## Experimental Protocol

- **Evaluation**: Nested CV (5-fold patient-level StratifiedKFold)
- **Inner split**: 70% calibration/threshold selection, 30% meta-learner training
- **Evaluation**: Outer-test folds only (never seen during training)
- **Baseline comparisons**: Original MIL, Enhanced meta-features ensemble

---

## Results Summary

| Experiment | Sampling | Bag Size | FN (mean ± std) | Cost (mean ± std) | Recall (mean ± std) |
|------------|----------|----------|-----------------|-------------------|---------------------|
| exp_1_1_entropy | entropy | 32 | 0.80 ± 1.60 | 16.40 ± 2.80 | 0.9810 ± 0.0381 |

---

## Final Verdict

### Did limited improvements help?

[To be filled based on results]

### Which modification (if any) is worth keeping?

[To be filled based on results]

### Should MIL remain in ensemble?

[To be filled based on results]

### Or rely primarily on CNN-based models + meta-features?

[To be filled based on results]
