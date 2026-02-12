# ROI-Based MIL Experiment: Results Report

**Date:** [FILL IN]  
**Experiment:** ROI-guided instance sampling for MIL model  
**Baseline:** Current ensemble with MIL coefficient = 0.092570

---

## Changes Made

### Files Modified:
1. ✅ `scripts/prepare_folds_with_seg.py` - Added segmentation paths to fold CSVs
2. ✅ `utils/dataset_mil_roi.py` - ROI-enabled MIL dataset class
3. ⏳ `scripts/training/train_dual_stream_mil.py` - Modified to use ROI dataset
4. ⏳ Training runs completed for all 5 folds
5. ⏳ OOF predictions generated
6. ⏳ Meta-learner re-trained

### Key Implementation Details:
- **ROI Sampling:** 70% from tumor region (seg > 0), 30% from context
- **Bag Size:** 32 (unchanged from baseline)
- **Folds:** Same patient-level splits (no data leakage)
- **Training:** Same hyperparameters as baseline MIL

---

## Results Comparison

### Ensemble-Level Metrics (Nested CV, Mean ± Std)

| Metric | Baseline | ROI-Guided MIL | Change |
|--------|----------|----------------|--------|
| **FN** | [BASELINE] | [ROI] | [DIFF] |
| **FP** | [BASELINE] | [ROI] | [DIFF] |
| **Recall** | [BASELINE] | [ROI] | [DIFF] |
| **Precision** | [BASELINE] | [ROI] | [DIFF] |
| **F1** | [BASELINE] | [ROI] | [DIFF] |
| **Accuracy** | [BASELINE] | [ROI] | [DIFF] |

### Meta-Learner Coefficients

| Base Model | Baseline Coefficient | ROI-Guided Coefficient | Change |
|------------|----------------------|------------------------|--------|
| **SwinUNETR-3D** | 4.137673 | [ROI] | [DIFF] |
| **ResNet50-3D** | 0.561032 | [ROI] | [DIFF] |
| **DualStreamMIL-3D** | 0.092570 | [ROI] | **[TARGET: >0.15]** |
| **Intercept** | -2.120502 | [ROI] | [DIFF] |

---

## Success Criteria Evaluation

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| **FP Reduction** | Decrease by ≥1 | [RESULT] | ✅/❌ |
| **MIL Coefficient** | Increase (target >0.15) | [RESULT] | ✅/❌ |
| **Recall** | ≥ 0.92 (no drop) | [RESULT] | ✅/❌ |
| **FN** | No significant increase | [RESULT] | ✅/❌ |

---

## Interpretation

### ROI Sampling Impact:
[FILL IN: Describe whether ROI sampling improved MIL contribution]

### Ensemble Behavior:
[FILL IN: How did the ensemble change? Did MIL become more useful?]

### Recommendations:
[FILL IN: Is ROI-guided sampling worth pursuing further?]

---

## Next Steps

[FILL IN based on results]

---

*Generated from: ROI MIL Experiment*  
*Baseline metrics from: `ensemble/models/meta_learner_logistic_regression.joblib`*

