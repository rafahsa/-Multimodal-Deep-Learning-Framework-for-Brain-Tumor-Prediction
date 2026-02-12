# Ensemble A/B Test: Baseline vs ROI MIL
## Direct Comparison Report

**Date:** 2026-02-11  
**Test Type:** A/B Comparison (OOF Predictions)  
**Threshold:** 0.22  
**Status:** ✅ Complete

---

## Executive Summary

**Recommendation: ❌ DO NOT REPLACE**

The ROI MIL ensemble variant shows **slight degradation** compared to the baseline ensemble:
- **FN rate increased** (+0.48%, 23 → 24 false negatives)
- **HGG recall decreased** (-0.48%, 0.8905 → 0.8857)
- **AUC slightly degraded** (-0.06%, 0.9074 → 0.9068)

While the differences are small, they represent a **regression** in critical medical metrics (FN rate and recall). The baseline ensemble should be retained.

---

## Step 1: New Ensemble OOF Created

### Files Created
- **New merged OOF:** `ensemble/oof_predictions/merged_oof_predictions_roi_mil.csv`
- **Source:** Replaced `mil_prob` with calibrated ROI MIL probabilities
- **Verification:**
  - ✅ Same 285 patients as original
  - ✅ No missing values
  - ✅ Only `mil_prob` column differs
  - ✅ Probability scale compatible (mean: 0.7324 → 0.7384, diff: 0.0060)

### Probability Comparison
| Metric | Baseline MIL | ROI MIL | Difference |
|--------|--------------|---------|------------|
| Mean | 0.7324 | 0.7384 | +0.0060 |
| Range | [0.6754, 0.8084] | [0.7263, 0.7584] | Similar |

---

## Step 2: Meta-Learner Training

### Baseline Ensemble
- **Model:** LogisticRegression
- **Features:** `[hgg_prob_resnet, hgg_prob_swin, mil_prob]`
- **Configuration:**
  - Random state: 42
  - Class weights: Balanced
  - Regularization: C=1.0, L2 penalty
  - Solver: lbfgs

### ROI MIL Ensemble
- **Model:** LogisticRegression (identical configuration)
- **Features:** `[hgg_prob_resnet, hgg_prob_swin, mil_prob]` (ROI MIL)
- **Configuration:**
  - Random state: 42 (same)
  - Class weights: Balanced (same)
  - Regularization: C=1.0, L2 penalty (same)
  - Solver: lbfgs (same)

### Model Coefficients Comparison

| Feature | Baseline Coef | ROI MIL Coef | Difference |
|---------|---------------|--------------|------------|
| hgg_prob_resnet | (not available) | 0.552091 | - |
| hgg_prob_swin | (not available) | 4.145874 | - |
| mil_prob | (not available) | 0.020900 | - |
| Intercept | (not available) | -2.063348 | - |

**Note:** Baseline coefficients not extracted, but models use identical configuration.

**ROI MIL Feature Importance:**
1. **hgg_prob_swin:** 4.145874 (dominant)
2. **hgg_prob_resnet:** 0.552091 (moderate)
3. **mil_prob:** 0.020900 (minimal contribution)

**Key Finding:** ROI MIL has **very low coefficient** (0.0209), suggesting minimal contribution to ensemble decisions. This may indicate ROI MIL probabilities are less informative than baseline MIL.

---

## Step 3: Evaluation Results

### Evaluation Protocol
- **Dataset:** OOF predictions (285 patients, 5-fold CV)
- **Threshold:** 0.22 (optimized for FN reduction)
- **Metrics:** AUC, Accuracy, Precision, Recall, F1, HGG Recall, FN Rate

### Performance Metrics

| Metric | Baseline Ensemble | ROI-MIL Ensemble | Δ | Winner |
|--------|-------------------|------------------|---|--------|
| **AUC-ROC** | 0.9074 | 0.9068 | -0.0006 | ✅ Baseline |
| **Accuracy** | 0.8456 | 0.8421 | -0.0035 | ✅ Baseline |
| **Precision** | 0.8990 | 0.8986 | -0.0005 | ✅ Baseline |
| **Recall** | 0.8905 | 0.8857 | -0.0048 | ✅ Baseline |
| **F1-Score** | 0.8947 | 0.8921 | -0.0027 | ✅ Baseline |
| **HGG Recall** | 0.8905 | 0.8857 | -0.0048 | ✅ Baseline |
| **FN Rate** | 0.1095 | 0.1143 | +0.0048 | ✅ Baseline |
| **FN Count** | 23 | 24 | +1 | ✅ Baseline |

### Confusion Matrices

**Baseline Ensemble:**
```
                Predicted
              LGG    HGG
Actual LGG     54     21
      HGG      23    187
```
- TN: 54, FP: 21, FN: 23, TP: 187

**ROI MIL Ensemble:**
```
                Predicted
              LGG    HGG
Actual LGG     54     21
      HGG      24    186
```
- TN: 54, FP: 21, FN: 24, TP: 186

**Key Difference:** ROI MIL ensemble has **1 additional false negative** (23 → 24).

---

## Step 4: Direct A/B Comparison

### Critical Metrics for Medical Decision

| Metric | Baseline | ROI-MIL | Δ | Assessment |
|--------|----------|---------|---|------------|
| **AUC-ROC** | 0.9074 | 0.9068 | -0.0006 | ⚠️ Slight degradation |
| **HGG Recall** | 0.8905 | 0.8857 | -0.0048 | ❌ Decreased |
| **FN Rate** | 0.1095 | 0.1143 | +0.0048 | ❌ Increased |
| **Precision** | 0.8990 | 0.8986 | -0.0005 | ⚠️ Slight degradation |
| **Accuracy** | 0.8456 | 0.8421 | -0.0035 | ⚠️ Slight degradation |

### Statistical Significance
- **Differences are small** (< 0.5% for most metrics)
- **FN count difference:** 1 additional false negative (23 → 24)
- **Clinical impact:** In medical imaging, even 1 additional FN can be significant

### Relative Changes
- **AUC:** -0.07% relative change
- **HGG Recall:** -0.54% relative change
- **FN Rate:** +4.38% relative change (worse)

---

## Step 5: Final Decision Logic

### Replacement Criteria

✅ **Replace IF:**
- AUC improves or stays stable
- HGG Recall improves
- FN rate decreases
- No degradation in calibration

❌ **Do NOT replace IF:**
- FN rate increases
- Recall drops
- AUC degrades

### Evaluation Results

| Criterion | Status | Result |
|-----------|--------|--------|
| AUC improves/stays stable | ❌ | AUC decreased (-0.0006) |
| HGG Recall improves | ❌ | Recall decreased (-0.0048) |
| FN rate decreases | ❌ | FN rate increased (+0.0048) |
| No degradation | ❌ | Multiple metrics degraded |

**Result:** **0/4 criteria met** → ❌ **DO NOT REPLACE**

---

## Detailed Analysis

### Why ROI MIL Performed Worse

1. **Low Model Coefficient**
   - ROI MIL coefficient: 0.0209 (very low)
   - Suggests ROI MIL probabilities are less informative
   - Ensemble relies primarily on Swin (4.15) and ResNet (0.55)

2. **Probability Calibration**
   - ROI MIL probabilities calibrated but may not align with ensemble needs
   - Baseline MIL probabilities better integrated with ensemble

3. **Ranking Quality**
   - ROI MIL has good standalone AUC (0.7897 vs baseline 0.7310)
   - But ensemble integration shows degradation
   - Suggests ROI MIL adds noise rather than signal

### Potential Issues

1. **Calibration Mismatch**
   - ROI MIL calibrated independently
   - May not align with ensemble's learned weights
   - Baseline MIL calibrated in ensemble context

2. **Feature Redundancy**
   - ROI MIL may be redundant with existing features
   - Low coefficient suggests minimal unique information

3. **Single-Fold Evaluation**
   - Original comparison was fold 0 only
   - Full OOF evaluation shows different results
   - Fold 0 may have been favorable to ROI MIL

---

## Recommendations

### Immediate Action
**❌ DO NOT REPLACE** baseline MIL with ROI MIL in production ensemble.

### Rationale
1. **FN rate increased** - Critical for medical applications
2. **Recall decreased** - More HGG cases missed
3. **AUC degraded** - Overall discrimination worse
4. **All criteria failed** - No improvement in any metric

### Future Work
1. **Investigate calibration mismatch** - ROI MIL may need ensemble-aware calibration
2. **Feature analysis** - Understand why ROI MIL coefficient is so low
3. **Threshold optimization** - Test if different threshold improves ROI MIL performance
4. **Multi-fold validation** - Verify results across individual folds

### Alternative Approaches
1. **Ensemble-aware calibration** - Calibrate ROI MIL considering ensemble context
2. **Feature engineering** - Extract different features from ROI MIL model
3. **Hybrid approach** - Combine baseline and ROI MIL (weighted average)

---

## Technical Notes

### Reproducibility
- **Random seed:** 42 (both models)
- **Threshold:** 0.22 (optimized for FN reduction)
- **Evaluation:** OOF predictions (no data leakage)
- **Model config:** Identical (LogisticRegression, C=1.0, balanced weights)

### Files Created
- `ensemble/oof_predictions/merged_oof_predictions_roi_mil.csv` - New OOF file
- `ensemble/models/roi_mil/meta_learner_logistic_regression_roi_mil.joblib` - ROI MIL model
- `ensemble/results/meta_learner_roi_mil/eval_threshold_0_22.json` - ROI MIL metrics
- `ensemble/results/meta_learner_roi_mil/meta_learner_metrics.json` - Main metrics file

### Baseline Files (Unchanged)
- `ensemble/oof_predictions/merged_oof_predictions.csv` - Original OOF
- `ensemble/models/meta_learner_logistic_regression.joblib` - Original model
- `ensemble/results/eval_threshold_0_22.json` - Original metrics

---

## Conclusion

The ROI MIL ensemble variant shows **slight but consistent degradation** across all critical metrics. While the differences are small, they represent a **regression** in medical imaging performance, particularly in false negative rate and recall.

**Final Recommendation: ❌ KEEP BASELINE ENSEMBLE**

The baseline ensemble remains superior and should be retained for production use. ROI MIL, while showing promise in standalone evaluation, does not improve ensemble performance when integrated.

---

**Report Status:** ✅ Complete  
**Decision:** ❌ Do Not Replace  
**Next Steps:** Investigate calibration mismatch, consider ensemble-aware calibration

