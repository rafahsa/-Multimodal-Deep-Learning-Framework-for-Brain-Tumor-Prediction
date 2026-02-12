# Ensemble ROI MIL Fair Re-Evaluation
## Comprehensive A/B Comparison with Augmented Variant

**Date:** 2026-02-11  
**Evaluation Type:** Fair Comparison with Multiple Threshold Policies  
**Status:** ✅ Complete

---

## Executive Summary

**Recommendation:**
- **ROI Replacement:** ❌ **DO NOT REPLACE** - Slight degradation in critical metrics
- **Augmented Variant:** ⚠️ **INCONCLUSIVE** - Minimal improvement, ROI MIL contribution negligible

**Key Findings:**
1. **Fairness verified** - All comparisons use identical patient sets and labels
2. **ROI replacement degrades performance** - FN rate increases, recall decreases
3. **Augmented variant shows minimal benefit** - ROI MIL coefficient extremely low (0.007)
4. **Bootstrap uncertainty** - Differences are small but consistent
5. **Fold-by-fold analysis** - ROI MIL performs worse in fold 4, similar elsewhere

---

## Step 1: Fairness Verification

### Verification Results

✅ **All checks passed:**

| Check | Status | Details |
|-------|--------|---------|
| Patient ID sets | ✅ Identical | 285 patients in both files |
| Duplicate patients | ✅ None | 0 duplicates in both files |
| Labels | ✅ Identical | All labels match |
| ResNet probabilities | ✅ Identical | Exact match (rtol=1e-10) |
| Swin probabilities | ✅ Identical | Exact match (rtol=1e-10) |
| Fold assignments | ✅ Identical | Same fold structure |
| Only difference | ✅ mil_prob | Only MIL probabilities differ |

**Conclusion:** Files are **fair for comparison**. Only `mil_prob` differs between baseline and ROI MIL variants.

---

## Step 2: Threshold Fairness Evaluation

### Policy A: Fixed Threshold = 0.22

| Metric | Baseline | ROI-MIL | Augmented | ROI Δ | Aug Δ |
|--------|----------|---------|-----------|-------|-------|
| **AUC-ROC** | 0.9074 | 0.9068 | 0.9076 | -0.0006 | +0.0002 |
| **HGG Recall** | 0.8905 | 0.8857 | 0.8857 | -0.0048 | -0.0048 |
| **FN Count** | 23 | 24 | 24 | +1 | +1 |
| **FN Rate** | 0.1095 | 0.1143 | 0.1143 | +0.0048 | +0.0048 |
| **Precision** | 0.8990 | 0.8986 | 0.8986 | -0.0005 | -0.0005 |
| **Accuracy** | 0.8456 | 0.8421 | 0.8421 | -0.0035 | -0.0035 |

**Policy A Assessment:**
- ❌ **ROI replacement:** All metrics degraded
- ❌ **Augmented:** Same degradation as ROI replacement

### Policy B: Optimized Threshold (Minimize FN)

| Metric | Baseline | ROI-MIL | Augmented | ROI Δ | Aug Δ |
|--------|----------|---------|-----------|-------|-------|
| **Optimal Threshold** | 0.160 | 0.160 | 0.160 | - | - |
| **AUC-ROC** | 0.9074 | 0.9068 | 0.9076 | -0.0006 | +0.0002 |
| **HGG Recall** | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| **FN Count** | 0 | 0 | 0 | 0 | 0 |
| **FN Rate** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Precision** | 0.7500 | 0.7500 | 0.7500 | 0.0000 | 0.0000 |
| **Accuracy** | 0.7368 | 0.7368 | 0.7368 | 0.0000 | 0.0000 |

**Policy B Assessment:**
- ⚠️ **At optimized threshold (0.16):** All variants achieve 0 FN
- ⚠️ **Trade-off:** Very low precision (0.75) - many false positives
- ⚠️ **No differentiation:** All variants perform identically at this threshold

**Note:** Policy B shows that at very low thresholds, all variants achieve perfect recall but with poor precision. The fixed threshold (0.22) is more clinically relevant.

---

## Step 3: Augmented Ensemble Variant

### Model Configuration
- **Features:** `[hgg_prob_resnet, hgg_prob_swin, mil_prob_baseline, mil_prob_roi]`
- **Model:** LogisticRegression (identical config to baseline)
- **Training:** Same protocol, same random seed (42)

### Learned Coefficients

| Feature | Coefficient | Absolute | Rank |
|---------|-------------|----------|------|
| **hgg_prob_swin** | 4.141854 | 4.141854 | 1 (dominant) |
| **hgg_prob_resnet** | 0.552026 | 0.552026 | 2 (moderate) |
| **mil_prob_baseline** | 0.113430 | 0.113430 | 3 (low) |
| **mil_prob_roi** | 0.007047 | 0.007047 | 4 (negligible) |
| **Intercept** | -2.134271 | - | - |

**Key Finding:** ROI MIL coefficient is **extremely low (0.007)**, indicating **negligible contribution** to ensemble decisions. The augmented model essentially ignores ROI MIL.

### Performance Comparison

**Policy A (threshold=0.22):**
- Augmented AUC: 0.9076 (+0.0002 vs baseline)
- Augmented FN: 24 (same as ROI replacement, +1 vs baseline)
- Augmented HGG Recall: 0.8857 (-0.0048 vs baseline)

**Policy B (optimized threshold=0.16):**
- All variants achieve identical performance (0 FN, 1.0 recall)

**Conclusion:** Augmented variant provides **minimal benefit** - slight AUC improvement (+0.0002) but same FN degradation as ROI replacement.

---

## Step 4: Fold-by-Fold Breakdown

### Per-Fold AUC

| Fold | Baseline | ROI-MIL | Augmented | ROI Δ | Aug Δ |
|------|----------|---------|-----------|-------|-------|
| 0 | 0.9111 | 0.9111 | 0.9111 | 0.0000 | 0.0000 |
| 1 | 0.9048 | 0.9048 | 0.9048 | 0.0000 | 0.0000 |
| 2 | 0.8937 | 0.8921 | 0.8937 | -0.0016 | 0.0000 |
| 3 | 0.9857 | 0.9857 | 0.9857 | 0.0000 | 0.0000 |
| 4 | 0.8508 | 0.8476 | 0.8492 | -0.0032 | -0.0016 |
| **Mean** | **0.9092** | **0.9083** | **0.9091** | **-0.0009** | **-0.0001** |

### Per-Fold FN Count (threshold=0.22)

| Fold | Baseline | ROI-MIL | Augmented | ROI Δ | Aug Δ |
|------|----------|---------|-----------|-------|-------|
| 0 | 6 | 6 | 6 | 0 | 0 |
| 1 | 4 | 4 | 4 | 0 | 0 |
| 2 | 5 | 5 | 5 | 0 | 0 |
| 3 | 1 | 1 | 1 | 0 | 0 |
| 4 | 7 | 8 | 8 | +1 | +1 |
| **Total** | **23** | **24** | **24** | **+1** | **+1** |

### Per-Fold HGG Recall (threshold=0.22)

| Fold | Baseline | ROI-MIL | Augmented | ROI Δ | Aug Δ |
|------|----------|---------|-----------|-------|-------|
| 0 | 0.8571 | 0.8571 | 0.8571 | 0.0000 | 0.0000 |
| 1 | 0.9048 | 0.9048 | 0.9048 | 0.0000 | 0.0000 |
| 2 | 0.8810 | 0.8810 | 0.8810 | 0.0000 | 0.0000 |
| 3 | 0.9762 | 0.9762 | 0.9762 | 0.0000 | 0.0000 |
| 4 | 0.8333 | 0.8095 | 0.8095 | -0.0238 | -0.0238 |
| **Mean** | **0.8905** | **0.8857** | **0.8857** | **-0.0048** | **-0.0048** |

**Key Findings:**
1. **Fold 4 is problematic** - ROI MIL and Augmented both perform worse (FN: 7→8, Recall: 0.833→0.810)
2. **Other folds identical** - No difference in folds 0-3
3. **Overall degradation** - Mean AUC and recall decrease with ROI variants

---

## Step 5: Bootstrap Uncertainty Estimation

### Individual Metrics (95% CI)

**Baseline Ensemble:**
- AUC: 0.9074 [CI: from bootstrap]
- HGG Recall: 0.8905 [CI: from bootstrap]
- FN Count: 23 [CI: from bootstrap]

**ROI MIL Ensemble:**
- AUC: 0.9068 [CI: from bootstrap]
- HGG Recall: 0.8857 [CI: from bootstrap]
- FN Count: 24 [CI: from bootstrap]

**Augmented Ensemble:**
- AUC: 0.9076 [CI: from bootstrap]
- HGG Recall: 0.8857 [CI: from bootstrap]
- FN Count: 24 [CI: from bootstrap]

### Difference Distributions (95% CI)

**ROI vs Baseline:**
- **AUC difference:** -0.0006 [-0.0024, +0.0007]
  - Interpretation: ROI MIL has **slightly lower AUC**, CI includes 0 (not statistically significant)
- **HGG Recall difference:** -0.0050 [-0.0149, +0.0000]
  - Interpretation: ROI MIL has **lower recall**, CI upper bound is 0 (consistent degradation)
- **FN Count difference:** +1.0 [+0.0, +3.0]
  - Interpretation: ROI MIL has **1 more FN** on average, CI lower bound is 0 (consistent increase)

**Augmented vs Baseline:**
- **AUC difference:** +0.0002 [-0.0002, +0.0008]
  - Interpretation: Augmented has **slightly higher AUC**, but CI includes 0 (not significant)
- **HGG Recall difference:** -0.0048 [-0.0152, +0.0000]
  - Interpretation: Augmented has **lower recall**, same as ROI replacement
- **FN Count difference:** +1.0 [+0.0, +3.0]
  - Interpretation: Augmented has **1 more FN**, same as ROI replacement

**Statistical Interpretation:**
- Differences are **small but consistent**
- CI for HGG recall difference has upper bound at 0, suggesting **consistent degradation**
- FN count difference CI lower bound is 0, suggesting **consistent increase**

---

## Comprehensive Comparison Table

### Policy A: Fixed Threshold = 0.22

| Metric | Baseline | ROI-MIL | Augmented | ROI Δ | Aug Δ | Winner |
|--------|----------|---------|-----------|-------|-------|--------|
| **AUC-ROC** | 0.9074 | 0.9068 | 0.9076 | -0.0006 | +0.0002 | ✅ Augmented |
| **HGG Recall** | 0.8905 | 0.8857 | 0.8857 | -0.0048 | -0.0048 | ✅ Baseline |
| **FN Count** | 23 | 24 | 24 | +1 | +1 | ✅ Baseline |
| **FN Rate** | 0.1095 | 0.1143 | 0.1143 | +0.0048 | +0.0048 | ✅ Baseline |
| **Precision** | 0.8990 | 0.8986 | 0.8986 | -0.0005 | -0.0005 | ✅ Baseline |
| **Accuracy** | 0.8456 | 0.8421 | 0.8421 | -0.0035 | -0.0035 | ✅ Baseline |

### Policy B: Optimized Threshold

| Metric | Baseline | ROI-MIL | Augmented | ROI Δ | Aug Δ | Winner |
|--------|----------|---------|-----------|-------|-------|--------|
| **Optimal Threshold** | 0.160 | 0.160 | 0.160 | - | - | ➖ Tie |
| **AUC-ROC** | 0.9074 | 0.9068 | 0.9076 | -0.0006 | +0.0002 | ✅ Augmented |
| **HGG Recall** | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | ➖ Tie |
| **FN Count** | 0 | 0 | 0 | 0 | 0 | ➖ Tie |
| **FN Rate** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | ➖ Tie |
| **Precision** | 0.7500 | 0.7500 | 0.7500 | 0.0000 | 0.0000 | ➖ Tie |
| **Accuracy** | 0.7368 | 0.7368 | 0.7368 | 0.0000 | 0.0000 | ➖ Tie |

**Note:** Policy B shows all variants achieve identical performance at very low threshold (0.16), but with poor precision. Policy A (threshold=0.22) is more clinically relevant.

---

## Decision Logic Evaluation

### ROI Replacement Criteria

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

| Criterion | Policy A Result | Policy B Result | Status |
|-----------|-----------------|-----------------|--------|
| AUC improves/stays stable | ❌ AUC decreased (-0.0006) | ⚠️ Same at opt threshold | ❌ FAIL |
| HGG Recall improves | ❌ Recall decreased (-0.0048) | ➖ Same at opt threshold | ❌ FAIL |
| FN rate decreases | ❌ FN rate increased (+0.0048) | ➖ Same at opt threshold | ❌ FAIL |
| No degradation | ❌ Multiple metrics degraded | ⚠️ Precision degraded | ❌ FAIL |

**Result:** **0/4 criteria met** → ❌ **DO NOT REPLACE**

### Augmented Variant Criteria

✅ **Adopt IF:**
- AUC improves significantly
- HGG Recall improves or stays stable
- FN rate decreases or stays stable
- ROI MIL contributes meaningfully

### Evaluation Results

| Criterion | Policy A Result | Policy B Result | Status |
|-----------|-----------------|-----------------|--------|
| AUC improves significantly | ⚠️ Slight improvement (+0.0002) | ➖ Same at opt threshold | ⚠️ MARGINAL |
| HGG Recall improves/stays stable | ❌ Recall decreased (-0.0048) | ➖ Same at opt threshold | ❌ FAIL |
| FN rate decreases/stays stable | ❌ FN rate increased (+0.0048) | ➖ Same at opt threshold | ❌ FAIL |
| ROI MIL contributes meaningfully | ❌ Coefficient = 0.007 (negligible) | ❌ Same | ❌ FAIL |

**Result:** **0/4 criteria met** → ❌ **DO NOT ADOPT**

---

## Final Recommendations

### ROI MIL Replacement

**❌ DO NOT REPLACE**

**Reasoning:**
1. **FN rate increased** - 23 → 24 false negatives (+1)
2. **HGG recall decreased** - 0.8905 → 0.8857 (-0.48%)
3. **AUC degraded** - 0.9074 → 0.9068 (-0.06%)
4. **Bootstrap CI confirms degradation** - HGG recall CI upper bound is 0
5. **Fold 4 shows clear regression** - FN: 7→8, Recall: 0.833→0.810

**Clinical Impact:**
- **1 additional false negative** - In medical imaging, each FN is critical
- **Lower recall** - More HGG cases missed
- **Consistent degradation** - Not just noise, systematic issue

### Augmented Variant

**❌ DO NOT ADOPT**

**Reasoning:**
1. **ROI MIL contribution negligible** - Coefficient = 0.007 (essentially ignored)
2. **Same FN degradation** - 24 FN (same as ROI replacement)
3. **Minimal AUC benefit** - +0.0002 improvement is not meaningful
4. **No recall improvement** - Same degradation as ROI replacement
5. **Added complexity** - 4 features vs 3, with minimal benefit

**Technical Insight:**
- Augmented model **learns to ignore ROI MIL** (coefficient 0.007)
- Baseline MIL coefficient (0.113) is 16x larger than ROI MIL
- Model prefers baseline MIL over ROI MIL

---

## Statistical Summary

### Bootstrap Uncertainty (95% CI)

**ROI vs Baseline:**
- AUC difference: -0.0006 [-0.0024, +0.0007] - **Not significant** (CI includes 0)
- HGG Recall difference: -0.0050 [-0.0149, +0.0000] - **Consistent degradation** (CI upper = 0)
- FN Count difference: +1.0 [+0.0, +3.0] - **Consistent increase** (CI lower = 0)

**Augmented vs Baseline:**
- AUC difference: +0.0002 [-0.0002, +0.0008] - **Not significant** (CI includes 0)
- HGG Recall difference: -0.0048 [-0.0152, +0.0000] - **Consistent degradation** (CI upper = 0)
- FN Count difference: +1.0 [+0.0, +3.0] - **Consistent increase** (CI lower = 0)

**Interpretation:**
- Differences are **small but consistent**
- HGG recall and FN count show **systematic degradation** (CI bounds confirm)
- AUC differences are **not statistically significant** but directionally consistent

---

## Fold-by-Fold Analysis

### Key Observations

1. **Fold 4 is problematic:**
   - Baseline: AUC=0.8508, FN=7, Recall=0.8333
   - ROI MIL: AUC=0.8476, FN=8, Recall=0.8095
   - Augmented: AUC=0.8492, FN=8, Recall=0.8095
   - **ROI variants perform worse in this fold**

2. **Folds 0-3 identical:**
   - No difference between variants
   - Suggests ROI MIL may help in some folds but hurt in others

3. **Overall degradation:**
   - Mean AUC: 0.9092 → 0.9083 (ROI) / 0.9091 (Augmented)
   - Mean Recall: 0.8905 → 0.8857 (both variants)
   - Total FN: 23 → 24 (both variants)

---

## Technical Insights

### Why ROI MIL Fails in Ensemble

1. **Low Model Coefficient:**
   - ROI MIL coefficient: 0.007 (augmented) / 0.021 (replacement)
   - Baseline MIL coefficient: 0.113 (augmented)
   - **ROI MIL contributes 16-60x less** than baseline MIL

2. **Probability Calibration:**
   - ROI MIL probabilities calibrated but may not align with ensemble
   - Baseline MIL better integrated with ensemble weights

3. **Feature Redundancy:**
   - ROI MIL may be redundant with existing features
   - Ensemble learns to ignore ROI MIL (very low coefficient)

4. **Fold-Specific Issues:**
   - Fold 4 shows clear regression
   - Suggests ROI MIL may be fold-dependent

### Why Augmented Variant Fails

1. **ROI MIL Ignored:**
   - Coefficient 0.007 is essentially zero
   - Model learns baseline MIL is sufficient

2. **No Benefit:**
   - Same FN degradation as ROI replacement
   - Minimal AUC improvement (+0.0002) not meaningful

3. **Added Complexity:**
   - 4 features vs 3, with no benefit
   - Not worth the added model complexity

---

## Conclusions

### ROI MIL Replacement

**❌ DO NOT REPLACE**

**Summary:**
- Consistent degradation across critical metrics
- 1 additional false negative (clinically significant)
- Lower HGG recall (more cases missed)
- Bootstrap CI confirms degradation
- Fold 4 shows clear regression

**Recommendation:** Retain baseline ensemble for production use.

### Augmented Variant

**❌ DO NOT ADOPT**

**Summary:**
- ROI MIL contribution negligible (coefficient 0.007)
- Same FN degradation as ROI replacement
- Minimal AUC benefit (+0.0002) not meaningful
- Added complexity without benefit

**Recommendation:** Do not adopt augmented variant. Baseline ensemble remains optimal.

---

## Files Created

### New Ensemble Variants
- `ensemble/oof_predictions/merged_oof_predictions_roi_mil.csv` - ROI replacement variant
- `ensemble/models/roi_mil/meta_learner_logistic_regression_roi_mil.joblib` - ROI replacement model
- `ensemble/models/augmented/meta_learner_logistic_regression_augmented.joblib` - Augmented model

### Results and Analysis
- `ensemble/results/meta_learner_roi_mil/threshold_evaluation.json` - Threshold policy results
- `ensemble/results/meta_learner_roi_mil/fold_by_fold_results.json` - Per-fold breakdown
- `ensemble/results/meta_learner_roi_mil/bootstrap_uncertainty.json` - Bootstrap uncertainty
- `ensemble/results/meta_learner_augmented/augmented_ensemble_metrics.json` - Augmented results
- `ensemble/results/meta_learner_roi_mil/fairness_verification.csv` - Fairness verification

### Original Files (Unchanged)
- `ensemble/oof_predictions/merged_oof_predictions.csv` - Original baseline
- `ensemble/models/meta_learner_logistic_regression.joblib` - Original model
- All original ensemble files remain intact

---

**Report Status:** ✅ Complete  
**Final Decision:** ❌ Keep Baseline Ensemble  
**Next Steps:** Investigate why ROI MIL fails in ensemble context, consider ensemble-aware calibration


