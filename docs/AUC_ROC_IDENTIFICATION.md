# AUC-ROC Identification for MICCAI Abstract

**Date:** 2026-02-12  
**Purpose:** Identify the correct AUC-ROC value for the final nested CV ensemble described in the abstract

---

## Final Ensemble Configuration (from Abstract)

The abstract describes:
- **Nested cross-validation** framework
- **Platt scaling calibration** for base model probabilities
- **Logistic regression meta-learner** with **meta-features** (probability statistics, entropy, margins)
- **5-fold patient-level cross-validation**
- **Strict separation** of calibration and meta-learner training

---

## AUC-ROC Value Identification

### ✅ CORRECT AUC-ROC FOR FINAL ENSEMBLE

**File:** `ensemble/results/nested_cv_meta_features/auc_roc_computed.json`

**JSON Key:** `mean_fold_auc_roc`

**Value:** **0.9000 ± 0.0477**

**Type:** Mean fold AUC-ROC (average across 5 outer folds)

**Per-Fold AUC-ROC:**
- Fold 0: 0.8746
- Fold 1: 0.9683
- Fold 2: 0.8683
- Fold 3: 0.9444
- Fold 4: 0.8444

**Overall AUC-ROC (pooled):** 0.8843

---

## Explanation

This AUC-ROC value corresponds to the **final nested CV ensemble with meta-features** described in the abstract:

1. ✅ **Nested CV structure:** Evaluated using 5-fold outer CV
2. ✅ **Platt scaling calibration:** Applied to base model probabilities before meta-learner training
3. ✅ **Meta-features:** 14 features including base probabilities + engineered features (statistics, entropy, margins)
4. ✅ **Logistic regression meta-learner:** Trained on calibrated OOF predictions
5. ✅ **No data leakage:** Outer-test sets never seen during training or calibration

**Computation Method:**
- AUC computed per outer fold from calibrated probabilities
- Mean and standard deviation computed across folds
- Overall AUC computed by pooling all predictions

---

## Comparison with Other AUC Values

| Source | AUC-ROC | Type | Configuration | Match Abstract? |
|--------|---------|------|---------------|-----------------|
| **Nested CV Meta-Features** | **0.9000 ± 0.0477** | Mean fold AUC | Nested CV + Platt + Meta-features | ✅ **YES** |
| Baseline Ensemble | 0.9074 | Overall AUC | No nested CV, no meta-features | ❌ NO |
| Baseline Ensemble (meta_learner_metrics.json) | 0.9126 | Overall AUC | No nested CV, no meta-features | ❌ NO |

**Recommendation:** Use **0.9000 ± 0.0477** (mean fold AUC-ROC) for the abstract.

---

## Verification

**Script Used:** `scripts/ensemble/compute_nested_cv_auc.py`

**Computation Details:**
- Re-ran nested CV evaluation matching `nested_cv_meta_features.py` protocol
- Applied same meta-feature engineering (14 features)
- Applied same Platt scaling calibration
- Computed AUC per outer fold
- Computed mean and standard deviation across folds

**Confirmation:**
- ✅ Matches the final ensemble configuration in abstract
- ✅ Uses nested CV protocol (no data leakage)
- ✅ Includes meta-features (probability statistics, entropy, margins)
- ✅ Uses Platt scaling calibration
- ✅ Computed from outer-test sets only

---

## Abstract Reporting Recommendation

**For Results Section:**

Report: **"The nested cross-validation ensemble achieved mean AUC-ROC of 0.9000 ± 0.0477 across five folds, with per-fold mean false negatives 2.8 ± 2.1..."**

**Rationale:**
- This is the AUC for the final ensemble configuration (nested CV + meta-features + calibration)
- Mean fold AUC is the standard reporting metric for cross-validation
- Standard deviation (0.0477) provides uncertainty estimate
- Matches the protocol described in the Method section

---

## Files Reference

**Computed AUC File:**
- `ensemble/results/nested_cv_meta_features/auc_roc_computed.json`

**Original Results File (no AUC):**
- `ensemble/results/nested_cv_meta_features/meta_features_results_20260209_005859.json`

**Computation Script:**
- `scripts/ensemble/compute_nested_cv_auc.py`

---

*Generated: 2026-02-12*

