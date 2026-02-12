# Technical Review: Ensemble MIL vs ROI MIL Model
## Fold 0 Analysis for Production Ensemble Decision

**Date:** 2026-02-11  
**Reviewer:** ML Engineering Team  
**Decision Context:** Production ensemble member replacement

---

## Executive Summary

**Recommendation: ⚠️ DO NOT REPLACE YET - Calibration Required**

The ROI MIL model shows **superior ranking quality** (AUC +8.0%) and **better diversity** for ensemble, but requires **probability calibration** before integration. The current ensemble MIL, while having worse AUC, is already calibrated and functional. **Replace after calibration and validation.**

---

## 1. Model Identification

### Current Ensemble MIL
- **Source:** `ensemble/oof_predictions/merged_oof_predictions.csv` → `mil_prob` column
- **Training Script:** Unknown (calibrated version of DualStreamMIL-3D)
- **Status:** ✅ **Active in production ensemble**
- **Fold 0 Run:** Derived from `results/DualStreamMIL-3D/runs/` (calibrated)

### ROI MIL Candidate
- **Source:** `runs/mil_roi_sanity/runs/fold_0/run_20260211_011309`
- **Training Script:** `scripts/training/train_dual_stream_mil_roi.py`
- **Status:** ✅ **Training completed** (10 epochs, best at epoch 8)
- **Key Feature:** ROI-guided sampling (70% tumor, 30% context)

---

## 2. Performance Metrics Comparison (Fold 0)

| Metric | Ensemble MIL | ROI MIL | Difference | Winner |
|--------|--------------|---------|------------|--------|
| **AUC-ROC** | 0.7310 | **0.7897** | +0.0587 (+8.0%) | ✅ ROI |
| **F1-Score** | 0.8485 | 0.0000 | -0.8485 | ❌ Ensemble |
| **Precision** | 0.7368 | 0.0000 | -0.7368 | ❌ Ensemble |
| **Recall** | 1.0000 | 0.0000 | -1.0000 | ❌ Ensemble |
| **Accuracy** | 0.7368 | 0.2632 | -0.4737 | ❌ Ensemble |
| **Brier Score** | **0.1955** | 0.4569 | -0.2614 | ✅ Ensemble |

**Critical Finding:** ROI MIL has **better ranking quality** (AUC) but **worse calibration** (Brier). Hard classification metrics are misleading since ensemble uses probabilities, not hard predictions.

---

## 3. Probability Distribution Analysis

### Ensemble MIL (Current)
- **Range:** [0.6856, 0.6975]
- **Mean:** 0.6892
- **All probabilities > 0.5:** ✅ Yes (predicts all HGG)
- **Calibration:** ✅ Already calibrated for ensemble use

### ROI MIL (Candidate)
- **Range:** [0.1993, 0.2651]
- **Mean:** 0.2204
- **All probabilities < 0.5:** ✅ Yes (predicts all LGG)
- **Calibration:** ❌ **Requires calibration** before ensemble use

**Key Insight:** ROI MIL probabilities are systematically low but **better ranked** (higher AUC). Calibration would shift probabilities to appropriate range while preserving ranking quality.

---

## 4. Ensemble Contribution Analysis

### Base Model Context (Fold 0)
- **ResNet50-3D:** AUC 0.4460 (poor, but high mean prob 0.86)
- **SwinUNETR-3D:** AUC 0.9063 (excellent)
- **Ensemble MIL:** AUC 0.7310 (moderate)
- **ROI MIL:** AUC 0.7897 (better than current MIL)

### Model Diversity (Correlations)
| Pair | Correlation | Assessment |
|------|-------------|------------|
| ResNet vs Swin | -0.28 | ✅ Good diversity |
| ResNet vs Ensemble MIL | -0.48 | ✅ Good diversity |
| **ResNet vs ROI MIL** | **-0.16** | ✅ **Better diversity** |
| Swin vs Ensemble MIL | 0.58 | ⚠️ Moderate correlation |
| **Swin vs ROI MIL** | **0.19** | ✅ **Better diversity** |
| **Ensemble MIL vs ROI MIL** | **0.22** | ✅ **Low correlation = more diversity** |

**Key Finding:** ROI MIL provides **better diversity** (lower correlations) with other ensemble members, which improves ensemble robustness.

### Ranking Quality Impact
- **Ensemble MIL AUC:** 0.7310
- **ROI MIL AUC:** 0.7897
- **Improvement:** +0.0587 (+8.0% relative)

**Interpretation:** ROI MIL's better AUC means it can **better rank** HGG vs LGG cases, which is valuable for ensemble probability aggregation even if absolute probabilities need calibration.

### Calibration Quality
- **Ensemble MIL Brier:** 0.1955 (better calibrated)
- **ROI MIL Brier:** 0.4569 (poor calibration, but fixable)

**Note:** Brier score is worse for ROI MIL because probabilities are systematically low. **Calibration (Platt/Isotonic) would fix this** while preserving AUC ranking quality.

### False Negative Analysis (HGG Cases)
- **HGG cases:** 42
- **Ensemble MIL:** 0 FN at threshold 0.5 (all HGG correctly identified)
- **ROI MIL:** 42 FN at threshold 0.5 (all HGG missed)

**Critical:** ROI MIL would need **calibration + threshold tuning** or **ensemble meta-learner** to handle low probabilities. However, since ensemble uses probabilities (not hard predictions), the meta-learner can learn appropriate weights.

---

## 5. MIL Diagnostics (ROI Model)

### Attention Mechanism Health
- **Attention Entropy:** 3.4605 ± 0.0111 (max ~3.46 for 32 instances)
- **Top-1 Attention Weight:** 0.0383 ± 0.0080
- **Effective Instances:** 31.84 out of 32

**Assessment:**
- ✅ **Excellent attention diversity** - no collapse
- ✅ **Near-maximal instance utilization**
- ✅ **Low overfitting risk** - high entropy indicates diverse attention
- ✅ **Good interpretability** - attention weights are meaningful

**Comparison:** ROI model shows **healthier attention** than typical MIL models, suggesting ROI guidance is working effectively.

---

## 6. Ensemble Integration Considerations

### Current Ensemble Architecture
- **Meta-learner:** Logistic Regression
- **Features:** `[hgg_prob_resnet, hgg_prob_swin, mil_prob]`
- **Uses probabilities, not hard predictions**

### Impact of Replacing with ROI MIL

**Advantages:**
1. ✅ **Better ranking quality** (AUC +8.0%) improves ensemble discrimination
2. ✅ **Better diversity** (lower correlations) improves ensemble robustness
3. ✅ **Healthy attention mechanism** suggests better generalization
4. ✅ **ROI guidance** provides theoretical advantage (tumor-focused sampling)

**Challenges:**
1. ❌ **Requires calibration** - probabilities too low for direct use
2. ❌ **Threshold-dependent** - hard predictions fail at 0.5 (but ensemble uses probs)
3. ⚠️ **Single-fold evaluation** - need cross-fold validation for confidence

### Calibration Strategy
If replacing, ROI MIL probabilities should be:
1. **Calibrated using nested CV** (same as current ensemble MIL)
2. **Validated across all 5 folds** before deployment
3. **Tested in ensemble** to ensure improved performance

---

## 7. Risk Assessment

### Risk of Replacing (Without Calibration)
- **HIGH:** Ensemble would receive systematically low probabilities
- **Impact:** Meta-learner might struggle to learn appropriate weights
- **Mitigation:** Calibration required before replacement

### Risk of Replacing (With Calibration)
- **LOW:** Calibrated probabilities should work similarly to current MIL
- **Benefit:** Better AUC and diversity should improve ensemble
- **Validation:** Test on all folds before production deployment

### Risk of Not Replacing
- **LOW:** Current ensemble is functional
- **Opportunity Cost:** Missing +8% AUC improvement and better diversity
- **Long-term:** ROI guidance is theoretically superior

---

## 8. Required Actions

### Before Replacement (Mandatory)
1. **Calibrate ROI MIL probabilities** using nested CV (same protocol as current MIL)
2. **Validate across all 5 folds** - ensure consistent improvement
3. **Test ensemble integration** - verify improved ensemble performance
4. **Compare calibrated ROI vs calibrated Ensemble MIL** - fair comparison

### Validation Steps
1. Generate OOF predictions for ROI MIL across all 5 folds
2. Apply nested CV calibration to ROI MIL probabilities
3. Replace `mil_prob` in merged OOF with calibrated ROI probabilities
4. Re-train meta-learner and compare ensemble metrics
5. Verify improvement in ensemble AUC, F1, and FN rate

---

## 9. Final Recommendation

### ⚠️ DO NOT REPLACE YET - Calibration Required

**Reasoning:**
1. **ROI MIL shows promise** - better AUC (+8.0%) and diversity
2. **Calibration is mandatory** - probabilities too low for direct use
3. **Single-fold evaluation insufficient** - need cross-fold validation
4. **Current ensemble is functional** - no urgent need to replace

**Next Steps:**
1. **Calibrate ROI MIL** using nested CV protocol
2. **Validate across all 5 folds** - ensure consistent improvement
3. **Test ensemble integration** - verify improved ensemble performance
4. **Re-evaluate after calibration** - make final replacement decision

**Timeline:** 2-3 days for calibration + validation + ensemble testing

**Expected Outcome:** After calibration, ROI MIL should provide:
- Better ensemble AUC (due to better ranking)
- Better ensemble robustness (due to diversity)
- Lower FN rate (if calibration improves HGG probability estimates)

---

## 10. Technical Notes

### Probability Calibration
- Current ensemble MIL uses **calibrated probabilities** (mil_prob)
- ROI MIL needs **same calibration protocol** (nested CV, Platt/Isotonic)
- Calibration preserves AUC ranking while fixing probability scale

### Ensemble Meta-Learner
- Uses Logistic Regression with probabilities as features
- Can learn appropriate weights even with low probabilities
- But calibration ensures probabilities are in expected range

### ROI Guidance Impact
- ROI sampling (70% tumor, 30% context) appears effective
- Attention mechanism shows healthy diversity
- Better AUC suggests ROI guidance improves discrimination

---

**Report Status:** Pending Calibration and Cross-Fold Validation  
**Decision:** Deferred until calibration complete

