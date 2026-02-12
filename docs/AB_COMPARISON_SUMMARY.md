# A/B Comparison Summary: Baseline vs ROI-MIL Ensemble

## 1. Ensemble Variant Definitions

### A) Baseline Ensemble
- **Input:** `ensemble/oof_predictions/merged_oof_predictions.csv`
- **Script:** `scripts/ensemble/train_meta_learner.py`
- **Model:** `ensemble/models/meta_learner_logistic_regression.joblib`
- **Features:** `hgg_prob_resnet`, `hgg_prob_swin`, `mil_prob` (baseline MIL)

### B) ROI-MIL Ensemble  
- **Input:** `ensemble/oof_predictions/merged_oof_predictions_roi_mil.csv`
- **Script:** `scripts/ensemble/train_meta_learner_roi_mil.py`
- **Model:** `ensemble/models/roi_mil/meta_learner_logistic_regression_roi_mil.joblib`
- **Features:** `hgg_prob_resnet`, `hgg_prob_swin`, `mil_prob` (ROI-MIL)

**Difference:** ONLY `mil_prob` column differs; all else identical.

---

## 2. Fairness Verification ✅

| Check | Status | Max Diff |
|-------|--------|----------|
| Same patient IDs (285) | ✅ PASS | - |
| Same labels | ✅ PASS | - |
| Same fold assignments | ✅ PASS | - |
| Same ResNet probs | ✅ PASS | 0.00e+00 |
| Same Swin probs | ✅ PASS | 0.00e+00 |
| MIL probs differ | ✅ PASS | 0.055266 |

**Conclusion:** Fair comparison - only `mil_prob` differs.

---

## 3. What is "Nested"?

**"Nested CV" is an evaluation protocol, NOT a separate ensemble.**

- **Protocol:** Inner folds train calibration; outer folds apply calibration
- **Used for:** Probability calibration (Platt scaling) of base models
- **Both A and B use:** Same nested CV protocol for calibration
- **Meta-learner training:** Trains on all 285 OOF predictions (not nested structure)

**"Nested CV ensemble" in abstract:** Refers to a separate experiment with meta-features, NOT part of A/B comparison.

---

## 4. A/B Comparison Results (Threshold = 0.22)

### Overall Metrics

| Metric | Baseline | ROI-MIL | Δ |
|--------|----------|---------|---|
| AUC-ROC | 0.9074 | 0.9068 | -0.0006 |
| HGG Recall | 0.8905 | 0.8857 | -0.0048 |
| FN Count | 23 | 24 | +1 |
| FN Rate | 0.1095 | 0.1143 | +0.0048 |

### Per-Fold Results

| Fold | Baseline FN | ROI-MIL FN | Δ | Baseline Recall | ROI-MIL Recall | Δ |
|------|-------------|------------|---|-----------------|----------------|---|
| 0 | 6 | 6 | 0 | 0.8571 | 0.8571 | 0.0000 |
| 1 | 4 | 4 | 0 | 0.9048 | 0.9048 | 0.0000 |
| 2 | 5 | 5 | 0 | 0.8810 | 0.8810 | 0.0000 |
| 3 | 1 | 1 | 0 | 0.9762 | 0.9762 | 0.0000 |
| **4** | **7** | **8** | **+1** | **0.8333** | **0.8095** | **-0.0238** |

**Key Finding:** All degradation comes from Fold 4.

---

## 5. Final Conclusion

### ❌ DO NOT REPLACE: Keep Baseline Ensemble

**Strongest Evidence:** Fold 4 regression
- FN: 7 → 8 (+1)
- Recall: 0.8333 → 0.8095 (-0.0238, 2.4% relative decrease)

**Supporting Evidence:**
- Overall FN: 23 → 24 (+1)
- Overall recall: 0.8905 → 0.8857 (-0.0048)
- No improvement in other folds to offset Fold 4 regression

**Recommendation:** Retain baseline ensemble for production.

---

## Commands Used

```bash
# Run A/B comparison
python scripts/ensemble/verify_and_compare_ab.py
```

**Results saved to:** `ensemble/results/ab_comparison/ab_comparison_results.json`

---

*Generated: 2026-02-11*

