# Threshold Tuning Analysis for Ensemble Classifier

## Executive Summary

**Problem**: High False Negatives (FN=48) and low Recall (0.7714) for HGG classification at default threshold (0.5), despite high Precision (0.9643).

**Solution**: Threshold tuning reduces FN from 48 to 21 (balanced) or to 12 (high-sensitivity), with precision 0.90 and 0.8319 respectively.

**We report two operating points**:
1. **Balanced threshold (0.22)**: Reduces FN from 48 to 21 (↓56%), improves Recall to 0.90, maintains Precision at 0.90. Recommended for general use and reporting (Policy B/C: Max F1 / Min FN with Precision ≥ 0.90).
2. **High-sensitivity threshold (0.19)**: Reduces FN from 48 to 12 (↓75%), improves Recall to 0.9429, maintains Precision at 0.8319 (≥ 0.80). Recommended when prioritizing recall for HGG detection (Policy A: Recall target + Precision ≥ 0.80).

---

## 1. Baseline Results (Threshold = 0.5)

**Source**: `ensemble/results/test_evaluation_metrics.json`

**Confusion Matrix**:
- TN = 69 (LGG correctly classified)
- FP = 6 (LGG misclassified as HGG)
- FN = 48 (HGG misclassified as LGG) ⚠️ **HIGH**
- TP = 162 (HGG correctly classified)

**Metrics**:
- **Precision**: 0.9643 (96.43%)
- **Recall**: 0.7714 (77.14%) ⚠️ **LOW**
- **F1-Score**: 0.8571 (85.71%)
- **Accuracy**: 0.8105 (81.05%)

**Data Source**: OOF predictions (validation data from 5-fold cross-validation)
- Total samples: 285
- LGG: 75, HGG: 210

---

## 2. Threshold Sweep Results

**Method**: Swept thresholds from 0.05 to 0.95 (step 0.01), computed metrics for each.

**Key Findings**:
- Very low thresholds yield near-perfect recall but excessive FP, hence not suitable.
- Threshold 0.22: Optimal balance (Recall=0.90, Precision=0.90, F1=0.90)
- Threshold 0.23-0.24: Maintains Precision ≥0.90 with Recall around 0.87

---

## 3. Best Thresholds by Policy

### Policy A: Target Recall + Precision Constraint (Precision ≥ 0.80)

**Policy Logic**: Find the smallest threshold that achieves the target recall AND maintains precision ≥ 0.80.

**Recall ≥ 0.85 AND Precision ≥ 0.80**:
- **Threshold**: 0.19
- **FN**: 12 (↓75% from baseline 48)
- **FP**: 40 (↑567% from baseline 6)
- **TN**: 35 (↓49% from baseline 69)
- **TP**: 198 (↑22% from baseline 162)
- **Recall**: 0.9429 (↑22% from baseline 0.7714)
- **Precision**: 0.8319 (↓14% from baseline 0.9643)
- **F1**: 0.8839 (↑3% from baseline 0.8571)
- **Accuracy**: 0.8175 (↑1% from baseline 0.8105)

**Recall ≥ 0.90 AND Precision ≥ 0.80**:
- **Threshold**: 0.19 (same as above - minimum threshold achieving recall ≥ 0.90 AND precision ≥ 0.80)
- **FN**: 12 (↓75% from baseline 48)
- **FP**: 40 (↑567% from baseline 6)
- **TN**: 35 (↓49% from baseline 69)
- **TP**: 198 (↑22% from baseline 162)
- **Recall**: 0.9429 (↑22% from baseline 0.7714)
- **Precision**: 0.8319 (↓14% from baseline 0.9643)
- **F1**: 0.8839 (↑3% from baseline 0.8571)
- **Accuracy**: 0.8175 (↑1% from baseline 0.8105)

### Policy B: Maximum F1-Score

- **Threshold**: 0.22
- **FN**: 21 (↓56% from baseline 48)
- **FP**: 21 (↑250% from baseline 6)
- **TN**: 54 (↓22% from baseline 69)
- **TP**: 189 (↑17% from baseline 162)
- **Recall**: 0.9000 (↑17% from baseline 0.7714)
- **Precision**: 0.9000 (↓7% from baseline 0.9643)
- **F1**: 0.9000 (↑5% from baseline 0.8571)
- **Accuracy**: 0.8526 (↑5% from baseline 0.8105)

### Policy C: Minimum FN with Precision ≥ 0.90

- **Threshold**: 0.22
- **FN**: 21 (↓56% from baseline 48)
- **FP**: 21 (↑250% from baseline 6)
- **TN**: 54 (↓22% from baseline 69)
- **TP**: 189 (↑17% from baseline 162)
- **Recall**: 0.9000 (↑17% from baseline 0.7714)
- **Precision**: 0.9000 (↓7% from baseline 0.9643)
- **F1**: 0.9000 (↑5% from baseline 0.8571)
- **Accuracy**: 0.8526 (↑5% from baseline 0.8105)

**Note**: The script includes a fallback to Precision ≥ 0.85 if Precision ≥ 0.90 is not achievable, but in our results, Precision ≥ 0.90 was achievable, so the fallback was not used.

---

## 4. Recommendation

### **Recommended Threshold: 0.22** (Policy B/C)

**Rationale**:
1. **Significant FN Reduction**: Reduces False Negatives from 48 to 21 (56% reduction)
2. **Balanced Metrics**: Achieves Recall=0.90 and Precision=0.90 (balanced trade-off)
3. **Improved F1**: Increases F1-score from 0.857 to 0.900 (5% improvement)
4. **Maintains High Precision**: Precision remains at 0.90 (acceptable for medical diagnosis)
5. **Better Overall Accuracy**: Improves accuracy from 0.8105 to 0.8526 (5% improvement)

**Trade-offs**:
- **FP Increase**: False Positives increase from 6 to 21 (250% increase, but still acceptable)
- **Precision Decrease**: Precision decreases from 0.964 to 0.900 (7% decrease, but still high)

**Medical Context**: In brain tumor classification, **reducing False Negatives (missing HGG cases) is more critical** than reducing False Positives (incorrectly flagging LGG as HGG), as HGG requires immediate treatment. The threshold of 0.22 achieves this goal while maintaining acceptable precision.

---

## 5. Comparison Table

| Metric | Baseline (0.5) | Recommended (0.22) | Change |
|--------|----------------|-------------------|--------|
| **Threshold** | 0.50 | 0.22 | -56% |
| **FN** | 48 | 21 | **↓56%** ✅ |
| **FP** | 6 | 21 | ↑250% |
| **Recall** | 0.7714 | 0.9000 | **↑17%** ✅ |
| **Precision** | 0.9643 | 0.9000 | ↓7% |
| **F1-Score** | 0.8571 | 0.9000 | **↑5%** ✅ |
| **Accuracy** | 0.8105 | 0.8526 | **↑5%** ✅ |

---

## 6. Implementation

**To apply the new threshold**:

1. **For evaluation scripts**: Modify the threshold parameter from 0.5 to 0.22
2. **For production**: Update ensemble prediction code to use threshold=0.22
3. **For high-sensitivity mode** (recall-prioritized), use threshold=0.19.

**Example code change**:
```python
# Before
y_pred = (ensemble_proba >= 0.5).astype(int)

# After
y_pred = (ensemble_proba >= 0.22).astype(int)
```

**Files to update**:
- `scripts/ensemble/test_ensemble_on_new_patients.py` (if threshold is hardcoded)
- Any inference/evaluation scripts that use the ensemble

---

## 7. Slide-Ready Summaries

### English Summary

**Problem**: Our ensemble classifier has high False Negatives (48 HGG cases missed) and low Recall (77%) at the default threshold of 0.5, despite excellent Precision (96%).

**Solution**: By lowering the classification threshold from 0.5 to 0.22, we reduce False Negatives by 56% (from 48 to 21) and improve Recall to 90%, while maintaining Precision at 90%.

**Impact**: This change significantly improves our ability to detect HGG cases (critical for early treatment) while keeping false alarms at an acceptable level.

For a high-sensitivity operating point, we use threshold=0.19 (Precision ≥0.80), reducing FN to 12 and increasing recall to 0.9429.

### Arabic Summary (ملخص بالعربية)

**المشكلة**: نموذجنا المجمع لديه عدد كبير من النتائج السلبية الخاطئة (48 حالة HGG لم يتم اكتشافها) ومعدل استدعاء منخفض (77%) عند العتبة الافتراضية 0.5، رغم الدقة العالية (96%).

**الحل**: بتخفيض عتبة التصنيف من 0.5 إلى 0.22، نقوم بتقليل النتائج السلبية الخاطئة بنسبة 56% (من 48 إلى 21) وتحسين معدل الاستدعاء إلى 90%، مع الحفاظ على الدقة عند 90%.

**التأثير**: هذا التغيير يحسن بشكل كبير قدرتنا على اكتشاف حالات HGG (الحرجة للعلاج المبكر) مع الحفاظ على الإنذارات الخاطئة عند مستوى مقبول.
 
 ولنقطة تشغيل عالية الحساسية، نستخدم العتبة 0.19 (Precision ≥0.80) مما يخفض FN إلى 12 ويرفع الاستدعاء إلى 0.9429.
---

## 8. Files and Artifacts

**Source Files**:
- OOF Predictions: `ensemble/oof_predictions/merged_oof_predictions.csv`
- Baseline Metrics: `ensemble/results/test_evaluation_metrics.json`
- Meta-learner Coefficients: `ensemble/results/meta_learner_metrics.json`

**Generated Files**:
- Threshold Tuning Results: `ensemble/results/threshold_tuning_results.json`
- This Analysis: `ensemble/results/threshold_tuning_analysis.md`

**Scripts**:
- Threshold Tuning Script: `scripts/ensemble/threshold_tuning.py`
- Meta-learner Training: `scripts/ensemble/train_meta_learner.py`

---

## 9. Validation

**Important**: All threshold tuning was performed on **OOF (validation) data only**, not on test data. This ensures:
- No data leakage
- Generalizable results
- Fair comparison with baseline

**Next Steps**:
1. Apply threshold=0.22 to test set evaluation
2. Compare test set performance with validation performance
3. Monitor for any overfitting to validation set

---

*Analysis completed: 2026-02-06*
*Data source: OOF predictions from 5-fold cross-validation (285 samples)*

