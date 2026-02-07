# Threshold Tuning Summary - Final Results

## 1. Baseline Confirmation

**Source**: `ensemble/results/test_evaluation_metrics.json`

**Confusion Matrix** (Threshold = 0.5):
```
                Predicted
                LGG  HGG
Actual LGG       69    6
        HGG      48  162
```

**Metrics**:
- **TN**: 69, **FP**: 6, **FN**: 48, **TP**: 162
- **Precision**: 0.9643 (96.43%)
- **Recall**: 0.7714 (77.14%) ⚠️ **LOW - TARGET FOR IMPROVEMENT**
- **F1-Score**: 0.8571 (85.71%)
- **Accuracy**: 0.8105 (81.05%)

**Data Type**: OOF predictions (validation data from 5-fold cross-validation)
- Total: 285 samples (LGG: 75, HGG: 210)
- **File**: `ensemble/oof_predictions/merged_oof_predictions.csv`

---

## 2. Threshold Sweep Results

**Method**: Computed ensemble probabilities from base model predictions using meta-learner coefficients, then swept thresholds from 0.05 to 0.95 (step 0.01).

**Meta-learner Coefficients** (from `ensemble/results/meta_learner_metrics.json`):
- ResNet50-3D: 0.537
- SwinUNETR-3D: 4.063 (most important)
- DualStreamMIL-3D: 0.890
- Intercept: -2.405

**Ensemble Probability Range**: 0.135 to 0.952 (mean: 0.621)

**Full Results**: `ensemble/results/threshold_tuning_results.json`

---

## 3. Best Thresholds by Policy

### Policy A: Target Recall + Precision Constraint (Precision ≥ 0.80)

**Policy Logic**: Find the smallest threshold that achieves the target recall AND maintains precision ≥ 0.80.

**Recall ≥ 0.85 AND Precision ≥ 0.80**:
- **Threshold**: **0.19**
- **FN**: 12 (↓75% from baseline 48)
- **FP**: 40 (↑567% from baseline 6)
- **TN**: 35 (↓49% from baseline 69)
- **TP**: 198 (↑22% from baseline 162)
- **Recall**: 0.9429 (↑22% from baseline 0.7714)
- **Precision**: 0.8319 (↓14% from baseline 0.9643)
- **F1**: 0.8839 (↑3% from baseline 0.8571)
- **Accuracy**: 0.8175 (↑1% from baseline 0.8105)

**Recall ≥ 0.90 AND Precision ≥ 0.80**:
- **Threshold**: **0.19** (same as above - minimum threshold achieving recall ≥ 0.90 AND precision ≥ 0.80)
- **FN**: 12 (↓75% from baseline 48)
- **FP**: 40 (↑567% from baseline 6)
- **TN**: 35 (↓49% from baseline 69)
- **TP**: 198 (↑22% from baseline 162)
- **Recall**: 0.9429 (↑22% from baseline 0.7714)
- **Precision**: 0.8319 (↓14% from baseline 0.9643)
- **F1**: 0.8839 (↑3% from baseline 0.8571)
- **Accuracy**: 0.8175 (↑1% from baseline 0.8105)

### Policy B: Maximum F1-Score

- **Threshold**: **0.22**
- **FN**: 21 (↓56% from baseline 48)
- **FP**: 21 (↑250% from baseline 6)
- **Recall**: 0.9000 (↑17% from baseline 0.7714)
- **Precision**: 0.9000 (↓7% from baseline 0.9643)
- **F1**: 0.9000 (↑5% from baseline 0.8571)
- **Accuracy**: 0.8526 (↑5% from baseline 0.8105)

### Policy C: Minimum FN with Precision ≥ 0.90

- **Threshold**: **0.22**
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
1. **Significant FN Reduction**: Reduces False Negatives from 48 to 21 (**56% reduction**)
2. **Balanced Metrics**: Achieves Recall=0.90 and Precision=0.90 (optimal balance)
3. **Improved F1**: Increases F1-score from 0.857 to 0.900 (**5% improvement**)
4. **Maintains High Precision**: Precision remains at 0.90 (acceptable for medical diagnosis)
5. **Better Overall Accuracy**: Improves accuracy from 0.8105 to 0.8526 (**5% improvement**)

**Trade-offs**:
- **FP Increase**: False Positives increase from 6 to 21 (250% increase, but still acceptable - only 21 LGG cases incorrectly flagged out of 75 total)
- **Precision Decrease**: Precision decreases from 0.964 to 0.900 (7% decrease, but still very high)

**Medical Context**: In brain tumor classification, **reducing False Negatives (missing HGG cases) is more critical** than reducing False Positives (incorrectly flagging LGG as HGG), as HGG requires immediate treatment. The threshold of 0.22 achieves this goal while maintaining acceptable precision.

**Note on Policy A Results**: Policy A (recall target + precision ≥ 0.80) selects threshold 0.19, which achieves high recall (0.94) while maintaining acceptable precision (0.83). This reduces FN from 48 to 12 (75% reduction) but increases FP from 6 to 40. This is a practical alternative to Policy B/C if higher recall is prioritized.

---

## 5. Comparison Table

| Metric | Baseline (0.5) | Recommended (0.22) | Change |
|--------|----------------|-------------------|--------|
| **Threshold** | 0.50 | 0.22 | -56% |
| **FN** | 48 | 21 | **↓56%** ✅ |
| **FP** | 6 | 21 | ↑250% |
| **TN** | 69 | 54 | ↓22% |
| **TP** | 162 | 189 | ↑17% |
| **Recall** | 0.7714 | 0.9000 | **↑17%** ✅ |
| **Precision** | 0.9643 | 0.9000 | ↓7% |
| **F1-Score** | 0.8571 | 0.9000 | **↑5%** ✅ |
| **Accuracy** | 0.8105 | 0.8526 | **↑5%** ✅ |

**Key Improvements**:
- **27 more HGG cases correctly identified** (189 vs 162)
- **27 fewer HGG cases missed** (21 vs 48)
- **15 more LGG cases incorrectly flagged** (21 vs 6) - acceptable trade-off

---

## 6. Slide-Ready Summaries

### English Summary

**Problem**: Our ensemble classifier has high False Negatives (48 HGG cases missed) and low Recall (77%) at the default threshold of 0.5, despite excellent Precision (96%).

**Solution**: By lowering the classification threshold from 0.5 to 0.22, we reduce False Negatives by 56% (from 48 to 21) and improve Recall to 90%, while maintaining Precision at 90%.

**Impact**: This change significantly improves our ability to detect HGG cases (critical for early treatment) while keeping false alarms at an acceptable level. We correctly identify 27 more HGG cases while only incorrectly flagging 15 additional LGG cases.

### Arabic Summary (ملخص بالعربية)

**المشكلة**: نموذجنا المجمع لديه عدد كبير من النتائج السلبية الخاطئة (48 حالة HGG لم يتم اكتشافها) ومعدل استدعاء منخفض (77%) عند العتبة الافتراضية 0.5، رغم الدقة العالية (96%).

**الحل**: بتخفيض عتبة التصنيف من 0.5 إلى 0.22، نقوم بتقليل النتائج السلبية الخاطئة بنسبة 56% (من 48 إلى 21) وتحسين معدل الاستدعاء إلى 90%، مع الحفاظ على الدقة عند 90%.

**التأثير**: هذا التغيير يحسن بشكل كبير قدرتنا على اكتشاف حالات HGG (الحرجة للعلاج المبكر) مع الحفاظ على الإنذارات الخاطئة عند مستوى مقبول. نكتشف بشكل صحيح 27 حالة HGG إضافية بينما نخطئ فقط في تصنيف 15 حالة LGG إضافية.

---

## 7. Implementation Instructions

### Files to Update

1. **Evaluation Scripts**: Update threshold from 0.5 to 0.22
   - Check: `scripts/ensemble/test_ensemble_on_new_patients.py`
   - Check: Any other evaluation/inference scripts

2. **Code Change Example**:
```python
# Before
y_pred = (ensemble_proba >= 0.5).astype(int)

# After
y_pred = (ensemble_proba >= 0.22).astype(int)
```

### Reproducing Results

**To regenerate threshold tuning analysis**:
```bash
cd /workspace/brain_tumor_project
python scripts/ensemble/threshold_tuning.py
```

**Output files**:
- `ensemble/results/threshold_tuning_results.json` (full results)
- `ensemble/results/threshold_tuning_analysis.md` (detailed analysis)
- `ensemble/results/THRESHOLD_TUNING_SUMMARY.md` (this file)

---

## 8. Validation Notes

**Important**: All threshold tuning was performed on **OOF (validation) data only**, not on test data. This ensures:
- ✅ No data leakage
- ✅ Generalizable results
- ✅ Fair comparison with baseline

**Next Steps**:
1. Apply threshold=0.22 to test set evaluation (if available)
2. Compare test set performance with validation performance
3. Monitor for any overfitting to validation set

---

## 9. Artifact Locations

**Source Data**:
- OOF Predictions: `ensemble/oof_predictions/merged_oof_predictions.csv`
- Baseline Metrics: `ensemble/results/test_evaluation_metrics.json`
- Meta-learner Metrics: `ensemble/results/meta_learner_metrics.json`

**Generated Results**:
- Threshold Tuning Results: `ensemble/results/threshold_tuning_results.json`
- Detailed Analysis: `ensemble/results/threshold_tuning_analysis.md`
- This Summary: `ensemble/results/THRESHOLD_TUNING_SUMMARY.md`

**Scripts**:
- Threshold Tuning: `scripts/ensemble/threshold_tuning.py`
- Meta-learner Training: `scripts/ensemble/train_meta_learner.py`

---

*Analysis Date: 2026-02-06*  
*Data: OOF predictions from 5-fold cross-validation (285 samples: 75 LGG, 210 HGG)*  
*Method: Threshold sweep (0.05-0.95, step 0.01) on ensemble probabilities*

