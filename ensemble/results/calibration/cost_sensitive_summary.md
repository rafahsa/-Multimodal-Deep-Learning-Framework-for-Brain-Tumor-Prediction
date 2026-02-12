# Cost-Sensitive Threshold Selection

## Overview

This analysis implements cost-sensitive threshold selection on calibrated probabilities (Platt scaling) to optimize the trade-off between False Negatives (FN) and False Positives (FP) in a medical setting.

**Evaluation Set**: Held-out threshold selection set (30% of OOF, seed=42, n=86)  
**Calibration Run**: `2026-02-07_22-29-29_platt_seed42`  
**Cost Function**: `Cost = (w_FN × FN) + (w_FP × FP)`

---

## Cost Configurations

| Configuration | w_FN | w_FP | Description |
|---------------|------|------|-------------|
| **Default** | 2 | 1 | Standard medical setting: FN cost = 2× FP cost |
| **Conservative** | 3 | 1 | Conservative medical setting: FN cost = 3× FP cost |

---

## Optimal Thresholds

### Default Configuration (w_FN = 2, w_FP = 1)

**Optimal Threshold**: 0.36

| Metric | Value |
|--------|-------|
| Cost | 11.0 |
| FN | 1 |
| FP | 9 |
| TN | 14 |
| TP | 62 |
| Precision | 0.8732 |
| Recall | 0.9841 |
| F1 | 0.9254 |
| Accuracy | 0.8837 |
| Specificity | 0.6087 |

### Conservative Configuration (w_FN = 3, w_FP = 1)

**Optimal Threshold**: 0.36

| Metric | Value |
|--------|-------|
| Cost | 12.0 |
| FN | 1 |
| FP | 9 |
| TN | 14 |
| TP | 62 |
| Precision | 0.8732 |
| Recall | 0.9841 |
| F1 | 0.9254 |
| Accuracy | 0.8837 |
| Specificity | 0.6087 |

**Note**: Both configurations select the same threshold (0.36), indicating robustness to the cost weight ratio.

---

## Comparison: Cost-Sensitive vs Current Balanced Threshold

### Current Balanced Threshold (0.41)

| Metric | Value |
|--------|-------|
| Cost (default) | 12.0 |
| Cost (conservative) | 15.0 |
| FN | 4 |
| FP | 4 |
| TN | 19 |
| TP | 59 |
| Precision | 0.9365 |
| Recall | 0.9365 |
| F1 | 0.9365 |
| Accuracy | 0.9070 |
| Specificity | 0.8261 |

### Cost-Sensitive Optimal Threshold (0.36)

| Metric | Value |
|--------|-------|
| Cost (default) | 11.0 |
| Cost (conservative) | 12.0 |
| FN | 1 |
| FP | 9 |
| TN | 14 |
| TP | 62 |
| Precision | 0.8732 |
| Recall | 0.9841 |
| F1 | 0.9254 |
| Accuracy | 0.8837 |
| Specificity | 0.6087 |

---

## Side-by-Side Comparison

| Metric | Current (0.41) | Cost-Sensitive (0.36) | Change |
|--------|----------------|----------------------|--------|
| **Threshold** | 0.41 | 0.36 | -0.05 |
| **Cost (default)** | 12.0 | 11.0 | **-1.0 (8.3% reduction)** |
| **Cost (conservative)** | 15.0 | 12.0 | **-3.0 (20.0% reduction)** |
| **FN** | 4 | 1 | **-3 (75% reduction)** ✅ |
| **FP** | 4 | 9 | +5 (125% increase) ⚠️ |
| **TN** | 19 | 14 | -5 |
| **TP** | 59 | 62 | +3 |
| **Precision** | 0.9365 | 0.8732 | -0.0633 (-6.8%) |
| **Recall** | 0.9365 | 0.9841 | +0.0476 (+5.1%) ✅ |
| **F1** | 0.9365 | 0.9254 | -0.0111 (-1.2%) |
| **Accuracy** | 0.9070 | 0.8837 | -0.0233 (-2.6%) |
| **Specificity** | 0.8261 | 0.6087 | -0.2174 (-26.3%) |

---

## Key Findings

### ✅ Advantages of Cost-Sensitive Threshold (0.36)

1. **Significantly Lower FN**: 1 vs 4 (75% reduction)
   - **Medical Impact**: 3 fewer missed HGG cases
   - **Critical for medical application**: Reducing false negatives is paramount

2. **Lower Total Cost**: 
   - Default: 11.0 vs 12.0 (8.3% reduction)
   - Conservative: 12.0 vs 16.0 (25.0% reduction)

3. **Higher Recall**: 0.9841 vs 0.9365 (+5.1%)
   - Better sensitivity for HGG detection

4. **More True Positives**: 62 vs 59 (+3 additional HGG cases correctly identified)

### ⚠️ Trade-offs

1. **Higher FP**: 9 vs 4 (+5 additional false alarms)
   - **Medical Impact**: More patients flagged for further investigation
   - **Acceptable trade-off**: False alarms are less critical than missed diagnoses

2. **Lower Precision**: 0.8732 vs 0.9365 (-6.8%)
   - More false positives reduce precision

3. **Lower Specificity**: 0.6087 vs 0.8261 (-26.3%)
   - More LGG cases incorrectly classified as HGG

4. **Slightly Lower F1**: 0.9254 vs 0.9365 (-1.2%)
   - Small decrease due to precision-recall trade-off

5. **Lower Accuracy**: 0.8837 vs 0.9070 (-2.6%)
   - Overall accuracy slightly reduced

---

## Medical Decision Analysis

### Cost-Benefit Assessment

**Cost Reduction**:
- Default: 1.0 cost unit saved (8.3% reduction)
- Conservative: 3.0 cost units saved (20.0% reduction)

**Clinical Impact**:
- **FN Reduction**: 3 fewer missed HGG cases (critical for patient safety)
- **FP Increase**: 5 additional false alarms (less critical, but requires follow-up)

**Medical Justification**:
- In brain tumor classification, **missing a high-grade glioma (FN) is far more serious** than a false alarm (FP)
- False alarms can be resolved with additional imaging or biopsy
- Missed diagnoses can lead to delayed treatment and worse outcomes
- The 75% reduction in FN (4 → 1) is clinically significant

---

## Recommendation

### ✅ **Recommend Adopting Cost-Sensitive Threshold (0.36)**

**Rationale**:

1. **Medical Priority**: The 75% reduction in false negatives (4 → 1) is clinically more important than the increase in false positives (4 → 9). Missing HGG cases has serious consequences.

2. **Cost Efficiency**: The cost-sensitive threshold achieves lower total cost (8.3-25.0% reduction depending on configuration) while prioritizing FN reduction.

3. **High Recall**: Recall of 0.9841 (vs 0.9365) ensures that 98.4% of HGG cases are correctly identified, which is critical for medical screening.

4. **Acceptable Trade-off**: The increase in false positives (5 additional cases) is acceptable given:
   - False alarms can be resolved with follow-up
   - The substantial reduction in missed diagnoses
   - The overall cost reduction

**Implementation**:
- Replace current balanced threshold (0.41) with cost-sensitive threshold (0.36)
- Use for both default and conservative cost configurations (same threshold selected)
- Monitor in production to validate performance on new data

**Caveats**:
- Lower precision (0.8732 vs 0.9365) means more false alarms
- Lower specificity (0.6087 vs 0.8261) means more LGG cases flagged
- Consider clinical workflow impact of increased false positives

---

## Files Generated

- `ensemble/results/calibration/cost_sensitive_thresholds.json`: Detailed results and comparison
- `ensemble/results/calibration/cost_sensitive_summary.md`: This summary document

---

## Next Steps

1. **Review with clinical team**: Validate that the FN/FP trade-off is acceptable for clinical workflow
2. **Update inference script**: Modify default threshold to 0.36 if approved
3. **Monitor performance**: Track FN/FP rates in production to validate the threshold selection
4. **Consider cost weight tuning**: If needed, adjust w_FN/w_FP based on clinical feedback

