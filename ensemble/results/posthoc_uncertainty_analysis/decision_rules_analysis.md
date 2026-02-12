# Post-Hoc Uncertainty-Aware Decision Analysis for Swin-1

**Objective:** Evaluate decision-rule variants on Swin-1 predictions to reduce False Negatives (FN) while keeping False Positives (FP) under control.

**Method:** Post-hoc decision logic analysis (NO retraining, NO model modification)

---

## Comparison Table

| Method | Threshold | Reject Zone | Precision | Recall | F1 | FN Count | FP Count |
|--------|-----------|-------------|-----------|--------|----|----------|----------|
| Simple Threshold | 0.50 | - | 0.9874 | 0.7476 | 0.8509 | 53 | 2 |
| Simple Threshold | 0.45 | - | 0.9755 | 0.7571 | 0.8525 | 51 | 4 |
| Simple Threshold | 0.40 | - | 0.9758 | 0.7667 | 0.8587 | 49 | 4 |
| Simple Threshold | 0.35 | - | 0.9643 | 0.7714 | 0.8571 | 48 | 6 |
| Reject Zone | - | [0.40-0.60] | 0.9758 | 0.7667 | 0.8587 | 49 | 4 |
| Reject Zone | - | [0.35-0.65] | 0.9643 | 0.7714 | 0.8571 | 48 | 6 |
| Reject Zone | - | [0.45-0.65] | 0.9755 | 0.7571 | 0.8525 | 51 | 4 |

## Analysis

### 1. Best FN Reduction

**Configuration:** Simple Threshold (threshold = 0.35)

- **FN:** 48 (lowest)
- **FP:** 6
- **Precision:** 0.9643
- **Recall:** 0.7714
- **F1:** 0.8571

### 2. Best FP Control

**Configuration:** Simple Threshold (threshold = 0.50)

- **FN:** 53
- **FP:** 2 (lowest)
- **Precision:** 0.9874
- **Recall:** 0.7476
- **F1:** 0.8509

### 3. Best Overall Tradeoff (F1 Score)

**Configuration:** Simple Threshold (threshold = 0.40)

- **FN:** 49
- **FP:** 4
- **Precision:** 0.9758
- **Recall:** 0.7667
- **F1:** 0.8587 (highest)

## Recommendations

### Recommended Operating Point

No configuration achieves both recall ≥ 0.90 and precision ≥ 0.85.

**Alternative:** Use the configuration with best F1 score (balanced tradeoff).

## Key Findings

1. **FN Reduction:** Lower thresholds and wider reject zones reduce FN but may increase FP.
2. **FP Control:** Higher thresholds and narrower reject zones reduce FP but may increase FN.
3. **Tradeoff:** The optimal configuration depends on clinical priorities (FN reduction vs FP control).
4. **Reject Zones:** Uncertainty-aware reject zones can provide a good balance between FN and FP.

---

*Generated: Post-hoc decision analysis for Swin-1*  
*No model retraining or modification performed*
