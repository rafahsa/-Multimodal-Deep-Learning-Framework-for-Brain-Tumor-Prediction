# Post-Hoc Decision Analysis for Swin-1

## Comparison Table

| Method | Configuration | Precision | Recall | F1 | FN | FP |
|--------|---------------|-----------|--------|----|----|----|
| Simple Threshold | threshold=0.50 | 0.9874 | 0.7476 | 0.8509 | 53 | 2 |
| Simple Threshold | threshold=0.45 | 0.9755 | 0.7571 | 0.8525 | 51 | 4 |
| Simple Threshold | threshold=0.40 | 0.9758 | 0.7667 | 0.8587 | 49 | 4 |
| Simple Threshold | threshold=0.35 | 0.9643 | 0.7714 | 0.8571 | 48 | 6 |
| Reject Zone | [0.40 - 0.60] | 0.9758 | 0.7667 | 0.8587 | 49 | 4 |
| Reject Zone | [0.35 - 0.65] | 0.9643 | 0.7714 | 0.8571 | 48 | 6 |
| Reject Zone | [0.45 - 0.65] | 0.9755 | 0.7571 | 0.8525 | 51 | 4 |


## Analysis

### 1. Best FN Reduction

**Method:** Simple Threshold with threshold=0.35

- **FN:** 48
- **FP:** 6
- **Precision:** 0.9643
- **Recall:** 0.7714
- **F1:** 0.8571

### 2. Best FP Control

**Method:** Simple Threshold with threshold=0.50

- **FN:** 53
- **FP:** 2
- **Precision:** 0.9874
- **Recall:** 0.7476
- **F1:** 0.8509

### 3. Best Overall Tradeoff

**Method:** Simple Threshold with threshold=0.40

- **FN:** 49
- **FP:** 4
- **Precision:** 0.9758
- **Recall:** 0.7667
- **F1:** 0.8587

## Written Analysis

### Which configuration gives the best FN reduction?

The Simple Threshold with threshold=0.35 achieves the lowest FN count (48). This configuration prioritizes recall (0.7714) over precision (0.9643), resulting in 6 false positives.

### Which configuration keeps FP under control?

The Simple Threshold with threshold=0.50 achieves the best FP control (2 FP) while maintaining reasonable recall (0.7476). This configuration has 53 false negatives and precision of 0.9874.

### Which configuration offers the best overall tradeoff?

The Simple Threshold with threshold=0.40 offers the best overall tradeoff. It achieves:
- **FN:** 49
- **FP:** 4
- **Precision:** 0.9758
- **Recall:** 0.7667
- **F1:** 0.8587

⚠️ **This configuration does not reach 90% recall. Consider if this meets clinical requirements.**

### Final Recommendation

**Selected Decision Rule:** Simple Threshold with threshold=0.40

**Rationale:**
- Minimizes FN as much as possible (49 FN)
- Keeps FP reasonably low (4 FP)
- Achieves reasonable recall (0.7667) with precision (0.9758)
- Best overall F1-score (0.8587)

---
*Analysis Date: 2026-02-10*
*Model: Swin-1 (no retraining, post-hoc decision logic only)*
