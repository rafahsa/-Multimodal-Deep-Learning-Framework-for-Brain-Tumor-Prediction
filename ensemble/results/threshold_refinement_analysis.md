# Threshold Refinement Analysis: 0.35 vs 0.36

## Executive Summary

**Recommendation: KEEP threshold 0.35**

While threshold 0.36 shows lower FP on the held-out threshold selection set (n=86), evaluation on the full OOF dataset (n=285) reveals a **significant increase in FN (+4 cases, 36% increase)**. Given the medical priority of minimizing false negatives (missed HGG diagnoses), threshold 0.35 should be retained as the final operating point.

---

## Evaluation Results

### Full OOF Dataset (n=285)

| Metric | Threshold 0.35 | Threshold 0.36 | Change |
|--------|----------------|----------------|--------|
| **FN** | **11** | 15 | **+4 (36% increase)** ⚠️ |
| **FP** | 41 | **36** | **-5 (12% decrease)** ✅ |
| **TN** | 34 | **39** | +5 |
| **TP** | **199** | 195 | -4 |
| **Precision** | 0.8292 | **0.8442** | +0.0150 |
| **Recall** | **0.9476** | 0.9286 | -0.0190 |
| **F1** | 0.8844 | 0.8844 | -0.0001 |
| **Accuracy** | 0.8175 | **0.8211** | +0.0035 |
| **Specificity** | 0.4533 | **0.5200** | +0.0667 |

### Held-Out Threshold Selection Set (n=86)

| Metric | Threshold 0.35 | Threshold 0.36 | Change |
|--------|----------------|----------------|--------|
| **FN** | 1 | 1 | **0 (no change)** ✅ |
| **FP** | 11 | **9** | **-2 (18% decrease)** ✅ |
| **TN** | 12 | **14** | +2 |
| **TP** | 62 | 62 | 0 |
| **Precision** | 0.8493 | **0.8732** | +0.0239 |
| **Recall** | 0.9841 | 0.9841 | 0.0000 |
| **F1** | 0.9118 | **0.9254** | +0.0136 |

---

## Key Findings

### 1. Discrepancy Between Evaluation Sets

**Held-Out Set (n=86)**:
- FN remains constant (1) at both thresholds
- FP decreases by 2 at 0.36
- **Conclusion**: 0.36 appears strictly better on this set

**Full OOF Set (n=285)**:
- FN increases by 4 (11 → 15) at 0.36
- FP decreases by 5 (41 → 36) at 0.36
- **Conclusion**: Trade-off exists; FN increase is significant

### 2. Medical Safety Analysis

**Threshold 0.35 (Full OOF)**:
- FN = 11 (5.24% of HGG cases missed)
- Recall = 0.9476 (94.76% sensitivity)
- **Medical Impact**: 11 missed HGG diagnoses

**Threshold 0.36 (Full OOF)**:
- FN = 15 (7.14% of HGG cases missed)
- Recall = 0.9286 (92.86% sensitivity)
- **Medical Impact**: 15 missed HGG diagnoses (+4 additional misses)

**Critical Finding**: Threshold 0.36 results in **4 additional missed HGG cases** compared to 0.35. This represents a **36% increase in false negatives**, which is clinically significant.

### 3. False Positive Burden

**Threshold 0.35 (Full OOF)**:
- FP = 41 (54.67% of LGG cases flagged)
- **Clinical Impact**: 41 false alarms requiring follow-up

**Threshold 0.36 (Full OOF)**:
- FP = 36 (48.00% of LGG cases flagged)
- **Clinical Impact**: 36 false alarms (5 fewer than 0.35)

**Trade-off**: While 0.36 reduces FP by 5 cases (12% reduction), this benefit is outweighed by the 4 additional FN cases.

---

## Medical Decision Analysis

### Priority: Minimizing False Negatives

In brain tumor classification, **missing a high-grade glioma (FN) is far more serious** than a false alarm (FP):

1. **False Negatives (FN)**:
   - Missed HGG diagnosis → Delayed treatment → Worse patient outcomes
   - Can lead to disease progression and reduced survival
   - **Unacceptable risk** in medical screening

2. **False Positives (FP)**:
   - LGG case flagged as HGG → Additional imaging/biopsy → Resolved with follow-up
   - Causes patient anxiety and additional testing, but **no direct harm**
   - **Acceptable trade-off** for improved sensitivity

### Cost-Benefit Assessment

**Threshold 0.35**:
- Cost: 41 FP (false alarms)
- Benefit: 11 FN (missed diagnoses)
- **Net**: Prioritizes medical safety

**Threshold 0.36**:
- Cost: 36 FP (false alarms) ✅ Lower
- Benefit: 15 FN (missed diagnoses) ⚠️ Higher
- **Net**: Reduces false alarms but increases missed diagnoses

**Conclusion**: The 5-case reduction in FP at 0.36 does **not justify** the 4-case increase in FN, given the medical priority of minimizing missed HGG diagnoses.

---

## Statistical Analysis

### Why the Discrepancy?

The held-out threshold selection set (n=86) is smaller and may not fully represent the distribution of the full OOF dataset (n=285). The threshold selection process optimized on the held-out set, but evaluation on the full set reveals different behavior:

- **Small sample size effect**: The held-out set (n=86) may have different class distributions or difficulty patterns
- **Threshold sensitivity**: Small changes in threshold (0.35 → 0.36) can have different impacts on different subsets
- **Generalization**: The full OOF set (n=285) provides a more robust evaluation

### Recommendation: Use Full OOF Set for Final Decision

The full OOF dataset (n=285) is:
- **More representative**: Larger sample size, better statistical power
- **More reliable**: Less sensitive to small-sample variations
- **More clinically relevant**: Reflects performance on the complete validation set

---

## Final Recommendation

### ✅ **KEEP Threshold 0.35**

**Rationale**:

1. **Medical Safety**: Threshold 0.35 maintains FN=11 (5.24% miss rate) vs FN=15 (7.14% miss rate) at 0.36. The 4 additional missed HGG cases at 0.36 are clinically unacceptable.

2. **Priority Alignment**: The system's medical priority is minimizing false negatives (missed HGG diagnoses). Threshold 0.35 better aligns with this priority.

3. **Robust Evaluation**: While 0.36 appears better on the held-out set (n=86), evaluation on the full OOF set (n=285) shows a significant FN increase. The full set provides a more reliable assessment.

4. **Acceptable Trade-off**: The 5 additional FP cases at 0.35 (41 vs 36) are acceptable given:
   - False alarms can be resolved with follow-up
   - The substantial reduction in missed diagnoses
   - The medical priority of patient safety

**Conclusion**: Threshold 0.35 should remain the final adopted operating point. The slight increase in FP is medically justified by the critical reduction in FN.

---

## Implementation

- **Final Threshold**: 0.35 (unchanged)
- **Visualizations**: Current visualizations using threshold 0.35 are correct and should be retained
- **Documentation**: No changes needed to system configuration

---

## Files Generated

- `ensemble/results/threshold_comparison_0_35_vs_0_36.json`: Detailed comparison metrics
- `ensemble/results/threshold_refinement_analysis.md`: This analysis document

