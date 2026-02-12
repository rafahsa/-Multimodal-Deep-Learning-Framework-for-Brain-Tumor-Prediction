# Nested CV with Enhanced Meta-Features: Results Report

**Date**: 2026-02-09 00:58:59

## Executive Summary

This report presents results from nested cross-validation using **enhanced meta-features**
engineered from base model probabilities.

---

## Meta-Feature Engineering

### Features Generated

**Core Probability Features**:
- `p_resnet`, `p_swin`, `p_mil`: Individual base model probabilities

**Agreement / Disagreement**:
- `prob_mean`: Mean across base models
- `prob_std`: Standard deviation (measures disagreement)
- `prob_max`, `prob_min`: Range of predictions
- `prob_range`: Max - min (measures spread)

**Confidence / Margin**:
- `margin_mean`: |mean - 0.5| (distance from uncertainty)
- `margin_max`: Maximum margin across models
- `entropy_mean`: Binary entropy of mean probability (uncertainty measure)

**Model Dominance**:
- `argmax_resnet`, `argmax_swin`, `argmax_mil`: One-hot encoded model with highest probability

**Total Features**: 15 (3 base + 12 engineered)

### Medical Relevance

- **Agreement features** help identify cases where all models agree (high confidence)
- **Disagreement features** flag uncertain cases requiring human review
- **Margin features** measure distance from decision boundary (confidence)
- **Entropy** quantifies prediction uncertainty
- **Model dominance** captures which model drives the decision

---

## Results Summary

**Meta-Learner**: Logistic Regression with Enhanced Meta-Features
**Evaluation**: Nested Cross-Validation (5 outer folds)
**Calibration**: Robust Platt scaling (5 repeats, median threshold)

| Metric | Mean ± Std | Range |
|--------|------------|-------|
| FN | 2.80 ± 2.14 | [0, 6] |
| FP | 7.80 ± 2.79 | - |
| Cost | 13.40 ± 5.54 | - |
| Recall | 0.9333 ± 0.0508 | - |
| Precision | 0.8362 ± 0.0530 | - |
| F1 | 0.8812 ± 0.0431 | - |

---

## Comparison with Baseline

| Metric | Baseline (Simple) | Enhanced (Meta-Features) | Change |
|--------|------------------|--------------------------|--------|
| FN | 4.20 ± 2.04 | 2.80 ± 2.14 | +1.40 |
| FP | 6.40 ± 2.73 | 7.80 ± 2.79 | +1.40 |
| Cost | 14.80 ± 2.79 | 13.40 ± 5.54 | +1.40 |

### Improvement Analysis

✅ **FN decreased** by 1.40 (improvement)
✅ **FP change acceptable**: +1.40
✅ **Cost reduced** by 1.40 (improvement)

### Consistency Across Folds

- **FN range**: [0, 6]
- **Worst-case FN**: 6 (medical safety critical)
- **FN std**: 2.14 (variable)

---

## Per-Fold Details

| Fold | FN | FP | Cost | Recall | Precision | F1 | Threshold |
|------|----|----|------|--------|-----------|----|-----------|
| 0 | 4 | 7 | 15.0 | 0.9048 | 0.8444 | 0.8736 | 0.3100 |
| 1 | 0 | 11 | 11.0 | 1.0000 | 0.7925 | 0.8842 | 0.3500 |
| 2 | 3 | 8 | 14.0 | 0.9286 | 0.8298 | 0.8764 | 0.3400 |
| 3 | 1 | 3 | 5.0 | 0.9762 | 0.9318 | 0.9535 | 0.3700 |
| 4 | 6 | 10 | 22.0 | 0.8571 | 0.7826 | 0.8182 | 0.3400 |

---

## Conclusion

✅ **Meta-feature engineering improved performance**

- FN reduced by 1.40
- Cost reduced by 1.40
- Worst-case FN: 6 (acceptable for medical safety)

**Recommendation**: Adopt enhanced meta-features for final model.