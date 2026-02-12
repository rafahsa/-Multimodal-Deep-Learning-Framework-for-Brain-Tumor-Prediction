# Meta-Learner V2 Experiment Report

**Experiment Date**: 2026-02-08T21:32:25.370346

## Executive Summary

**Recommendation**: **ADOPT V2**

**Reason**: Best candidate reduces cost from 63.0 to 1.0 (FN: 11 → 0, FP: 41 → 1)

---

## Baseline Performance

- **Model**: Baseline_LogisticRegression
- **Threshold**: 0.3500
- **FN**: 11
- **FP**: 41
- **Cost** (2×FN + FP): 63.0
- **Recall**: 0.9476
- **Precision**: 0.8292
- **F1**: 0.8844

---

## Candidate Models Comparison

| Model | Threshold | FN | FP | Cost | Recall | Precision | F1 |
|-------|-----------|----|----|------|--------|-----------|----|
| **Baseline_LogisticRegression** (baseline) | 0.3500 | 11 | 41 | **63.0** | 0.9476 | 0.8292 | 0.8844 |
| XGBoost_depth4_lr0.1_n100 ⭐ **BEST** | 0.3900 | 0 | 1 | 1.0 | 1.0000 | 0.9953 | 0.9976 |
| XGBoost_depth3_lr0.1_n100 | 0.1600 | 1 | 17 | 19.0 | 0.9952 | 0.9248 | 0.9587 |
| XGBoost_depth4_lr0.1_n50 | 0.4200 | 6 | 9 | 21.0 | 0.9714 | 0.9577 | 0.9645 |
| XGBoost_depth3_lr0.1_n50 | 0.2100 | 0 | 31 | 31.0 | 1.0000 | 0.8714 | 0.9313 |
| LogisticRegression_C0.1_balanced | 0.3900 | 17 | 28 | 62.0 | 0.9190 | 0.8733 | 0.8956 |
| LogisticRegression_C1_none | 0.3800 | 17 | 28 | 62.0 | 0.9190 | 0.8733 | 0.8956 |
| LinearSVC | 0.3900 | 15 | 34 | 64.0 | 0.9286 | 0.8515 | 0.8884 |
| LogisticRegression_C10_balanced | 0.3700 | 15 | 35 | 65.0 | 0.9286 | 0.8478 | 0.8864 |
| Baseline_Reproduction | 0.3600 | 15 | 36 | 66.0 | 0.9286 | 0.8442 | 0.8844 |
| LogisticRegression_C0.1_none | 0.3800 | 15 | 36 | 66.0 | 0.9286 | 0.8442 | 0.8844 |
| LogisticRegression_C1_balanced | 0.3600 | 15 | 36 | 66.0 | 0.9286 | 0.8442 | 0.8844 |
| LogisticRegression_C10_none | 0.3700 | 15 | 36 | 66.0 | 0.9286 | 0.8442 | 0.8844 |

---

## Best Candidate

- **Model**: XGBoost_depth4_lr0.1_n100
- **Threshold**: 0.3900
- **FN**: 0 (-11 vs baseline)
- **FP**: 1 (-40 vs baseline)
- **Cost**: 1.0 (-62.0 vs baseline)
- **Recall**: 1.0000 (+0.0524 vs baseline)
- **Precision**: 0.9953 (+0.1661 vs baseline)
- **F1**: 0.9976 (+0.1132 vs baseline)

---

## Decision Criteria

A candidate meta-learner is recommended if:

1. **Cost reduction**: Lower cost (2×FN + FP) than baseline
2. **FN constraint**: FN ≤ baseline + 2 (medical priority)
3. **FP reduction**: Significant FP reduction if FN is similar

---

## Final Decision

**ADOPT V2**

Best candidate reduces cost from 63.0 to 1.0 (FN: 11 → 0, FP: 41 → 1)
