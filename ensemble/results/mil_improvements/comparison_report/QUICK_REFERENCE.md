# Quick Reference: MIL Model Comparison

## Decision Matrix

| Criterion | Weight | OLD MIL | NEW MIL | Winner | Notes |
|-----------|--------|---------|---------|--------|-------|
| **Recall** | ⭐⭐⭐⭐⭐ | 0.224 | 0.524 | **NEW** | Critical for medical use |
| **False Negatives** | ⭐⭐⭐⭐⭐ | 163 | 100 | **NEW** | 63 fewer missed cases |
| **ROC-AUC** | ⭐⭐⭐⭐ | 0.730 | 0.621 | **OLD** | Overall discrimination |
| **PR-AUC** | ⭐⭐⭐ | 0.878 | 0.843 | **OLD** | Precision-recall balance |
| **Class Separation** | ⭐⭐⭐ | 0.097 | 0.051 | **OLD** | Signal quality |
| **Probability Range** | ⭐⭐ | 0.838 | 0.565 | **OLD** | Ensemble informativeness |
| **HGG High Confidence** | ⭐⭐⭐ | 34 (16%) | 67 (32%) | **NEW** | 2x improvement |

## Final Decision: **CONDITIONAL** → **NO** (with path to YES)

### Quick Summary:
- ✅ **NEW MIL wins on medical priorities** (recall, FN reduction)
- ⚠️ **OLD MIL wins on discrimination** (ROC-AUC, separation)
- 🎯 **NEW MIL needs improvement** before replacement

## Target Metrics for Replacement:

| Metric | Current NEW | Target | Gap |
|--------|------------|--------|-----|
| ROC-AUC | 0.621 | ≥0.70 | -0.079 |
| Recall | 0.524 | ≥0.85 | -0.326 |
| FN Count | 100 | ≤50 | +50 |
| Separation | 0.051 | ≥0.08 | -0.029 |
| Range Width | 0.565 | ≥0.70 | -0.135 |

## Top 3 Improvement Priorities:

1. **Class Separation** → Tune entropy sampling, add confidence regularization
2. **Recall** → Adjust loss function (FN penalty), class weights
3. **Probability Range** → Regularization schedule, multi-scale entropy

## Files Generated:

- `comparison_report.txt` - Full detailed report
- `EXECUTIVE_SUMMARY.md` - Comprehensive analysis (this file's parent)
- `comparison_plots.png` - Visualizations (ROC, PR, distributions)
- `QUICK_REFERENCE.md` - This file

