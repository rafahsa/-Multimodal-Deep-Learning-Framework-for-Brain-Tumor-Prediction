# Rule Override Evaluation Report

## Summary

This report compares baseline Swin-1 predictions with rule-based override predictions.

### Key Changes

- **FN Reduction**: 4 (+7.5%)
- **FP Change**: +2 (+100.0%)

## Metrics Comparison

| Metric | Baseline | Final | Change |
|--------|----------|-------|--------|
| **TP** | 157 | 161 | +4 |
| **TN** | 73 | 71 | -2 |
| **FP** | 2 | 4 | +2 |
| **FN** | 53 | 49 | -4 |
| **Precision** | 0.9874 | 0.9758 | -0.0117 |
| **Recall** | 0.7476 | 0.7667 | +0.0190 |
| **F1** | 0.8509 | 0.8587 | +0.0077 |
| **Accuracy** | 0.8070 | 0.8140 | +0.0070 |

## Baseline Confusion Matrix

```
                Predicted
              LGG    HGG
Actual LGG      73       2
      HGG       53     157
```

## Final Confusion Matrix

```
                Predicted
              LGG    HGG
Actual LGG      71       4
      HGG       49     161
```

## Interpretation

- **False Negative Reduction**: The rule override reduced false negatives by 4 (+7.5%), which means 4 more HGG cases were correctly identified.
- **False Positive Impact**: The rule override increased false positives by 2 (+100.0%).
- **Overall Performance**: Improved F1 score from 0.8509 to 0.8587.

## Notes

- The rule override only applies to uncertain LGG predictions (baseline_pred==0 AND uncertainty_status=="uncertain").
- Confident samples always keep their baseline predictions.
- Swin-1 threshold remains fixed at 0.5 for baseline predictions.
