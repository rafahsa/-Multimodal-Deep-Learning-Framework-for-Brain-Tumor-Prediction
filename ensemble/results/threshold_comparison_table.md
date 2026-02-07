# Threshold Comparison Table

This table compares ensemble classifier performance at three different thresholds on OOF (validation) data.

| Threshold | TN | FP | FN | TP | Precision | Recall | F1 | Accuracy |
|-----------|----|----|----|----|-----------|--------|----|----------|
| 0.50 | 69 | 6 | 48 | 162 | 0.9643 | 0.7714 | 0.8571 | 0.8105 |
| 0.22 | 54 | 21 | 21 | 189 | 0.9000 | 0.9000 | 0.9000 | 0.8526 |
| 0.19 | 35 | 40 | 12 | 198 | 0.8319 | 0.9429 | 0.8839 | 0.8175 |

## Notes

- **Baseline (0.50)**: Default threshold, high precision but lower recall
- **Balanced (0.22)**: Optimal F1 score, balanced precision and recall
- **High-sensitivity (0.19)**: Higher recall for HGG detection, precision ≥ 0.80

All results are from OOF predictions (validation data from 5-fold cross-validation).
