# Ensemble Meta-Learner Visualizations

This directory contains visualization plots generated from ensemble evaluation results.

## Operating Points

The ensemble meta-learner uses configurable decision thresholds for binary classification (LGG=0, HGG=1). We select **0.22 as the default operating threshold for the main system (best F1, balanced precision/recall)**. We provide **0.19 as a high-sensitivity operating point to reduce false negatives in HGG detection**.

- **Default (Balanced)**: threshold 0.22 (optimal F1 score, balanced precision and recall)
- **High-sensitivity**: threshold 0.19 (precision ≥ 0.80, higher recall for HGG detection, lower FN)
- **Baseline**: threshold 0.50 (reference point for comparison)

## Generated Plots

### Threshold-Specific Plots

These plots are generated for **each detected threshold**:

- `confusion_matrix_thr_0_19.png`: Confusion matrix at threshold 0.19
- `per_class_performance_thr_0_19.png`: Per-class metrics at threshold 0.19
- `confusion_matrix_thr_0_22.png`: Confusion matrix at threshold 0.22
- `per_class_performance_thr_0_22.png`: Per-class metrics at threshold 0.22
- `confusion_matrix_thr_0_50.png`: Confusion matrix at threshold 0.50
- `per_class_performance_thr_0_50.png`: Per-class metrics at threshold 0.50

**Legacy Mode**: If `--legacy-main-output` flag is used, additional legacy filenames are generated for the main threshold (default: 0.22):
- `confusion_matrix.png`
- `per_class_performance.png`
- `performance_metrics_summary.png`

### Shared Plots (All Thresholds)

- `roc_curve.png`: ROC curve with markers for all operating points
- `precision_recall_curve.png`: Precision-Recall curve with markers for all operating points
- `prediction_distribution.png`: Distribution of predicted probabilities with threshold markers
- `feature_importance.png`: Feature importance (coefficient magnitudes)
- `performance_metrics_comparison.png`: Comparison of all metrics across thresholds

## Performance Summary

| Threshold | Precision | Recall | F1 | Accuracy | FN | FP |
|-----------|-----------|--------|----|----------|----|----|
| 0.19 | 0.8319 | 0.9429 | 0.8839 | 0.8175 | 12 | 40 |
| 0.22 | 0.9000 | 0.9000 | 0.9000 | 0.8526 | 21 | 21 |
| 0.50 | 0.9643 | 0.7714 | 0.8571 | 0.8105 | 48 | 6 |

## How to Reproduce

Run the visualization script with:

```bash
python scripts/ensemble/visualize_meta_learner_results.py --eval-jsons ensemble/results/eval_threshold_0_50.json ensemble/results/eval_threshold_0_22.json ensemble/results/eval_threshold_0_19.json
```

## Notes

- All plots are generated from evaluation JSON files in `ensemble/results/`
- ROC/PR curves and prediction distribution require probability data from OOF predictions
- If probability data is unavailable, those plots will be skipped gracefully
