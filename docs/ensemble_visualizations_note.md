# Ensemble Meta-Learner Visualizations: Important Note

## About Learning Curves

**Important**: Logistic Regression does not train over "epochs" like neural networks. Instead, it optimizes the objective function directly using iterative solvers (e.g., LBFGS) until convergence. Therefore, **there are no epoch-by-epoch metrics to plot as learning curves**.

Instead, the visualizations include:
- **Performance Metrics Summary**: A comprehensive view of all final performance metrics (Accuracy, Precision, Recall, F1-Score, AUC-ROC)
- **ROC Curve**: Shows model performance across all decision thresholds (not epoch-based)
- **Precision-Recall Curve**: Shows precision-recall trade-off (not epoch-based)
- **Confusion Matrix**: Final classification performance
- **Feature Importance**: Importance of each base model in the ensemble

These visualizations provide a complete picture of the meta-learner's performance, even though they don't show training progression over epochs (which doesn't exist for Logistic Regression).

## Generated Visualizations

All visualizations are saved in `ensemble/visualizations/`:

1. **roc_curve.png**: ROC curve with AUC value
2. **precision_recall_curve.png**: Precision-Recall curve with Average Precision
3. **confusion_matrix.png**: Confusion matrix (counts and normalized)
4. **feature_importance.png**: Feature importance based on learned coefficients
5. **performance_metrics_summary.png**: Summary of all performance metrics
6. **per_class_performance.png**: Per-class (LGG/HGG) performance metrics
7. **prediction_distribution.png**: Distribution of predicted probabilities by true class

## Usage

To regenerate visualizations:

```bash
python scripts/ensemble/generate_visualizations.py
```

All visualizations are generated at 300 DPI for publication quality.

