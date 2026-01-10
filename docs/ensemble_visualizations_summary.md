# Ensemble Meta-Learner: Visualization Summary

**Date**: January 9, 2026  
**Status**: ✅ All Visualizations Generated

---

## Important Note on Learning Curves

**Logistic Regression does not have "epochs"** in the same sense as neural networks. Unlike iterative neural network training, Logistic Regression optimizes its objective function directly using iterative solvers (e.g., LBFGS, Newton-CG) until convergence, typically in a single pass or a few iterations.

Therefore, **learning curves showing performance over epochs cannot be generated** for this model, as there is no epoch-by-epoch training progression to plot.

Instead, we provide:
- **Performance Metrics Summary**: A comprehensive visualization of all final performance metrics
- **ROC Curve**: Performance across all decision thresholds
- **Precision-Recall Curve**: Precision-recall trade-off analysis
- **Additional diagnostic plots**: Confusion matrix, feature importance, prediction distributions

These visualizations provide a complete picture of the meta-learner's performance on the OOF predictions.

---

## Generated Visualizations

All visualizations are saved in `ensemble/visualizations/` and are generated at 300 DPI for publication quality.

### 1. ROC Curve (`roc_curve.png`)

- **Description**: Receiver Operating Characteristic curve showing the trade-off between True Positive Rate (Sensitivity) and False Positive Rate (1 - Specificity) across all decision thresholds
- **Metric**: AUC-ROC = 0.9126
- **Interpretation**: The curve shows strong discriminative ability, with AUC significantly above the random classifier baseline (0.50)

### 2. Precision-Recall Curve (`precision_recall_curve.png`)

- **Description**: Precision-Recall curve showing the relationship between Precision and Recall across all decision thresholds
- **Metric**: Average Precision (AP) calculated from the curve
- **Interpretation**: Useful for imbalanced datasets; shows precision-recall trade-off for the ensemble

### 3. Confusion Matrix (`confusion_matrix.png`)

- **Description**: Two-panel visualization showing:
  - **Left panel**: Confusion matrix with absolute counts
  - **Right panel**: Confusion matrix normalized to percentages
- **Classes**: LGG (class 0) and HGG (class 1)
- **Results**:
  - True Negatives (LGG predicted as LGG): 69
  - False Positives (LGG predicted as HGG): 6
  - False Negatives (HGG predicted as LGG): 48
  - True Positives (HGG predicted as HGG): 162

### 4. Feature Importance (`feature_importance.png`)

- **Description**: Two-panel visualization showing:
  - **Left panel**: Absolute coefficient values (feature importance)
  - **Right panel**: Signed coefficient values (positive/negative contributions)
- **Feature Importance Ranking**:
  1. **SwinUNETR-3D** (coefficient: 4.0634) - Most important
  2. **DualStreamMIL-3D** (coefficient: 0.8900) - Moderate importance
  3. **ResNet50-3D** (coefficient: 0.5370) - Least important
- **Interpretation**: Higher absolute coefficient indicates greater influence on the final prediction

### 5. Performance Metrics Summary (`performance_metrics_summary.png`)

- **Description**: Two-panel visualization showing final performance metrics:
  - **Left panel**: Vertical bar chart
  - **Right panel**: Horizontal bar chart (for better readability)
- **Metrics Displayed**:
  - Accuracy: 0.8105 (81.05%)
  - Precision: 0.9643 (96.43%)
  - Recall: 0.7714 (77.14%)
  - F1-Score: 0.8571 (85.71%)
  - AUC-ROC: 0.9126 (91.26%)
- **Note**: This serves as a substitute for "learning curves" since Logistic Regression doesn't train over epochs

### 6. Per-Class Performance (`per_class_performance.png`)

- **Description**: Bar chart showing Precision, Recall, and F1-Score for each class separately
- **LGG (Class 0)**:
  - Precision: 0.5897
  - Recall: 0.9200
  - F1-Score: 0.7188
- **HGG (Class 1)**:
  - Precision: 0.9643
  - Recall: 0.7714
  - F1-Score: 0.8571

### 7. Prediction Distribution (`prediction_distribution.png`)

- **Description**: Two-panel visualization showing distribution of predicted probabilities:
  - **Left panel**: Histogram showing probability distributions for LGG and HGG classes
  - **Right panel**: Box plots showing probability distributions by true class
- **Interpretation**: Shows how well-separated the predicted probabilities are between the two classes, and the location of the decision boundary (threshold = 0.5)

---

## Visualization Files

| File | Size | Description |
|------|------|-------------|
| `roc_curve.png` | 206 KB | ROC curve with AUC value |
| `precision_recall_curve.png` | 162 KB | Precision-Recall curve with AP |
| `confusion_matrix.png` | 186 KB | Confusion matrix (counts and normalized) |
| `feature_importance.png` | 168 KB | Feature importance (coefficients) |
| `performance_metrics_summary.png` | 244 KB | Performance metrics summary |
| `per_class_performance.png` | 121 KB | Per-class performance metrics |
| `prediction_distribution.png` | 270 KB | Prediction probability distributions |

**Total**: 7 visualization files, all at 300 DPI for publication quality

---

## Performance Metrics Summary

### Overall Metrics

- **Accuracy**: 0.8105 (81.05%)
- **Precision**: 0.9643 (96.43%)
- **Recall**: 0.7714 (77.14%)
- **F1-Score**: 0.8571 (85.71%)
- **AUC-ROC**: 0.9126 (91.26%)

### Per-Class Metrics

**LGG (Class 0)**:
- Precision: 0.5897
- Recall: 0.9200
- F1-Score: 0.7188
- Support: 75 patients

**HGG (Class 1)**:
- Precision: 0.9643
- Recall: 0.7714
- F1-Score: 0.8571
- Support: 210 patients

### Confusion Matrix

```
                Predicted
                LGG  HGG
Actual LGG       69    6
      HGG        48  162
```

---

## Feature Importance Analysis

The Logistic Regression meta-learner assigned the following importance to base model predictions:

1. **SwinUNETR-3D** (coefficient: 4.0634)
   - Highest importance: Most influential in final predictions
   - Positive coefficient: Higher SwinUNETR-3D predictions increase HGG probability

2. **DualStreamMIL-3D** (coefficient: 0.8900)
   - Moderate importance: Second most influential
   - Positive coefficient: Higher MIL predictions increase HGG probability

3. **ResNet50-3D** (coefficient: 0.5370)
   - Lowest importance: Least influential
   - Positive coefficient: Higher ResNet50-3D predictions increase HGG probability

**Intercept**: -2.4049 (negative bias toward LGG class)

---

## Visualization Generation Script

**Script**: `scripts/ensemble/generate_visualizations.py`

**Usage**:
```bash
python scripts/ensemble/generate_visualizations.py
```

**Dependencies**:
- matplotlib
- seaborn
- scikit-learn
- numpy
- pandas
- joblib

**Output**: All visualizations saved to `ensemble/visualizations/` directory

---

## Report Readiness

All visualizations are:
- ✅ Generated at 300 DPI (publication quality)
- ✅ Properly labeled with axes, titles, and legends
- ✅ Include relevant performance metrics
- ✅ Use clear, professional styling
- ✅ Suitable for inclusion in academic reports or presentations

The visualizations provide a comprehensive evaluation of the ensemble meta-learner performance and can be directly included in project reports.

---

**Generation Date**: January 9, 2026  
**Script**: `scripts/ensemble/generate_visualizations.py`  
**Output Directory**: `ensemble/visualizations/`

