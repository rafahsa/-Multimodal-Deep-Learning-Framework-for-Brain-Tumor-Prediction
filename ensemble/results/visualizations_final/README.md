# Final Visualization Set

## System Configuration

**Meta-Learner**: Logistic Regression  
**Calibration**: Platt scaling (from run: 2026-02-07_22-29-29_platt_seed42)  
**Final Threshold**: 0.35 (cost-sensitive, stability-averaged)  
**Medical Priority**: Minimizing false negatives (HGG misses)

All plots use **calibrated ensemble probabilities** and reflect the FINAL system configuration.

---

## Generated Plots

### 1. `reliability_diagram_before_after.png`
Reliability diagram comparing uncalibrated vs Platt-calibrated probabilities. Shows Expected Calibration Error (ECE) for both. Demonstrates improved probability reliability after calibration.

### 2. `confusion_matrix_final_thr_0_35.png`
Confusion matrix at the final threshold (0.35). Clearly labels LGG (negative) and HGG (positive) classes. Annotates FN and FP counts for medical interpretation.

### 3. `per_class_performance_thr_0_35.png`
Bar chart showing Precision, Recall, and F1-score for LGG and HGG classes at threshold 0.35. Provides class-specific performance metrics.

### 4. `prediction_distribution_thr_0_35.png`
Histogram of calibrated ensemble probabilities, separated by true class (LGG vs HGG). Vertical line marks the decision threshold (0.35).

### 5. `roc_curve_calibrated.png`
ROC curve using calibrated probabilities. Threshold 0.35 is marked on the curve. **Note**: ROC is NOT used for threshold selection; shown for informational purposes only.

### 6. `precision_recall_curve_calibrated.png`
Precision-Recall curve using calibrated probabilities. Threshold 0.35 is marked on the curve.

### 7. `meta_learner_feature_importance.png`
Logistic Regression coefficients showing the relative contribution of each base model (ResNet50-3D, SwinUNETR-3D, DualStreamMIL-3D) to the ensemble meta-learner.

### 8. `fn_fp_tradeoff_curve.png`
**CRITICAL**: FN-FP trade-off curve showing the relationship between false positives and false negatives across different thresholds. Highlights three key operating points:
- **0.41**: Previous balanced threshold
- **0.36**: Single-run cost-sensitive threshold
- **0.35**: FINAL adopted threshold (stability-averaged)

This plot visually justifies the medical decision to prioritize FN reduction.

---

## Medical Interpretation

**Threshold Selection Rationale**: The final threshold (0.35) was selected through cost-sensitive optimization with stability analysis across multiple calibration runs. This ensures:
- **Minimized False Negatives**: Critical for HGG detection (missed diagnoses can lead to delayed treatment)
- **Stable Performance**: Robust across different data splits
- **Medical Justification**: Acceptable trade-off between FN and FP, prioritizing patient safety

**All plots correspond to the FINAL system configuration and are ready for presentation/publication.**
