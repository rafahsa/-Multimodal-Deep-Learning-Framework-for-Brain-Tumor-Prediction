# Ensemble Base Model Contribution Analysis

**Generated:** 2026-02-10T22:11:20.034138

## Overview

This report analyzes the contribution of each base model to the final ensemble decision by examining the meta-learner (Logistic Regression) coefficients.

---

## Meta-Learner Coefficients Summary Table

| Base Model | Coefficient | |Coefficient| | Rank | Interpretation |
|------------|------------|--------------|------|----------------|
| SwinUNETR-3D | +4.137673 | 4.137673 | 1 | Dominant influence |
| ResNet50-3D | +0.561032 | 0.561032 | 2 | Strong influence |
| DualStreamMIL-3D | +0.092570 | 0.092570 | 3 | Minimal influence |

**Intercept**: -2.120502


---

## Coefficient Interpretation

### Base Model Contributions

1. **SwinUNETR-3D**: Coefficient = 4.137673 (|coef| = 4.137673)
   - Positive influence: Higher SwinUNETR-3D probability → Higher ensemble HGG probability
   - **Strong influence**: Model is a key contributor to ensemble decision

2. **ResNet50-3D**: Coefficient = 0.561032 (|coef| = 0.561032)
   - Positive influence: Higher ResNet50-3D probability → Higher ensemble HGG probability
   - **Strong influence**: Model is a key contributor to ensemble decision

3. **DualStreamMIL-3D**: Coefficient = 0.092570 (|coef| = 0.092570)
   - Positive influence: Higher DualStreamMIL-3D probability → Higher ensemble HGG probability
   - ⚠️ **Very small coefficient**: Model has minimal influence

### Model Dominance Analysis

- **Dominance Ratio**: 44.70x (strongest / weakest)
- **Strongest Model**: SwinUNETR-3D
- **Weakest Model**: DualStreamMIL-3D
- **Interpretation**: SwinUNETR-3D dominates the ensemble decision (high dominance)

### Overall Ensemble Behavior

- All base models have **positive coefficients**: Ensemble combines models additively
- Higher probabilities from any model increase ensemble HGG probability

---

## Note on Enhanced Meta-Learner

The best performing configuration uses an **Enhanced Meta-Learner** with meta-features (see `nested_cv_meta_features/meta_features_results_20260209_005859.json`). However, the deployed model file uses only the 3 base model probabilities.

**Enhanced Meta-Learner Features:**
- p_resnet
- p_swin
- p_mil
- prob_mean
- prob_std
- prob_max
- prob_min
- prob_range
- margin_mean
- margin_max
- entropy_mean
- argmax_resnet
- argmax_swin
- argmax_mil

The enhanced version includes probability statistics, margins, entropy, and argmax indicators in addition to base model probabilities, which improves performance but makes coefficient interpretation more complex.
