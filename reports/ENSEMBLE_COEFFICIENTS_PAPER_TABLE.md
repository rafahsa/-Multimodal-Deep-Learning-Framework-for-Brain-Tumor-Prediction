# Meta-Learner Coefficients: Paper/Thesis Summary Table

**Model:** Logistic Regression Meta-Learner  
**Source:** `ensemble/models/meta_learner_logistic_regression.joblib` (deployed model)  
**Training Data:** 285 patients, 5-fold cross-validation OOF predictions

---

## Base Model Contribution Coefficients

| Base Model | Coefficient | |Coefficient| | Rank | Relative Contribution | Interpretation |
|------------|------------|--------------|------|----------------------|----------------|
| **SwinUNETR-3D** | +4.137673 | 4.137673 | 1 | 87.0% | **Dominant influence** |
| **ResNet50-3D** | +0.561032 | 0.561032 | 2 | 11.8% | Moderate influence |
| **DualStreamMIL-3D** | +0.092570 | 0.092570 | 3 | 1.9% | Minimal influence |
| **Intercept** | -2.120502 | - | - | - | Baseline bias term |

**Total Absolute Contribution:** 4.791275

---

## Key Findings

### 1. Model Dominance
- **SwinUNETR-3D dominates** the ensemble with a coefficient of 4.14, representing **87% of total absolute contribution**
- **Dominance ratio:** 44.7× (SwinUNETR-3D / DualStreamMIL-3D)
- This indicates **high dominance** rather than balanced complementarity

### 2. Model Contributions
- **SwinUNETR-3D**: Strongest positive influence (4.14)
  - A 0.1 increase in SwinUNETR-3D probability increases log-odds by ~0.41
  - Primary driver of ensemble HGG predictions
  
- **ResNet50-3D**: Moderate positive influence (0.56)
  - A 0.1 increase in ResNet50-3D probability increases log-odds by ~0.06
  - Secondary contributor, provides complementary signal
  
- **DualStreamMIL-3D**: Minimal positive influence (0.09)
  - A 0.1 increase in MIL probability increases log-odds by ~0.01
  - Very small contribution, may indicate redundancy or calibration issues

### 3. Ensemble Behavior
- **All coefficients are positive**: Ensemble combines models additively
- **No negative coefficients**: No model acts as a "veto" or negative signal
- **High dominance**: SwinUNETR-3D's influence is 7.6× stronger than ResNet50-3D and 44.7× stronger than DualStreamMIL-3D

### 4. Interpretation
- The meta-learner has learned that **SwinUNETR-3D is the most reliable predictor**
- ResNet50-3D provides **complementary information** but with much lower weight
- DualStreamMIL-3D has **minimal impact**, suggesting it may be:
  - Redundant with other models
  - Poorly calibrated
  - Less informative for this task

---

## Mathematical Formulation

The ensemble prediction is computed as:

```
P(HGG) = σ(4.14 × p_swin + 0.56 × p_resnet + 0.09 × p_mil - 2.12)
```

Where:
- `σ` is the sigmoid function
- `p_swin`, `p_resnet`, `p_mil` are base model HGG probabilities
- The intercept (-2.12) provides a baseline bias toward LGG predictions

---

## Comparison: Deployed Model vs Metrics File

**Note:** The metrics file (`meta_learner_metrics.json`) contains slightly different coefficients from a different training run:

| Model | Deployed Model | Metrics File | Difference |
|-------|----------------|--------------|------------|
| SwinUNETR-3D | 4.137673 | 4.063425 | -0.074 |
| ResNet50-3D | 0.561032 | 0.536982 | -0.024 |
| DualStreamMIL-3D | 0.092570 | 0.890013 | +0.797 |
| Intercept | -2.120502 | -2.404859 | -0.284 |

**Key Difference:** The metrics file shows DualStreamMIL-3D with a much higher coefficient (0.89 vs 0.09), suggesting different training conditions or data. The **deployed model** (analyzed here) is the actual model used in production.

---

## Recommendations

1. **SwinUNETR-3D is critical**: Any degradation in SwinUNETR-3D performance will significantly impact ensemble performance
2. **ResNet50-3D provides value**: Despite lower weight, it contributes complementary information
3. **DualStreamMIL-3D review**: Consider investigating why MIL has minimal contribution:
   - Check calibration quality
   - Evaluate if MIL predictions are redundant
   - Consider retraining or recalibration if needed

---

*Generated from: `scripts/analysis/analyze_ensemble_contributions.py`*  
*Date: 2026-02-10*

