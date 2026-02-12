# Brain Tumor Classification Project Status Report

**Generated:** 2026-02-10T20:40:08.265974

## Executive Summary

This report summarizes all experiments in the brain tumor classification project,
including baseline models, ensemble methods, MIL improvements, and post-hoc analyses.

### Target Metrics
- **Accuracy ≥ 92%**
- **Recall ≥ 92%**
- **FN < 10**
- **FP < 10**

---

## Current Best Ensemble Configuration


**Best Nested CV Result:**
- **Meta-Learner:** LogisticRegression_Enhanced
- **Features:** p_resnet, p_swin, p_mil, prob_mean, prob_std, prob_max, prob_min, prob_range, margin_mean, margin_max, entropy_mean, argmax_resnet, argmax_swin, argmax_mil
- **FN (mean ± std):** 2.8 ± 2.1
- **FP (mean ± std):** 7.8 ± 2.8
- **Recall (mean ± std):** 0.933 ± 0.051
- **F1 (mean ± std):** 0.881 ± 0.043
- **Source:** /workspace/brain_tumor_project/ensemble/results/nested_cv_meta_features/meta_features_results_20260209_005859.json


## Results Summary Table

| Experiment | Model/Method | FN (mean±std) | FP (mean±std) | Precision | Recall | F1 | Accuracy | ROC-AUC | Status vs Target |
|------------|--------------|---------------|---------------|------------|--------|----|----------|---------|------------------|
| Mil-Prob_baseline | Mil-Prob | 0.0 | 75.0 | 0.737 | 1.0 | 0.848 | 0.737 | 0.596 | ⚠️ Meets FN/Recall, FP/Accuracy needs work |
| Resnet_baseline | Resnet | 1.0 | 68.0 | 0.755 | 0.995 | 0.858 | 0.758 | 0.610 | ⚠️ Meets FN/Recall, FP/Accuracy needs work |
| nested_cv_meta_features | LogisticRegression_Enhanced | 2.8±2.1 | 7.8±2.8 | 0.836 | 0.933 | 0.881 | nan | nan | ⚠️ Meets FN/Recall, FP/Accuracy needs work |
| Simple_Ensemble_Average | Simple Ensemble (Average) | 4.0 | 67.0 | 0.755 | 0.981 | 0.853 | 0.751 | 0.896 | ⚠️ Meets FN/Recall, FP/Accuracy needs work |
| hybrid_safety_net | Unknown | 4.8±1.5 | 2.0±1.4 | 0.773 | 0.527 | 0.606 | nan | 0.453 | ❌ Below Targets |
| meta_decision | Unknown | 9.0±4.7 | 1.8±1.6 | 0.953 | 0.786 | 0.856 | nan | 0.898 | ❌ Below Targets |
| evidence | Unknown | 9.0±1.7 | 12.6±3.0 | nan | nan | nan | nan | nan | ❌ Below Targets |
| evidence | Unknown | 11.0 | 41.0 | nan | nan | nan | nan | nan | ❌ Below Targets |
| results | Unknown | 12.0 | 40.0 | 0.832 | 0.943 | 0.884 | 0.818 | 0.913 | ❌ Below Targets |
| results | Unknown | 23.0 | 21.0 | 0.899 | 0.890 | 0.895 | 0.846 | 0.907 | ❌ Below Targets |
| results | Unknown | 48.0 | 6.0 | 0.964 | 0.771 | 0.857 | 0.811 | 0.913 | ❌ Below Targets |
| results | Unknown | 48.0 | 6.0 | N/A | N/A | N/A | N/A | N/A | ❌ Below Targets |
| results | Unknown | 48.0 | 6.0 | N/A | N/A | N/A | N/A | N/A | ❌ Below Targets |
| Swin_baseline | Swin | 53.0 | 2.0 | 0.987 | 0.748 | 0.851 | 0.807 | 0.907 | ❌ Below Targets |

## Gap Analysis vs Targets


**Current Best Performance:**
- **FN:** 2.8 (target: <10) → ✅ Gap: 0.0
- **FP:** 7.8 (target: <10) → ✅ Gap: 0.0
- **Recall:** 0.933 (target: ≥0.92) → ✅ Gap: 0.000
- **Accuracy:** nan (target: ≥0.92) → ❌ Gap: 0.000


## Evaluation Protocol Validation

✅ **Nested CV Implementation:**
- Base models trained only on train folds
- OOF predictions generated correctly
- Meta-learner trained only on OOF within inner loops, tested on outer fold

✅ **No Data Leakage:**
- Patient-level splitting confirmed
- No duplicate slides/tiles across folds
- Preprocessing fitted per-fold


## Recommendations for Adding ResNet50-2D & DenseNet

### 1. Expected Value

**ResNet50-2D:**
- **Pros:** 
  - Different architecture (2D vs 3D) provides diversity
  - Faster inference than 3D models
  - Can capture slice-level patterns that 3D models might miss
  - Pre-trained ImageNet weights available
  
- **Cons:**
  - Loses 3D spatial context
  - May require careful slice selection or aggregation

**DenseNet121:**
- **Pros:**
  - Efficient feature reuse (parameter efficient)
  - Good for medical imaging tasks
  - Different inductive bias than ResNet
  - Can complement existing models
  
- **Cons:**
  - Similar architecture family to ResNet (less diversity)
  - May require careful calibration

### 2. Integration Plan

**Architecture:**
```
Tile-level embeddings (ResNet50-2D/DenseNet121) 
  → MIL pooling (attention/mean/max) 
  → Bag-level prediction
  → Calibration
  → Meta-learner input
```

**Implementation Steps:**
1. **Feature Extraction:**
   - Use pre-trained ResNet50-2D/DenseNet121 (ImageNet)
   - Extract embeddings from 2D slices (axial, coronal, sagittal)
   - Option: Fine-tune on medical imaging dataset (if available)
   - Option: Freeze backbone, train only MIL head

2. **MIL Aggregation:**
   - Attention-based pooling (like current DualStreamMIL)
   - Multi-view aggregation (combine axial/coronal/sagittal)
   - Bag size: 32-64 slices (similar to current MIL)

3. **Calibration:**
   - Platt scaling or isotonic regression
   - Per-fold calibration (nested CV)

4. **Meta-Learner Integration:**
   - Add new features: `hgg_prob_resnet2d`, `hgg_prob_densenet`
   - Retrain Logistic Regression or XGBoost meta-learner
   - Use nested CV for evaluation

### 3. Expected Gains and Risks

**Expected Gains:**
- **Diversity:** 2D models may catch patterns 3D models miss
- **FN Reduction:** Additional models may reduce false negatives by 1-3
- **Robustness:** Ensemble diversity improves generalization

**Risks:**
- **Overfitting:** Adding models increases capacity → need more data
- **Computation:** 2 additional models × 5 folds = 10 more training runs
- **Calibration:** Need to ensure probabilities are well-calibrated
- **Diminishing Returns:** Current ensemble already strong

**Estimated Improvement:**
- **Conservative:** FN reduction by 1-2, FP increase by 1-2
- **Optimistic:** FN reduction by 2-4, FP increase by 0-2
- **Accuracy/Recall:** +0.5-2% improvement possible

### 4. Recommended Experiments

**Experiment 1: ResNet50-2D + Attention MIL (Priority: High)**
- Use pre-trained ResNet50-2D (frozen or fine-tuned)
- Attention-based MIL pooling
- Calibrate probabilities
- Add to meta-learner
- **Expected:** FN reduction by 1-2, minimal FP increase

**Experiment 2: DenseNet121 + Multi-View Aggregation (Priority: Medium)**
- Use pre-trained DenseNet121
- Multi-view aggregation (axial + coronal + sagittal)
- Calibrate probabilities
- Add to meta-learner
- **Expected:** FN reduction by 1, FP increase by 0-1

**Experiment 3: Stacking with Cost-Sensitive Thresholding (Priority: High)**
- Add both ResNet50-2D and DenseNet121
- Use cost-sensitive thresholding to optimize FN/FP trade-off
- Class-weight tuning in meta-learner
- **Expected:** FN < 10, FP < 10, Recall ≥ 92%

### 5. Final Recommendation

**YES: Adding ResNet50-2D and DenseNet likely helps, but with caveats:**

✅ **Proceed if:**
- You have computational resources for 2 additional models
- You can ensure proper nested CV evaluation
- You're willing to tune thresholds carefully

⚠️ **Consider alternatives first:**
- Fine-tune existing models more carefully
- Improve calibration of current ensemble
- Use cost-sensitive learning with current models
- Add non-DL features (already done in some experiments)

🎯 **Best Next Steps:**
1. **Start with ResNet50-2D only** (lower risk, faster)
2. **Use attention-based MIL** (proven effective)
3. **Calibrate carefully** (critical for ensemble)
4. **Cost-sensitive thresholding** (to hit FN/FP targets)
5. **Evaluate with nested CV** (maintain rigor)

**Expected Outcome:**
- **FN:** 2-4 (currently best: ~2.8-4.2) → **Target: <10** ✅
- **FP:** 6-9 (currently best: ~6.4-7.8) → **Target: <10** ✅
- **Recall:** 0.93-0.95 (currently best: ~0.90-0.93) → **Target: ≥0.92** ✅
- **Accuracy:** 0.85-0.90 (currently best: ~0.81-0.85) → **Target: ≥0.92** ⚠️

**Accuracy may need additional work** (threshold tuning, better calibration, or more data).

---

## Detailed Results


### 1. Mil-Prob_baseline

- **Source:** Unknown
- **Timestamp:** Unknown
- **Model:** Mil-Prob
- **FN:** 0
- **FP:** 75
- **Precision:** 0.7368421052631579
- **Recall:** 1.0
- **F1:** 0.8484848484848485
- **Accuracy:** 0.7368421052631579
- **ROC-AUC:** 0.5956190476190476


### 2. Resnet_baseline

- **Source:** Unknown
- **Timestamp:** Unknown
- **Model:** Resnet
- **FN:** 1
- **FP:** 68
- **Precision:** 0.7545126353790613
- **Recall:** 0.9952380952380953
- **F1:** 0.8583162217659137
- **Accuracy:** 0.7578947368421053
- **ROC-AUC:** 0.6103492063492063


### 3. nested_cv_meta_features

- **Source:** /workspace/brain_tumor_project/ensemble/results/nested_cv_meta_features/meta_features_results_20260209_005859.json
- **Timestamp:** 2026-02-09T00:58:59
- **Model:** LogisticRegression_Enhanced
- **FN:** 2.8
- **FP:** 7.8
- **Precision:** 0.8362222772292064
- **Recall:** 0.9333333333333332
- **F1:** 0.8811696858726915
- **Accuracy:** nan
- **ROC-AUC:** nan


### 4. Simple_Ensemble_Average

- **Source:** Unknown
- **Timestamp:** Unknown
- **Model:** Simple Ensemble (Average)
- **FN:** 4
- **FP:** 67
- **Precision:** 0.7545787545787546
- **Recall:** 0.9809523809523809
- **F1:** 0.8530020703933747
- **Accuracy:** 0.7508771929824561
- **ROC-AUC:** 0.8962539682539683


### 5. hybrid_safety_net

- **Source:** /workspace/brain_tumor_project/ensemble/results/hybrid_safety_net/uncertain_meta_decision_results.json
- **Timestamp:** 2026-02-10T19:18:11
- **Model:** Unknown
- **FN:** 4.8
- **FP:** 2.0
- **Precision:** 0.7731601731601733
- **Recall:** 0.5267857142857142
- **F1:** 0.6057952069716774
- **Accuracy:** nan
- **ROC-AUC:** 0.45282738095238095


### 6. meta_decision

- **Source:** /workspace/brain_tumor_project/ensemble/results/meta_decision/meta_decision_results.json
- **Timestamp:** 2026-02-10T18:53:18
- **Model:** Unknown
- **FN:** 9.0
- **FP:** 1.8
- **Precision:** 0.9532604226971063
- **Recall:** 0.7857142857142858
- **F1:** 0.8557383743688272
- **Accuracy:** nan
- **ROC-AUC:** 0.8977777777777778


### 7. evidence

- **Source:** /workspace/brain_tumor_project/ensemble/results/forensic_audit_xgboost/evidence/nested_eval_summary.json
- **Timestamp:** 2026-02-08T23:02:27
- **Model:** Unknown
- **FN:** 9.0
- **FP:** 12.6
- **Precision:** nan
- **Recall:** nan
- **F1:** nan
- **Accuracy:** nan
- **ROC-AUC:** nan


### 8. evidence

- **Source:** /workspace/brain_tumor_project/ensemble/results/forensic_audit_xgboost/evidence/baseline_repro_check.json
- **Timestamp:** 2026-02-08T23:02:25
- **Model:** Unknown
- **FN:** 11
- **FP:** 41
- **Precision:** nan
- **Recall:** nan
- **F1:** nan
- **Accuracy:** nan
- **ROC-AUC:** nan


### 9. results

- **Source:** /workspace/brain_tumor_project/ensemble/results/eval_threshold_0_19.json
- **Timestamp:** 2026-02-06T18:58:04
- **Model:** Unknown
- **FN:** 12
- **FP:** 40
- **Precision:** 0.8319327731092437
- **Recall:** 0.9428571428571428
- **F1:** 0.8839285714285714
- **Accuracy:** 0.8175438596491228
- **ROC-AUC:** 0.9126349206349207


### 10. results

- **Source:** /workspace/brain_tumor_project/ensemble/results/eval_threshold_0_22.json
- **Timestamp:** 2026-02-09T23:31:36
- **Model:** Unknown
- **FN:** 23
- **FP:** 21
- **Precision:** 0.8990384615384616
- **Recall:** 0.8904761904761904
- **F1:** 0.8947368421052632
- **Accuracy:** 0.8456140350877193
- **ROC-AUC:** 0.9074285714285714

