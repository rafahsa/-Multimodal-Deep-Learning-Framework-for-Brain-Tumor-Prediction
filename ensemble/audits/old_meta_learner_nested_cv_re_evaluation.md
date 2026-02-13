# Old Meta-Learner Nested CV Re-Evaluation Audit

**Generated:** 2026-02-13T02:08:06.837052

## Executive Summary

This audit report documents the rigorous re-evaluation of an earlier ensemble meta-learner using the **exact same nested cross-validation protocol** as the final model.

**Decision:** B

**Conclusion:** The old meta-learner does not outperform the final model when evaluated fairly and should remain excluded.

---

## STEP 1: Compatibility Inspection

### Old Meta-Learner Details

- **Path:** `/workspace/brain_tumor_project/ensemble/models/meta_learner_logistic_regression.joblib`
- **Type:** LogisticRegression
- **Number of Features:** 3
- **Feature Order Assumption:** `['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']`

### Coefficients

```
ResNet50-3D (hgg_prob_resnet): 0.561032
SwinUNETR-3D (hgg_prob_swin): 4.137673
DualStreamMIL-3D (hgg_prob_mil): 0.092570
Intercept: -2.120502
```

### Compatibility Status

✓ **COMPATIBLE**: Old model has 3 features matching standard feature order.

---

## STEP 2: Evaluation Inputs

### Data Sources

- **OOF Predictions:** `/workspace/brain_tumor_project/ensemble/oof_predictions/merged_oof_predictions.csv`
- **Total Patients:** 285
- **Feature Columns:** ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']
- **Target Column:** label

### Evaluation Protocol

- **Outer CV Folds:** 5
- **Random Seed:** 42 (matching final model)
- **Calibration Fraction:** 0.7
- **Threshold Sweep:** 0.05 to 0.95 (step: 0.01)
- **Cost Function:** 2*FN + FP

---

## STEP 3: Nested CV Results (Old Model)

### Per-Fold Results


**Fold 1:**
- Outer-train size: 228
- Outer-test size: 57
- Selected threshold: 0.2900
- TN: 9, FP: 6, FN: 4, TP: 38
- Precision: 0.8636
- Recall: 0.9048
- F1: 0.8837
- AUC: 0.8857
- Cost: 14.0


**Fold 2:**
- Outer-train size: 228
- Outer-test size: 57
- Selected threshold: 0.3800
- TN: 9, FP: 6, FN: 2, TP: 40
- Precision: 0.8696
- Recall: 0.9524
- F1: 0.9091
- AUC: 0.9508
- Cost: 10.0


**Fold 3:**
- Outer-train size: 228
- Outer-test size: 57
- Selected threshold: 0.3300
- TN: 5, FP: 10, FN: 2, TP: 40
- Precision: 0.8000
- Recall: 0.9524
- F1: 0.8696
- AUC: 0.9206
- Cost: 14.0


**Fold 4:**
- Outer-train size: 228
- Outer-test size: 57
- Selected threshold: 0.4700
- TN: 13, FP: 2, FN: 6, TP: 36
- Precision: 0.9474
- Recall: 0.8571
- F1: 0.9000
- AUC: 0.9460
- Cost: 14.0


**Fold 5:**
- Outer-train size: 228
- Outer-test size: 57
- Selected threshold: 0.3600
- TN: 7, FP: 8, FN: 6, TP: 36
- Precision: 0.8182
- Recall: 0.8571
- F1: 0.8372
- AUC: 0.8286
- Cost: 20.0


### Aggregated Metrics (Old Model)

- **Mean FN:** 4.00 ± 1.79 (range: [2, 6])
- **Mean FP:** 6.40 ± 2.65
- **Mean Recall:** 0.9048 ± 0.0426
- **Mean Precision:** 0.8598 ± 0.0512
- **Mean F1:** 0.8799 ± 0.0253
- **Mean AUC:** 0.9063 ± 0.0452
- **Mean Cost:** 14.40 ± 3.20

### Global Confusion Matrix (Old Model)

Summed across all 5 folds:

```
        Predicted
        LGG    HGG
True LGG    43    32
True HGG    20   190
```

---

## STEP 4: Comparison with Final Model


### Final Model Results (Nested CV with Meta-Features)

- **Mean FN:** 2.80 ± 2.14 (range: [0, 6])
- **Mean FP:** 7.80 ± 2.79
- **Mean Recall:** 0.9333 ± 0.0508
- **Mean Precision:** 0.8362 ± 0.0530
- **Mean F1:** 0.8812 ± 0.0431

### Comparison Table

| Metric | Old Model | Final Model | Difference (Old - Final) |
|--------|-----------|------------|--------------------------|
| Mean FN | 4.00 ± 1.79 | 2.80 ± 2.14 | +1.20 |
| Mean FP | 6.40 ± 2.65 | 7.80 ± 2.79 | -1.40 |
| Mean Recall | 0.9048 ± 0.0426 | 0.9333 ± 0.0508 | -0.0286 |
| Mean Precision | 0.8598 ± 0.0512 | 0.8362 ± 0.0530 | +0.0235 |
| Mean F1 | 0.8799 ± 0.0253 | 0.8812 ± 0.0431 | -0.0013 |

### Statistical Interpretation

- **FN:** Old model has **higher** mean FN (4.00 vs 2.80), which is **worse** for clinical safety.
- **Recall:** Old model has **lower** mean recall (0.9048 vs 0.9333), which is **worse**.
- **F1:** Old model has **lower** mean F1 (0.8799 vs 0.8812), which is **worse**.


---

## STEP 5: Final Decision

**Decision Code:** B

**Conclusion:** The old meta-learner does not outperform the final model when evaluated fairly and should remain excluded.

### Justification


The old meta-learner does not demonstrate superior performance:
- **FN (Clinical Safety):** 4.00 vs 2.80 - Higher (worse)
- **Recall:** 0.9048 vs 0.9333 - Lower (worse)
- **F1 (Balanced Performance):** 0.8799 vs 0.8812 - Lower (worse)

The final model (with enhanced meta-features) maintains its position as the superior ensemble configuration.


---

## Technical Details

### Files Used

- **Old Meta-Learner:** `/workspace/brain_tumor_project/ensemble/models/meta_learner_logistic_regression.joblib`
- **OOF Predictions:** `/workspace/brain_tumor_project/ensemble/oof_predictions/merged_oof_predictions.csv`
- **Final Model Results:** `/workspace/brain_tumor_project/ensemble/results/nested_cv_meta_features/meta_features_results_20260209_005859.json`

### Evaluation Constraints

✓ Same nested CV structure (5 outer folds)
✓ Same random seed (42)
✓ Same fold-specific threshold selection
✓ Same cost function (2*FN + FP)
✓ Same calibration protocol (Platt scaling)
✓ No data leakage (outer-test never seen during training/calibration/threshold selection)
✓ No base model retraining

### Reproducibility

All results can be reproduced by running:
```bash
python scripts/analysis/reevaluate_old_meta_learner.py
```

---

## Appendix: Complete Per-Fold Results


### Fold 1 Details

```json
{
  "fold": 0,
  "outer_train_size": 228,
  "outer_test_size": 57,
  "selected_threshold": 0.29000000000000004,
  "threshold": 0.29000000000000004,
  "tn": 9,
  "fp": 6,
  "fn": 4,
  "tp": 38,
  "precision": 0.8636363636363636,
  "recall": 0.9047619047619048,
  "f1": 0.8837209302325582,
  "accuracy": 0.8245614035087719,
  "specificity": 0.6,
  "cost": 14.0,
  "auc": 0.8857142857142858
}
```


### Fold 2 Details

```json
{
  "fold": 1,
  "outer_train_size": 228,
  "outer_test_size": 57,
  "selected_threshold": 0.38000000000000006,
  "threshold": 0.38000000000000006,
  "tn": 9,
  "fp": 6,
  "fn": 2,
  "tp": 40,
  "precision": 0.8695652173913043,
  "recall": 0.9523809523809523,
  "f1": 0.9090909090909091,
  "accuracy": 0.8596491228070176,
  "specificity": 0.6,
  "cost": 10.0,
  "auc": 0.9507936507936507
}
```


### Fold 3 Details

```json
{
  "fold": 2,
  "outer_train_size": 228,
  "outer_test_size": 57,
  "selected_threshold": 0.33,
  "threshold": 0.33,
  "tn": 5,
  "fp": 10,
  "fn": 2,
  "tp": 40,
  "precision": 0.8,
  "recall": 0.9523809523809523,
  "f1": 0.8695652173913043,
  "accuracy": 0.7894736842105263,
  "specificity": 0.3333333333333333,
  "cost": 14.0,
  "auc": 0.9206349206349206
}
```


### Fold 4 Details

```json
{
  "fold": 3,
  "outer_train_size": 228,
  "outer_test_size": 57,
  "selected_threshold": 0.4700000000000001,
  "threshold": 0.4700000000000001,
  "tn": 13,
  "fp": 2,
  "fn": 6,
  "tp": 36,
  "precision": 0.9473684210526315,
  "recall": 0.8571428571428571,
  "f1": 0.9,
  "accuracy": 0.8596491228070176,
  "specificity": 0.8666666666666667,
  "cost": 14.0,
  "auc": 0.946031746031746
}
```


### Fold 5 Details

```json
{
  "fold": 4,
  "outer_train_size": 228,
  "outer_test_size": 57,
  "selected_threshold": 0.36000000000000004,
  "threshold": 0.36000000000000004,
  "tn": 7,
  "fp": 8,
  "fn": 6,
  "tp": 36,
  "precision": 0.8181818181818182,
  "recall": 0.8571428571428571,
  "f1": 0.8372093023255814,
  "accuracy": 0.7543859649122807,
  "specificity": 0.4666666666666667,
  "cost": 20.0,
  "auc": 0.8285714285714285
}
```

