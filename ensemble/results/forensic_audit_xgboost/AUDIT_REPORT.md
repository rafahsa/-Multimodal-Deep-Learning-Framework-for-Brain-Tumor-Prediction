# Forensic Audit Report: XGBoost Meta-Learner

**Audit Date**: 2026-02-08T23:02:25.587147

## Executive Summary

**Overall Verdict**: **NOT VERIFIED**

**Reason**: Failed checks: B4_leakage_tests, C6_nested_evaluation

---

## Check Summary

| Check | Status | Notes |
|-------|--------|-------|
| A1_data_integrity | ✅ PASSED | {'n_rows': 285, 'n_features': 3, 'class_distributi... |
| A2_label_sanity | ✅ PASSED | {'threshold': 0.35, 'tn': 34, 'fp': 41, 'fn': 11, ... |
| B3_oof_provenance | ✅ PASSED | {'merged_file': 'ensemble/oof_predictions/merged_o... |
| B4_leakage_tests | ❌ FAILED | {'trivial_index_classifier': {'mean_accuracy': 0.8... |
| C5_calibration_equivalence | ❌ INCONCLUSIVE | {'sklearn_method_failed': "The 'cv' parameter of C... |
| C6_nested_evaluation | ❌ FAILED | {'n_seeds': 5, 'fn_mean': 9.0, 'fn_std': 1.6733200... |
| D7_margin_analysis | ✅ PASSED | {'n_fp_cases': 1, 'n_borderline_hgg': 5}... |
| D8_overfitting_sensitivity | ✅ PASSED | {'n_combinations': 5, 'results': [{'max_depth': 2,... |

---

## Nested Evaluation Results (CRITICAL)

This is the most important check: performance on a truly held-out test set.

**Tested across 5 random seeds**

| Metric | Mean ± Std | Range |
|--------|------------|-------|
| FN | 9.00 ± 1.67 | [7, 12] |
| FP | 12.60 ± 3.01 | - |
| Cost | 30.60 ± 5.39 | - |

---

## Detailed Findings

### A1_data_integrity

**Status**: PASSED

```json
{
  "n_rows": 285,
  "n_features": 3,
  "class_distribution": {
    "1": 210,
    "0": 75
  },
  "feature_stats": {
    "hgg_prob_resnet": {
      "min": 0.16521406,
      "max": 0.99989307,
      "mean": 0.8883786005263158,
      "std": 0.11948225680537108
    },
    "hgg_prob_swin": {
      "min": 0.0023331007,
      "max": 0.99999225,
      "mean": 0.5798635070684212,
      "std": 0.44056338990687754
    },
    "hgg_prob_mil": {
      "min": 0.1113,
      "max": 0.9497,
      "mean": 0.484701052631579,
      "std": 0.15300543658853638
    }
  },
  "has_nan": {
    "hgg_prob_resnet": 0,
    "hgg_prob_swin": 0,
    "hgg_prob_mil": 0
  },
  "has_inf": {
    "hgg_prob_resnet": 0,
    "hgg_prob_swin": 0,
    "hgg_prob_mil": 0
  },
  "duplicate_rows": 0,
  "duplicate_features": 0,
  "patient_id_duplicates": 0,
  "fold_distribution": {
    "3": 57,
    "0": 57,
    "4": 57,
    "2": 57,
    "1": 57
  },
  "samples_per_fold": {
    "min": 57,
    "max": 57,
    "mean": 57.0
  }
}
```

### A2_label_sanity

**Status**: PASSED

```json
{
  "threshold": 0.35,
  "tn": 34,
  "fp": 41,
  "fn": 11,
  "tp": 199,
  "expected_tn": 34,
  "expected_fp": 41,
  "expected_fn": 11,
  "expected_tp": 199,
  "matches_expected": 1
}
```

### B3_oof_provenance

**Status**: PASSED

```json
{
  "merged_file": "ensemble/oof_predictions/merged_oof_predictions.csv",
  "file_exists": 1,
  "scripts_found": [
    "scripts/ensemble/verify_and_merge_oof.py",
    "scripts/ensemble/train_meta_learner.py",
    "scripts/ensemble/calibrate_and_sweep_thresholds.py"
  ],
  "fold_column_exists": 1,
  "oof_validation": {
    "has_fold_column": 1,
    "fold_distribution": {
      "3": 57,
      "0": 57,
      "4": 57,
      "2": 57,
      "1": 57
    },
    "all_samples_have_fold": 1,
    "fold_range": [
      0,
      4
    ]
  }
}
```

### B4_leakage_tests

**Status**: FAILED

```json
{
  "trivial_index_classifier": {
    "mean_accuracy": 0.8421052631578947,
    "std_accuracy": 0.1785685705263084,
    "expected_chance": 0.5,
    "suspicious": 1
  },
  "xgboost_shuffled_labels": {
    "mean_accuracy": 0.6912280701754385,
    "std_accuracy": 0.0377906302255053,
    "expected_chance": 0.7368421052631579,
    "suspicious": 0
  }
}
```

### C5_calibration_equivalence

**Status**: INCONCLUSIVE

```json
{
  "sklearn_method_failed": "The 'cv' parameter of CalibratedClassifierCV must be an int in the range [2, inf), an object implementing 'split' and 'get_n_splits', an iterable or None. Got 'prefit' instead.",
  "note": "Using clean implementation only"
}
```

### C6_nested_evaluation

**Status**: FAILED

```json
{
  "n_seeds": 5,
  "fn_mean": 9.0,
  "fn_std": 1.6733200530681511,
  "fn_min": 7,
  "fn_max": 12,
  "fp_mean": 12.6,
  "fp_std": 3.006659275674582,
  "cost_mean": 30.6,
  "cost_std": 5.388877434122992,
  "results": [
    {
      "seed": 21,
      "selected_threshold": 0.26000000000000006,
      "test_set_size": 86,
      "tn": 6,
      "fp": 17,
      "fn": 9,
      "tp": 54,
      "precision": 0.7605633802816901,
      "recall": 0.8571428571428571,
      "f1": 0.8059701492537313,
      "accuracy": 0.6976744186046512,
      "cost": 35.0
    },
    {
      "seed": 42,
      "selected_threshold": 0.5500000000000002,
      "test_set_size": 86,
      "tn": 13,
      "fp": 10,
      "fn": 8,
      "tp": 55,
      "precision": 0.8461538461538461,
      "recall": 0.873015873015873,
      "f1": 0.859375,
      "accuracy": 0.7906976744186046,
      "cost": 26.0
    },
    {
      "seed": 77,
      "selected_threshold": 0.31000000000000005,
      "test_set_size": 86,
      "tn": 14,
      "fp": 9,
      "fn": 9,
      "tp": 54,
      "precision": 0.8571428571428571,
      "recall": 0.8571428571428571,
      "f1": 0.8571428571428571,
      "accuracy": 0.7906976744186046,
      "cost": 27.0
    },
    {
      "seed": 123,
      "selected_threshold": 0.18000000000000005,
      "test_set_size": 86,
      "tn": 8,
      "fp": 15,
      "fn": 12,
      "tp": 51,
      "precision": 0.7727272727272727,
      "recall": 0.8095238095238095,
      "f1": 0.7906976744186046,
      "accuracy": 0.686046511627907,
      "cost": 39.0
    },
    {
      "seed": 202,
      "selected_threshold": 0.31000000000000005,
      "test_set_size": 86,
      "tn": 11,
      "fp": 12,
      "fn": 7,
      "tp": 56,
      "precision": 0.8235294117647058,
      "recall": 0.8888888888888888,
      "f1": 0.8549618320610687,
      "accuracy": 0.7790697674418605,
      "cost": 26.0
    }
  ]
}
```

### D7_margin_analysis

**Status**: PASSED

```json
{
  "n_fp_cases": 1,
  "n_borderline_hgg": 5
}
```

### D8_overfitting_sensitivity

**Status**: PASSED

```json
{
  "n_combinations": 5,
  "results": [
    {
      "max_depth": 2,
      "n_estimators": 50,
      "min_child_weight": 1,
      "subsample": 1.0,
      "colsample_bytree": 1.0,
      "selected_threshold": 0.4000000000000001,
      "fn": 6,
      "fp": 13,
      "cost": 25.0,
      "recall": 0.9047619047619048,
      "precision": 0.8142857142857143
    },
    {
      "max_depth": 3,
      "n_estimators": 50,
      "min_child_weight": 1,
      "subsample": 1.0,
      "colsample_bytree": 1.0,
      "selected_threshold": 0.28,
      "fn": 4,
      "fp": 16,
      "cost": 24.0,
      "recall": 0.9365079365079365,
      "precision": 0.7866666666666666
    },
    {
      "max_depth": 4,
      "n_estimators": 50,
      "min_child_weight": 1,
      "subsample": 1.0,
      "colsample_bytree": 1.0,
      "selected_threshold": 0.4100000000000001,
      "fn": 9,
      "fp": 13,
      "cost": 31.0,
      "recall": 0.8571428571428571,
      "precision": 0.8059701492537313
    },
    {
      "max_depth": 4,
      "n_estimators": 100,
      "min_child_weight": 1,
      "subsample": 1.0,
      "colsample_bytree": 1.0,
      "selected_threshold": 0.5500000000000002,
      "fn": 8,
      "fp": 10,
      "cost": 26.0,
      "recall": 0.873015873015873,
      "precision": 0.8461538461538461
    },
    {
      "max_depth": 4,
      "n_estimators": 100,
      "min_child_weight": 5,
      "subsample": 0.8,
      "colsample_bytree": 0.8,
      "selected_threshold": 0.4600000000000001,
      "fn": 5,
      "fp": 7,
      "cost": 17.0,
      "recall": 0.9206349206349206,
      "precision": 0.8923076923076924
    }
  ]
}
```

---

## Conclusion

**Verdict**: NOT VERIFIED

Failed checks: B4_leakage_tests, C6_nested_evaluation

❌ XGBoost performance is **NOT VERIFIED**. Do not claim adoption without further investigation.