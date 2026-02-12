# OOF Provenance Check

**Merged OOF File**: ensemble/oof_predictions/merged_oof_predictions.csv

**Fold Column Exists**: 1

## Scripts Found

- scripts/ensemble/verify_and_merge_oof.py
- scripts/ensemble/train_meta_learner.py
- scripts/ensemble/calibrate_and_sweep_thresholds.py

## OOF Validation

```json
{
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
```
