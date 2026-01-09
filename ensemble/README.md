# Ensemble Stacking: Out-of-Fold Predictions

This directory contains out-of-fold (OOF) predictions prepared for ensemble meta-learner training.

## Directory Structure

```
ensemble/
├── oof_predictions/          # OOF prediction CSV files (one per model)
│   ├── resnet50_3d_oof.csv
│   ├── swinunetr_3d_oof.csv
│   └── dualstream_mil_3d_oof.csv
└── README.md                 # This file
```

## File Format

Each OOF prediction CSV file contains the following columns:

- `patient_id`: Patient identifier (string, e.g., "Brats18_2013_10_1")
- `fold`: Fold number (0-4) from which this prediction came
- `hgg_prob`: HGG class probability (float, 0-1) - extracted from the second column of val_probs
- `label`: Ground truth label (int, 0=LGG, 1=HGG)

## Data Preparation Process

1. **Latest Run Selection**: For each model and each fold, the most recent run (by modification time) is automatically selected.

2. **Prediction Extraction**: HGG probabilities are extracted from `predictions/val_probs.npy` files.

3. **Patient ID Matching**: Predictions are matched with patient IDs from validation split CSV files (`splits/fold_{k}_val.csv`). The matching assumes predictions are in the same order as patients in the CSV file.

4. **Aggregation**: Predictions from all 5 folds are aggregated into a single CSV file per model.

5. **Verification**: Multiple checks are performed:
   - Uniqueness: Each patient appears exactly once
   - Completeness: All folds are represented
   - Probability range: All probabilities in [0, 1]
   - Label values: All labels are 0 or 1

## Usage

To regenerate OOF predictions:

```bash
python scripts/ensemble/prepare_oof_predictions.py
```

## Next Steps

✅ **Completed**:
1. ✅ Merged OOF predictions from all three models into a single CSV file
2. ✅ Trained logistic regression meta-learner on merged OOF predictions
3. ✅ Evaluated ensemble performance and saved metrics

**Remaining**:
- Prepare ensemble inference pipeline for test data
- Evaluate ensemble on held-out test set (if available)

## Important Notes

- **No Data Leakage**: Each prediction comes from the fold where that patient was in the validation set.
- **Latest Run Only**: Only the most recent run per fold is used (no averaging across runs).
- **Patient-Level**: Predictions are at the patient level, not instance/slice level.

## Meta-Learner Training

The Logistic Regression meta-learner has been trained and saved:

- **Model File**: `ensemble/models/meta_learner_logistic_regression.joblib`
- **Metrics File**: `ensemble/results/meta_learner_metrics.json`
- **Training Script**: `scripts/ensemble/train_meta_learner.py`

### Meta-Learner Performance

- **Accuracy**: 0.8105 (81.05%)
- **F1-Score**: 0.8571 (85.71%)
- **AUC-ROC**: 0.9126 (91.26%)

### Feature Importance

The meta-learner assigns the following weights:
1. **SwinUNETR-3D** (coefficient: 4.06) - Most important
2. **DualStreamMIL-3D** (coefficient: 0.89)
3. **ResNet50-3D** (coefficient: 0.54) - Least important

See `docs/ensemble_meta_learner_training_report.md` for detailed results.

