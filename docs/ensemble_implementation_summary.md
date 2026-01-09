# Ensemble Stacking: Implementation Summary

## What Was Implemented

A complete data preparation pipeline for ensemble stacking has been implemented. The implementation focuses on organizing out-of-fold (OOF) predictions from three base models into a format suitable for meta-learner training.

## Created Files

1. **`scripts/ensemble/prepare_oof_predictions.py`**: Main script for preparing OOF predictions
2. **`ensemble/README.md`**: Documentation for the ensemble directory
3. **`docs/ensemble_implementation_plan.md`**: Detailed implementation plan
4. **`ensemble/oof_predictions/`**: Directory for output CSV files (created automatically)

## Key Features

### Automatic Latest Run Selection
- For each model and each fold, the script automatically identifies the most recent run based on directory modification time
- Older runs within the same fold are ignored (no averaging)

### OOF Prediction Extraction
- Loads `val_probs.npy` from each selected run
- Extracts HGG probabilities (second column)
- Matches predictions with patient IDs from validation split CSV files

### Data Verification
- Uniqueness: Ensures each patient appears exactly once
- Completeness: Verifies all 5 folds are represented
- Probability range: Validates all probabilities are in [0, 1]
- Label consistency: Verifies labels match expected values

### Output Format
Each model produces a CSV file with columns:
- `patient_id`: Patient identifier
- `fold`: Fold number (0-4)
- `hgg_prob`: HGG class probability (0-1)
- `label`: Ground truth label (0=LGG, 1=HGG)

## Critical Assumption

**Prediction-Order Assumption**: The script assumes that validation predictions (`val_probs.npy`) are stored in the same order as patients appear in the validation split CSV file (`splits/fold_{k}_val.csv`). This assumption is based on standard PyTorch DataLoader behavior, which loads samples in the order specified by the dataset.

This assumption should be verified manually by comparing:
1. The number of predictions in `val_probs.npy`
2. The number of patients in the corresponding validation split CSV
3. The order of predictions (if patient IDs were logged during validation)

## Directory Structure

```
ensemble/
├── oof_predictions/          # Output CSV files (created by script)
│   ├── resnet50_3d_oof.csv
│   ├── swinunetr_3d_oof.csv
│   └── dualstream_mil_3d_oof.csv
└── README.md                 # Documentation

scripts/ensemble/
└── prepare_oof_predictions.py  # Main implementation script
```

## Usage

To prepare OOF predictions for all models:

```bash
python scripts/ensemble/prepare_oof_predictions.py
```

The script will:
1. Process each model (ResNet50-3D, SwinUNETR-3D, DualStreamMIL-3D)
2. For each fold, select the latest run
3. Extract and match predictions with patient IDs
4. Perform verification checks
5. Save results to `ensemble/oof_predictions/`

## Model Configuration

The script uses the following model-to-directory mappings:
- `ResNet50-3D` → `results/ResNet50-3D/runs/`
- `SwinUNETR-3D` → `results/SwinUNETR-3D/runs/`
- `DualStreamMIL-3D` → `results/DualStreamMIL-3D/runs/`

All models are expected to follow the same structure:
```
results/{ModelName}/runs/fold_{k}/run_{timestamp}/
├── predictions/
│   ├── val_probs.npy    # Shape: (n_samples, 2) with [LGG_prob, HGG_prob]
│   ├── val_labels.npy
│   └── val_preds.npy
└── metrics/
    └── metrics.json
```

## Next Steps (Not Implemented)

The following are out of scope for this implementation:

1. **Meta-Learner Training**: Training a logistic regression meta-learner on the OOF predictions
2. **Ensemble Evaluation**: Computing ensemble performance metrics
3. **Inference Pipeline**: Creating inference pipeline for test data

These should be implemented as separate scripts after OOF predictions are verified.

## Error Handling

The script includes robust error handling:
- Clear error messages if required files are missing
- Warnings if multiple runs exist (proceeds with latest)
- Detailed logging of which run was selected for each fold
- Verification checks that fail gracefully with informative messages

## Verification Checklist

Before proceeding to meta-learner training, verify:

- [ ] All three OOF CSV files are generated
- [ ] Each file has the expected number of predictions (should match total validation set size)
- [ ] No duplicate patient IDs within each file
- [ ] All probabilities are in valid range [0, 1]
- [ ] Patient IDs match between OOF files (for later merging)
- [ ] Fold assignments are correct (each patient from fold k should have fold=k in the OOF file)

## Potential Issues

1. **Missing Predictions**: If a fold is missing prediction files, the script will error. Check that all required runs completed successfully.

2. **Order Mismatch**: If the prediction-order assumption is violated, patient IDs may be incorrectly matched. This would require manual verification or modification of the script.

3. **Multiple Models Different Structures**: If any model has a different file structure, the script will need modification. Currently all three models follow the same structure.

## Testing Recommendation

Before using OOF predictions for meta-learner training:

1. Run the script and verify it completes without errors
2. Manually spot-check a few patients:
   - Load a validation split CSV
   - Check that the corresponding prediction in the OOF file has the correct fold assignment
   - Verify the patient_id matches
3. Compare fold sizes:
   - Count patients per fold in validation splits
   - Verify counts match in OOF files

