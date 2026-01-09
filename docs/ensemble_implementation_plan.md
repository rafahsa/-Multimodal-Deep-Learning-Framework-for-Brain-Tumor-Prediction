# Ensemble Stacking: Data Preparation Implementation Plan

## Overview

This document outlines the step-by-step implementation plan for preparing out-of-fold (OOF) predictions from three base models (ResNet50-3D, SwinUNETR-3D, Dual-Stream MIL-3D) for ensemble stacking. The goal is to create leakage-free OOF prediction tables that can be used to train a meta-learner.

## Assumptions

1. **Prediction-Order Assumption**: Validation predictions (`val_probs`, `val_labels`, `val_preds`) are stored in the same order as patients appear in the validation split CSV file (`splits/fold_{k}_val.csv`). This assumption is based on standard PyTorch DataLoader behavior, which loads samples in the order specified by the dataset.

2. **File Location Assumption**: All models follow a consistent results directory structure:
   - `results/{ModelName}/runs/fold_{k}/run_{timestamp}/`
   - Each run contains: `predictions/val_probs.npy`, `metrics/metrics.json`

3. **Data Format Assumption**: 
   - `val_probs.npy` contains probability arrays of shape `(n_samples, 2)` where columns are [LGG_prob, HGG_prob]
   - `val_labels.npy` contains ground truth labels (0 for LGG, 1 for HGG)
   - We will extract only the HGG probability (column index 1) for stacking

## Implementation Steps

### Step 1: Create Ensemble Directory Structure

Create the following directory structure:
```
ensemble/
├── oof_predictions/
│   ├── resnet50_3d_oof.csv
│   ├── swinunetr_3d_oof.csv
│   └── dualstream_mil_3d_oof.csv
└── README.md
```

### Step 2: Model and Results Directory Mapping

Define model name to results directory mapping:
- `ResNet50-3D` → `results/ResNet50-3D/runs/`
- `SwinUNETR-3D` → `results/SwinUNETR-3D/runs/`
- `DualStreamMIL-3D` → `results/DualStreamMIL-3D/runs/`

### Step 3: Latest Run Selection Algorithm

For each model and each fold (0-4):

1. List all run directories in `results/{ModelName}/runs/fold_{k}/`
2. Filter for directories matching pattern `run_*` (timestamp-based)
3. Get modification time for each run directory
4. Select the run directory with the most recent modification time
5. Verify required files exist:
   - `predictions/val_probs.npy`
   - `metrics/metrics.json` (optional, for verification)

### Step 4: Extract OOF Predictions Per Fold

For each selected run:

1. Load `predictions/val_probs.npy` as numpy array
2. Extract HGG probabilities: `val_probs[:, 1]` (second column)
3. Load corresponding validation split CSV: `splits/fold_{k}_val.csv`
4. Extract patient IDs from the CSV (first column: `patient_id`)
5. Create a DataFrame with columns: `patient_id`, `fold`, `hgg_prob`, `label`
6. Add ground truth label from validation split CSV (`class_label` column: 0=LGG, 1=HGG)

### Step 5: Aggregate OOF Predictions Across Folds

For each model:

1. Concatenate DataFrames from all 5 folds
2. Verify that each patient appears exactly once (should be true by CV design)
3. Sort by patient_id for consistency
4. Save as CSV: `ensemble/oof_predictions/{model_name}_oof.csv`
   - Columns: `patient_id`, `fold`, `hgg_prob`, `label`

### Step 6: Verification Checks

Before finalizing OOF predictions:

1. **Uniqueness Check**: Verify no duplicate patient IDs within each model's OOF file
2. **Completeness Check**: Verify all patients from all validation splits are present
3. **Fold Assignment Check**: Verify each patient's fold assignment matches the validation split they came from
4. **Probability Range Check**: Verify all probabilities are in [0, 1]
5. **Label Consistency Check**: Verify labels match between OOF file and original split files

### Step 7: Create Summary Documentation

Create `ensemble/README.md` documenting:
- Source of each OOF prediction file
- Latest run timestamp used for each fold
- Verification status
- Column descriptions

## Implementation Code Structure

The implementation should be organized as a single script: `scripts/ensemble/prepare_oof_predictions.py`

### Main Functions

1. `find_latest_run(model_name, fold) -> Path`: Returns path to latest run directory
2. `load_oof_predictions_from_run(run_dir, fold, val_split_csv) -> pd.DataFrame`: Loads and organizes predictions for one fold
3. `aggregate_oof_predictions(model_name) -> pd.DataFrame`: Aggregates all folds for one model
4. `verify_oof_predictions(df, model_name) -> bool`: Performs verification checks
5. `main()`: Orchestrates the entire process

### Expected Output Format

Each OOF CSV file (`ensemble/oof_predictions/{model}_oof.csv`) will have:
- `patient_id`: Patient identifier (string, e.g., "Brats18_2013_10_1")
- `fold`: Fold number (0-4) from which this prediction came
- `hgg_prob`: HGG class probability (float, 0-1)
- `label`: Ground truth label (int, 0=LGG, 1=HGG)

Example:
```csv
patient_id,fold,hgg_prob,label
Brats18_2013_10_1,0,0.8923,1
Brats18_2013_20_1,0,0.7541,1
Brats18_2013_27_1,1,0.1234,0
...
```

## Critical Constraints

1. **No Averaging Across Runs**: Only the latest run per fold is used. No aggregation of multiple runs.
2. **No Data Leakage**: Predictions come only from the fold where the patient was in the validation set.
3. **Patient-Level Only**: One prediction per patient per model, at the patient level (not instance/slice level).
4. **No Meta-Learner Training**: This implementation only prepares data; meta-learner training is a separate step.

## Next Steps (Out of Scope)

After OOF predictions are prepared:
1. Merge OOF predictions from all three models into a single DataFrame
2. Train logistic regression meta-learner on merged OOF predictions
3. Evaluate ensemble performance using cross-validation
4. Prepare ensemble inference pipeline for test data

## Error Handling

The implementation should:
- Raise clear errors if required files are missing
- Warn if multiple runs exist but proceed with the latest
- Log which run was selected for each fold
- Fail gracefully if verification checks fail

