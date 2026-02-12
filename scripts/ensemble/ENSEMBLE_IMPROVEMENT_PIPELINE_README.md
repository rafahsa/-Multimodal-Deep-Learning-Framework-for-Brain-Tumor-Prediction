# Ensemble Improvement Pipeline (No Retraining)

**Goal**: Improve ensemble performance WITHOUT retraining any deep learning models.

**Focus**: Reduce FN, stabilize probabilities, add orthogonal signal.

**Constraint**: NO retraining of Swin/ResNet/MIL models.

---

## Pipeline Overview

This pipeline implements 6 steps to improve ensemble performance:

1. **Test-Time Augmentation (TTA)** - Apply light augmentations to Swin and ResNet
2. **Calibration** - Nested-CV safe probability calibration
3. **Threshold Tuning** - Optimize thresholds for recall targets
4. **Non-DL Feature Extraction** - Extract patient-level features (no segmentation)
5. **Meta-Learner Retraining** - Retrain with new features
6. **Ablation Study** - Compare all configurations

---

## Usage

### Run Full Pipeline

```bash
python scripts/ensemble/improve_ensemble_no_retraining.py
```

### Run Individual Steps

```bash
# Step 1: TTA (WARNING: Very slow, ~2-4 hours for 285 patients)
python -c "from scripts.ensemble.step1_tta import apply_tta_to_oof; import pandas as pd; df = pd.read_csv('ensemble/oof_predictions/merged_oof_predictions.csv'); apply_tta_to_oof(df, Path('ensemble/results/ensemble_improvements'))"

# Step 2: Calibration
python -c "from scripts.ensemble.step2_calibration import apply_nested_cv_calibration; import pandas as pd; df = pd.read_csv('ensemble/results/ensemble_improvements/oof_predictions_with_tta.csv'); apply_nested_cv_calibration(df, Path('ensemble/results/ensemble_improvements'))"

# Step 3: Threshold Tuning
python -c "from scripts.ensemble.step3_threshold_tuning import tune_ensemble_thresholds; import pandas as pd; df = pd.read_csv('ensemble/results/ensemble_improvements/oof_predictions_with_calibration.csv'); tune_ensemble_thresholds(df, Path('ensemble/results/ensemble_improvements'))"

# Step 4: Non-DL Features
python -c "from scripts.ensemble.step4_non_dl_features import extract_non_dl_features; import pandas as pd; df = pd.read_csv('ensemble/results/ensemble_improvements/oof_predictions_with_calibration.csv'); extract_non_dl_features(df, Path('ensemble/results/ensemble_improvements'))"

# Step 5: Meta-Learner
python -c "from scripts.ensemble.step5_meta_learner import retrain_meta_learner_with_features; import pandas as pd; df = pd.read_csv('ensemble/results/ensemble_improvements/oof_predictions_with_features.csv'); retrain_meta_learner_with_features(df, Path('ensemble/results/ensemble_improvements'))"

# Step 6: Ablation
python -c "from scripts.ensemble.step6_ablation import run_ablation_study; ..."
```

---

## Step Details

### Step 1: Test-Time Augmentation (TTA)

**What it does**:
- Loads Swin and ResNet models for each fold
- Loads volumes for each patient in validation sets
- Applies light, MRI-safe augmentations (N=12 per patient)
- Averages predictions across augmentations
- Saves: `swin_prob_tta`, `resnet_prob_tta`

**Augmentations**:
- Small rotations (±5.7 degrees)
- Random flips (x, y, z axes)
- Small translations (±5 voxels)
- Small scale changes (0.95-1.05)
- Light Gaussian noise (std=0.01)

**WARNING**: This step is **very slow** (~2-4 hours for 285 patients):
- 285 patients × 12 augmentations × 2 models = 6,840 inference calls
- Each inference requires loading model and volume
- Consider running overnight or on GPU cluster

**Output**: `oof_predictions_with_tta.csv`

---

### Step 2: Calibration

**What it does**:
- Applies nested-CV safe calibration to Swin and ResNet probabilities
- For each fold: fit calibrator on other folds, apply to this fold
- Method: Platt scaling (LogisticRegression) or IsotonicRegression
- MIL probabilities already calibrated, kept as-is

**Output**: 
- `oof_predictions_with_calibration.csv`
- `calibrators.joblib` (saved calibrators per fold)

---

### Step 3: Threshold Tuning

**What it does**:
- Trains meta-learner on calibrated probabilities
- Performs threshold sweep (0.01 to 0.99, step 0.01)
- Finds thresholds achieving Recall ≥ 0.85 and ≥ 0.90
- Reports per-fold metrics

**Output**: `threshold_tuning_results.csv`

---

### Step 4: Non-DL Feature Extraction

**What it does**:
- Extracts patient-level features from volumes (no segmentation):
  - Intensity statistics per modality (mean, std, skew, kurtosis, percentiles)
  - Global entropy
  - Gradient energy
- One row per patient, fold-aware

**Output**: 
- `non_dl_features.csv`
- `oof_predictions_with_features.csv` (merged with probabilities)

---

### Step 5: Meta-Learner Retraining

**What it does**:
- Retrains meta-learner with:
  - Calibrated probabilities (swin_prob_cal, resnet_prob_cal, mil_prob_cal)
  - Non-DL features
- Models: LogisticRegression, XGBoost (if available)
- Nested-CV evaluation only

**Output**: `meta_learner_results.json`

---

### Step 6: Ablation Study

**What it does**:
- Evaluates all configurations:
  1. Baseline ensemble
  2. + TTA
  3. + TTA + Calibration
  4. + TTA + Calibration + Threshold tuning
  5. + All above + Non-DL features
- Reports: FN, FP, Recall, Precision, F1, AUC (mean ± std)

**Output**: `ablation_study_results.json`

---

## Output Files

All outputs saved to: `ensemble/results/ensemble_improvements/`

1. `oof_predictions_with_tta.csv` - Step 1 output
2. `oof_predictions_with_calibration.csv` - Step 2 output
3. `calibrators.joblib` - Saved calibrators
4. `threshold_tuning_results.csv` - Step 3 output
5. `non_dl_features.csv` - Step 4 output
6. `oof_predictions_with_features.csv` - Step 4 merged output
7. `meta_learner_results.json` - Step 5 output
8. `ablation_study_results.json` - Step 6 output

---

## Expected Runtime

- **Step 1 (TTA)**: ~2-4 hours (very slow, loads models and volumes)
- **Step 2 (Calibration)**: ~1-2 minutes
- **Step 3 (Threshold Tuning)**: ~1-2 minutes
- **Step 4 (Non-DL Features)**: ~5-10 minutes (loads volumes)
- **Step 5 (Meta-Learner)**: ~1-2 minutes
- **Step 6 (Ablation)**: ~2-3 minutes

**Total**: ~2.5-4.5 hours (mostly Step 1)

---

## Notes

1. **Step 1 is slow**: Consider running overnight or on GPU cluster
2. **GPU required**: Step 1 needs GPU for inference
3. **Memory**: Step 1 loads models into memory (need ~8GB GPU memory)
4. **Nested-CV safe**: All steps respect fold structure (no data leakage)

---

## Troubleshooting

### Step 1 fails with "CUDA out of memory"
- Reduce `NUM_TTA` in `step1_tta.py` (default: 12)
- Process fewer patients at a time
- Use CPU (slower but less memory)

### Step 1 fails with "Model checkpoint not found"
- Ensure models are trained for all 5 folds
- Check checkpoint paths in `find_latest_checkpoint()`

### Step 4 fails with "Volume not found"
- Ensure Stage 4 preprocessing is complete
- Check data paths in `step4_non_dl_features.py`

---

## Next Steps

After running the pipeline:

1. Review `ablation_study_results.json` to see what helped
2. Compare configurations in the ablation table
3. Decide which components to keep
4. If FN < 5 achieved, proceed with final ensemble
5. If not, consider additional improvements (see research document)


