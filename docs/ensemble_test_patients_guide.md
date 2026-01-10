# Ensemble Model Test on New Patients - Usage Guide

**Date**: January 10, 2026  
**Status**: ✅ Script Ready

---

## Overview

This script tests the trained ensemble meta-learner model on new patient MRI images. It automatically:

1. Scans the test directory for patient images
2. Loads all 4 modalities (T1, T1ce, T2, FLAIR) for each patient
3. Generates predictions from all three base models (ResNet50-3D, SwinUNETR-3D, DualStreamMIL-3D)
4. Combines predictions using the ensemble meta-learner
5. Outputs HGG probability for each patient

---

## Required File Structure

Place your patient images in the following directory structure:

```
splits/DATA_FOR_TEST/
├── UCSF-PDGM-0004_T1.nii
├── UCSF-PDGM-0004_T1ce.nii
├── UCSF-PDGM-0004_T2.nii
├── UCSF-PDGM-0004_FLAIR.nii
├── UCSF-PDGM-0005_T1.nii
├── UCSF-PDGM-0005_T1ce.nii
├── UCSF-PDGM-0005_T2.nii
└── UCSF-PDGM-0005_FLAIR.nii
```

### File Naming Convention

- **Format**: `{PATIENT_ID}_{MODALITY}.nii` or `{PATIENT_ID}_{MODALITY}.nii.gz`
- **Patient ID**: Can be any identifier (e.g., `UCSF-PDGM-0004`, `Patient001`, etc.)
- **Modalities**: Must be exactly `T1`, `T1ce`, `T2`, `FLAIR` (case-sensitive)
- **File Extensions**: Supports both `.nii` and `.nii.gz`

### Required Modalities

All four modalities are required for each patient:
- `T1`: T1-weighted MRI
- `T1ce`: T1-weighted contrast-enhanced MRI
- `T2`: T2-weighted MRI
- `FLAIR`: FLAIR MRI

---

## Usage

### Basic Usage

```bash
python scripts/ensemble/test_ensemble_on_new_patients.py
```

This uses default settings:
- Test directory: `splits/DATA_FOR_TEST/`
- Meta-learner: `ensemble/models/meta_learner_logistic_regression.joblib`
- Fold: 0 (for checkpoint selection)
- Device: Auto (CUDA if available, else CPU)

### Custom Test Directory

```bash
python scripts/ensemble/test_ensemble_on_new_patients.py \
    --test-dir /path/to/your/test/data
```

### Full Options

```bash
python scripts/ensemble/test_ensemble_on_new_patients.py \
    --test-dir splits/DATA_FOR_TEST \
    --meta-learner ensemble/models/meta_learner_logistic_regression.joblib \
    --fold 0 \
    --device cuda \
    --bag-size 32
```

### Command-Line Arguments

- `--test-dir`: Directory containing test patient images (default: `splits/DATA_FOR_TEST`)
- `--meta-learner`: Path to meta-learner model (default: `ensemble/models/meta_learner_logistic_regression.joblib`)
- `--fold`: Fold number for checkpoint selection (default: 0, choices: 0-4)
- `--device`: Device for inference (default: `auto`, choices: `auto`, `cuda`, `cpu`)
- `--bag-size`: Bag size for MIL model (default: 32)

---

## Output Format

The script provides detailed output for each patient:

### Example Output

```
================================================================================
ENSEMBLE MODEL TEST RESULTS
================================================================================

Patient: UCSF-PDGM-0004
--------------------------------------------------------------------------------
  ResNet50-3D HGG probability:    0.923456
  SwinUNETR-3D HGG probability:   0.875432
  DualStreamMIL-3D HGG probability: 0.901234
  Ensemble HGG probability:       0.945678
  Ensemble prediction:            HGG

✓ HGG probability for patient 0004: 0.95

Patient: UCSF-PDGM-0005
--------------------------------------------------------------------------------
  ResNet50-3D HGG probability:    0.654321
  SwinUNETR-3D HGG probability:   0.712345
  DualStreamMIL-3D HGG probability: 0.678901
  Ensemble HGG probability:       0.723456
  Ensemble prediction:            HGG

✓ HGG probability for patient 0005: 0.72

================================================================================
Processed 2 patients successfully
================================================================================
```

### Output Explanation

For each patient, the script displays:

1. **Base Model Predictions**: Individual HGG probabilities from each of the three base models
   - ResNet50-3D
   - SwinUNETR-3D
   - DualStreamMIL-3D

2. **Ensemble Prediction**: Final combined result from the meta-learner
   - Ensemble HGG probability (0.0 to 1.0)
   - Ensemble prediction (HGG or LGG)

3. **Formatted Output**: Simplified format as requested
   - `HGG probability for patient {NUMBER}: {PROBABILITY}`

---

## How It Works

1. **Patient Detection**: Scans the test directory for all T1 files to identify patients
2. **Image Loading**: For each patient, loads all 4 modalities and stacks them as channels
3. **Base Model Inference**: 
   - ResNet50-3D: Processes full 3D volume
   - SwinUNETR-3D: Processes full 3D volume
   - DualStreamMIL-3D: Samples slices and processes as bag
4. **Ensemble Prediction**: Combines base model probabilities using the meta-learner
5. **Results Output**: Displays predictions for each patient

---

## Requirements

### Data Requirements

- All 4 modalities (T1, T1ce, T2, FLAIR) must be present for each patient
- Images must be in NIfTI format (`.nii` or `.nii.gz`)
- Images should be preprocessed (same pipeline as training data recommended)

### Model Requirements

- Trained base models (ResNet50-3D, SwinUNETR-3D, DualStreamMIL-3D) checkpoints
- Trained ensemble meta-learner model
- Checkpoints should be in `results/{MODEL_NAME}/runs/fold_{FOLD}/run_*/checkpoints/`

---

## Troubleshooting

### Common Issues

1. **"Test directory not found"**
   - **Problem**: Test directory doesn't exist
   - **Solution**: Create the directory or provide correct path with `--test-dir`

2. **"Modality not found"**
   - **Problem**: Missing modality file for a patient
   - **Solution**: Ensure all 4 modalities are present with correct naming

3. **"No patients found"**
   - **Problem**: No T1 files found in test directory
   - **Solution**: Check file naming matches expected format: `{PATIENT_ID}_T1.nii`

4. **"Checkpoint not found"**
   - **Problem**: Base model checkpoints missing
   - **Solution**: Ensure models are trained and checkpoints exist in results directory

5. **"CUDA out of memory"**
   - **Problem**: GPU memory insufficient
   - **Solution**: Use CPU with `--device cpu` or process fewer patients at once

### File Naming Issues

- Ensure modality names match exactly: `T1`, `T1ce`, `T2`, `FLAIR` (case-sensitive)
- Patient ID can be any string, but must be consistent across modalities
- Files can be `.nii` or `.nii.gz`

---

## Example: Testing on Two Patients

Given the file structure:

```
splits/DATA_FOR_TEST/
├── UCSF-PDGM-0004_T1.nii
├── UCSF-PDGM-0004_T1ce.nii
├── UCSF-PDGM-0004_T2.nii
├── UCSF-PDGM-0004_FLAIR.nii
├── UCSF-PDGM-0005_T1.nii
├── UCSF-PDGM-0005_T1ce.nii
├── UCSF-PDGM-0005_T2.nii
└── UCSF-PDGM-0005_FLAIR.nii
```

Run:

```bash
python scripts/ensemble/test_ensemble_on_new_patients.py
```

The script will:
1. Automatically detect both patients (UCSF-PDGM-0004 and UCSF-PDGM-0005)
2. Load all 4 modalities for each patient
3. Generate predictions from all three base models
4. Combine predictions using the ensemble meta-learner
5. Output results in the requested format

---

## Notes

1. **Preprocessing**: For best results, test images should follow the same preprocessing pipeline as training data (N4 correction, normalization, cropping, resizing).

2. **Model Selection**: The script uses checkpoints from fold 0 by default. You can specify a different fold with `--fold`.

3. **Batch Processing**: The script processes patients sequentially. For large batches, you may want to modify the script to process multiple patients in parallel.

4. **Output Format**: The script outputs both detailed and simplified formats. The simplified format (`HGG probability for patient {NUMBER}: {PROBABILITY}`) matches the requested output.

---

**Script Location**: `scripts/ensemble/test_ensemble_on_new_patients.py`  
**Last Updated**: January 10, 2026

