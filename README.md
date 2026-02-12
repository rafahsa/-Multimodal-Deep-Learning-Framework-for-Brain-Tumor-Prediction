# Brain Tumor MRI Classification Project

## Overview

This is a research-grade medical imaging project for brain tumor grade classification (LGG vs HGG) using 3D MRI volumes from the BraTS 2018 dataset.

## Project Structure

```
/workspace/brain_tumor_project/
├── data/                    # All data (raw and processed)
│   ├── raw/                 # Original, unmodified data
│   │   └── BraTS2018/       # BraTS 2018 dataset organized by grade
│   │       ├── HGG/         # High-Grade Glioma cases
│   │       └── LGG/         # Low-Grade Glioma cases
│   └── processed/           # Stage-wise processed data
│       ├── stage_0_raw/     # Raw data reference
│       ├── stage_1_n4/      # N4 bias field correction
│       ├── stage_2_zscore/  # Z-score normalization
│       ├── stage_3_crop/    # ROI cropping
│       ├── stage_4_resize/  # Resize to fixed volume
│       ├── stage_5_augmented/ # (Runtime-only: intentionally empty)
│       └── stage_6_balanced/  # (Runtime-only: intentionally empty)
├── scripts/                 # All executable scripts
│   ├── preprocessing/       # Data preprocessing scripts
│   ├── training/            # Model training scripts
│   ├── evaluation/          # Model evaluation scripts
│   └── utils/               # Utility functions
├── configs/                 # Configuration files (YAML/JSON)
├── models/                  # Model definitions and saved checkpoints
├── experiments/             # Experiment tracking and results
│   ├── resnet50_3d/         # ResNet50-3D experiments
│   ├── swin_unetr/          # Swin UNETR experiments
│   └── mil/                 # MIL experiments
├── logs/                    # Training and preprocessing logs
│   ├── preprocessing/
│   ├── training/
│   └── evaluation/
├── notebooks/               # Jupyter notebooks (minimal, for exploration)
└── docs/                    # Documentation

```

## Data Persistence

**CRITICAL**: This entire project is stored on a persistent volume (`/workspace/`) that survives pod stop/restart. All data, code, and outputs are stored exclusively within `/workspace/brain_tumor_project/`.

## Dataset

- **Source**: BraTS 2018 (MICCAI Brain Tumor Segmentation Challenge 2018)
- **Format**: 3D NIfTI volumes (.nii / .nii.gz)
- **Modalities**: T1, T1CE, T2, FLAIR, Segmentation
- **Classes**: HGG (210 cases), LGG (75 cases)

## Models

- ResNet50-3D
- Swin UNETR (classification)
- MIL (Multiple Instance Learning, optional)

## Preprocessing Pipeline

The preprocessing pipeline consists of two distinct types of stages:

**Disk-based preprocessing stages** (Stages 1-4):
1. N4 Bias Field Correction
2. Z-score normalization
3. ROI cropping
4. Resize to fixed volume

These stages produce persistent output files stored in `data/processed/` and can be inspected independently.

**Runtime stages** (Stages 5-6):
5. Data augmentation (training only)
6. Oversampling / class balancing

These stages are applied dynamically during training and do not create files on disk. See the section below for details.

**Metadata generation** (Stage 7):
7. K-Fold cross-validation splits

This stage generates split definition files (CSV/JSON) for cross-validation. No image data is moved or modified.

## Runtime Stages (No Disk Output): Stage 5 & Stage 6

Stage 5 (Geometric Data Augmentation) and Stage 6 (Class Balancing / Oversampling) are **runtime-only stages** that are applied dynamically during training. Unlike Stages 1-4, these stages do **not** generate any files on disk.

### Stage 5: Geometric Data Augmentation

- **Location**: Implemented in `utils/augmentations_3d.py`
- **Application**: Applied on-the-fly inside the Dataset/DataLoader when `mode="train"`
- **Scope**: Training data only; validation and test data are **not** augmented
- **Transforms**: Medical-safe geometric augmentations (rotation, flip, zoom, translation) using MONAI
- **Storage**: No files created; augmentation happens in memory during DataLoader iteration

### Stage 6: Class Balancing / Oversampling

- **Application**: Applied dynamically during training data sampling
- **Purpose**: Addresses class imbalance between HGG and LGG cases
- **Scope**: Training data only; validation/test sets remain unchanged
- **Storage**: No files created; balancing happens at DataLoader level

### Design Rationale

This runtime-only design provides several critical advantages:

1. **Prevents Data Leakage**: Validation and test sets remain unmodified, ensuring fair model evaluation
2. **Reduces Storage Usage**: Avoids creating terabytes of augmented/balanced data files
3. **Enables Infinite Variety**: Same volume produces different augmentations each epoch
4. **Follows Best Practices**: Aligns with medical imaging ML best practices where augmentation is applied dynamically
5. **Facilitates Experimentation**: Augmentation parameters can be adjusted without reprocessing data

### Directory Structure Note

The directories `data/processed/stage_5_augmented/` and `data/processed/stage_6_balanced/` may exist in the project structure for organizational purposes, but they are **intentionally empty**. These directories are placeholders and do not contain any processed data files, as Stages 5 and 6 operate entirely in memory during training.

## Stage 7: K-Fold Cross-Validation Splits

Stage 7 generates patient-level K-Fold cross-validation splits for model training and evaluation. Unlike preprocessing stages 1-4, Stage 7 creates **metadata files only** (CSV and JSON) that define train/validation splits. No image files are moved, copied, or modified.

### Implementation

Stage 7 consists of two scripts:

1. **Index Builder** (`scripts/splits/build_stage4_index.py`):
   - Scans Stage 4 output directory
   - Creates patient index: `data/index/stage4_index.csv`
   - Contains patient metadata (ID, class, modality paths)

2. **Split Generator** (`scripts/splits/make_kfold_splits.py`):
   - Uses StratifiedKFold (k=5, seed=42) for patient-level splitting
   - Generates split files in `splits/` directory:
     - `splits/kfold_5fold_seed42.json`: JSON summary
     - `splits/fold_X_train.csv`: Training set for fold X
     - `splits/fold_X_val.csv`: Validation set for fold X

### Key Features

- **Patient-level splitting**: Prevents data leakage (entire patient in one fold)
- **Stratified**: Preserves class ratio (HGG:LGG) in each fold
- **Reproducible**: Fixed random seed ensures consistent splits
- **Metadata-only**: Only creates split definition files, no data movement
- **Git-friendly**: Small CSV/JSON files can be version-controlled

### Usage

```bash
# Step 1: Build patient index
python scripts/splits/build_stage4_index.py

# Step 2: Generate K-Fold splits
python scripts/splits/make_kfold_splits.py --k 5 --seed 42
```

See `docs/stage7_kfold.md` for detailed documentation.

## Entropy-based Slice Selection (MIL-only, Runtime Metadata Stage)

An entropy-based slice informativeness analysis stage is available for Multiple Instance Learning (MIL) models. This stage computes Shannon entropy for each 2D slice in 3D MRI volumes to identify the most informative slices for training.

**Key Characteristics:**
- **MIL-specific**: Used exclusively with slice-based MIL models (not ResNet50-3D or Swin UNETR)
- **Metadata-only**: Creates JSON metadata files only, no image modification
- **Runtime metadata**: Analysis results stored in `data/entropy/` for use during training
- **Not a preprocessing stage**: Does not modify Stages 1-4 outputs

**Implementation:**
- Utility module: `utils/entropy_analysis.py`
- Runner script: `scripts/analysis/run_entropy_analysis.py`
- Output: `data/entropy/<patient_id>_entropy.json` (entropy scores and top-k slice indices)

**Usage:**
```bash
python scripts/analysis/run_entropy_analysis.py --modality flair --axis axial --top-k 16
```

See `docs/stage_entropy_mil.md` for detailed documentation.

## Operating Threshold / Deployment Settings

The ensemble meta-learner uses configurable decision thresholds for binary classification (LGG=0, HGG=1). We select **0.22 as the default operating threshold for the main system (best F1, balanced precision/recall)**. We provide **0.19 as a high-sensitivity operating point to reduce false negatives in HGG detection**.

### Probability Calibration

We applied **post-hoc Platt probability calibration** to improve the reliability of ensemble predictions. Calibration significantly reduced Brier score (0.119 → 0.099, improvement: 0.021) and Expected Calibration Error (ECE: 0.119 → 0.087, improvement: 0.032) **without degrading classification performance**. Operating thresholds were re-selected on calibrated probabilities using a held-out validation set (30% of OOF predictions, seed=42) to prevent data leakage.

**Calibration is optional at inference time** via the `--calibration-mode` flag. Default inference remains uncalibrated for backward compatibility. When calibration is enabled, use the re-selected thresholds:
- **Balanced (calibrated)**: threshold = 0.41 (Precision=0.9365, Recall=0.9365, F1=0.9365, FN=4, FP=4)
- **High-sensitivity (calibrated)**: threshold = 0.38 (Precision=0.9091, Recall=0.9524, F1=0.9302, FN=3, FP=6)

### Operating Points

| Threshold | Precision | Recall | F1 | Accuracy | FN | FP | Use Case |
|-----------|-----------|--------|----|----------|----|----|----------|
| **0.22** | 0.9000 | 0.9000 | 0.9000 | 0.8526 | 21 | 21 | **Default (balanced)** |
| **0.19** | 0.8319 | 0.9429 | 0.8839 | 0.8175 | 12 | 40 | **High-sensitivity (lower FN)** |
| 0.50 | 0.9643 | 0.7714 | 0.8571 | 0.8105 | 48 | 6 | Baseline reference |

### Inference Commands

**Official default run (threshold 0.22):**
```bash
python scripts/ensemble/test_ensemble_on_new_patients.py
```

**High-sensitivity run (threshold 0.19):**
```bash
python scripts/ensemble/test_ensemble_on_new_patients.py --threshold 0.19
```

**Baseline reference (threshold 0.50):**
```bash
python scripts/ensemble/test_ensemble_on_new_patients.py --threshold 0.50
```

**With probability calibration (Platt scaling):**
```bash
python scripts/ensemble/test_ensemble_on_new_patients.py --calibration-mode platt --threshold 0.41
```

The inference script defaults to threshold 0.22 if no `--threshold` argument is provided. Calibration is disabled by default (`--calibration-mode none`) for backward compatibility.

## Usage

See individual README files in subdirectories for specific usage instructions.

## Reproducibility

All preprocessing steps are designed to be reproducible and traceable. Disk-based stages (1-4) output to their own directories, ensuring that raw data is never modified and intermediate results can be inspected. Runtime stages (5-6) use deterministic seeds when applicable to ensure reproducibility across training runs.

# -Multimodal-Deep-Learning-Framework-for-Brain-Tumor-Prediction
