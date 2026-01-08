# Brain Tumor Classification Project - Comprehensive Overview

## Table of Contents
1. [Project Structure Overview](#project-structure-overview)
2. [Details of Each Folder](#details-of-each-folder)
3. [Content of Key Files](#content-of-key-files)
4. [Directory and File Usage](#directory-and-file-usage)
5. [Metadata and Results](#metadata-and-results)
6. [Preprocessing Phases](#preprocessing-phases)
7. [Entropy Processing](#entropy-processing)
8. [Training Strategy](#training-strategy)
9. [Data Handling](#data-handling)

---

## Project Structure Overview

```
brain_tumor_project/
├── data/                          # All data (raw and processed)
│   ├── raw/                       # Original, unmodified data
│   │   └── BraTS2018/             # BraTS 2018 dataset organized by grade
│   │       ├── HGG/               # High-Grade Glioma cases (210 patients)
│   │       └── LGG/               # Low-Grade Glioma cases (75 patients)
│   ├── processed/                 # Stage-wise processed data
│   │   ├── stage_0_raw/           # Raw data reference
│   │   ├── stage_1_n4/            # N4 bias field correction
│   │   ├── stage_2_zscore/        # Z-score normalization
│   │   ├── stage_3_crop/          # ROI cropping
│   │   ├── stage_4_resize/        # Resize to fixed volume (128x128x128)
│   │   ├── stage_5_augmented/     # (Runtime-only: intentionally empty)
│   │   └── stage_6_balanced/      # (Runtime-only: intentionally empty)
│   ├── entropy/                   # Entropy scores and metadata (MIL-only)
│   │   └── <patient_id>_entropy.json  # Per-patient entropy JSON files
│   ├── index/                     # Patient index files
│   │   └── stage4_index.csv       # Index of all patients after Stage 4
│   └── kfold_splits/             # K-Fold cross-validation splits
│
├── models/                        # Model definitions and saved checkpoints
│   ├── resnet50_3d_fast/         # ResNet50-3D model architecture
│   ├── swin_unetr/               # Swin UNETR model architecture
│   └── dual_stream_mil/          # Multiple Instance Learning models
│       ├── model.py              # Single-modality MIL model
│       └── model_multi_modal.py   # Multi-modality MIL model (FLAIR + T1ce)
│
├── preprocessing/                # Preprocessing utilities
│   └── 01_n4_bias.py             # N4 bias correction utility
│
├── scripts/                       # All executable scripts
│   ├── preprocessing/            # Data preprocessing scripts
│   │   ├── run_stage1_n4.py      # Stage 1: N4 bias correction
│   │   ├── run_stage2_zscore.py  # Stage 2: Z-score normalization
│   │   ├── run_stage3_crop.py    # Stage 3: ROI cropping
│   │   └── run_stage4_resize.py  # Stage 4: Resize to fixed size
│   ├── training/                 # Model training scripts
│   │   ├── train_mil.py          # Single-modality MIL training
│   │   ├── train_mil_multi_modal.py  # Multi-modality MIL training
│   │   └── run_mil_kfold.py      # K-Fold cross-validation runner
│   ├── splits/                   # Data splitting scripts
│   │   ├── build_stage4_index.py # Build patient index from Stage 4
│   │   └── make_kfold_splits.py  # Generate K-Fold splits
│   └── analysis/                 # Analysis and visualization scripts
│       ├── run_entropy_analysis.py  # Compute entropy scores
│       └── visualize_entropy.py     # Visualize entropy results
│
├── utils/                         # Utility functions and helper scripts
│   ├── augmentations_3d.py       # 3D data augmentation transforms
│   ├── class_balancing.py         # Class balancing utilities
│   ├── entropy_analysis.py       # Entropy computation utilities
│   ├── ldam_loss.py              # LDAM loss function implementation
│   ├── mil_dataset.py             # Single-modality MIL dataset
│   └── mil_dataset_multi_modal.py # Multi-modality MIL dataset
│
├── configs/                       # Configuration files (YAML)
│   ├── stage_1_n4.yaml           # Stage 1 configuration
│   ├── stage_2_zscore.yaml        # Stage 2 configuration
│   ├── stage_3_crop.yaml          # Stage 3 configuration
│   └── stage_4_resize.yaml       # Stage 4 configuration
│
├── results/                       # Training results and evaluation metrics
│   ├── MIL/                      # MIL model results
│   │   ├── runs/                 # Individual training runs
│   │   │   └── fold_X/          # Per-fold results
│   │   │       ├── checkpoints/  # Model checkpoints
│   │   │       ├── metrics/      # Evaluation metrics (JSON)
│   │   │       ├── plots/        # Training curves, confusion matrices
│   │   │       └── predictions/  # Prediction outputs
│   │   └── kfold_summary/        # K-Fold cross-validation summary
│   ├── Swin_UNETR/               # Swin UNETR results
│   └── entropy_visualization/    # Entropy visualization outputs
│
├── splits/                        # K-Fold split definition files
│   ├── kfold_5fold_seed42.json   # K-Fold summary JSON
│   ├── fold_0_train.csv          # Training set for fold 0
│   ├── fold_0_val.csv            # Validation set for fold 0
│   └── ... (fold_1 through fold_4)
│
├── logs/                          # Training and preprocessing logs
│   ├── preprocessing/            # Preprocessing stage logs
│   └── training/                # Training logs
│
├── notebooks/                     # Jupyter notebooks (minimal, for exploration)
├── docs/                          # Documentation files
└── experiments/                   # Experiment tracking
```

---

## Details of Each Folder

### `data/raw/`
**Purpose**: Stores original, unmodified MRI data from BraTS 2018 dataset.

**Contents**:
- Original NIfTI files (.nii or .nii.gz format)
- Organized by tumor grade: `HGG/` (210 cases) and `LGG/` (75 cases)
- Each patient folder contains:
  - `*_t1.nii` / `*_t1.nii.gz`: T1-weighted MRI
  - `*_t1ce.nii` / `*_t1ce.nii.gz`: T1-weighted contrast-enhanced MRI
  - `*_t2.nii` / `*_t2.nii.gz`: T2-weighted MRI
  - `*_flair.nii` / `*_flair.nii.gz`: FLAIR MRI
  - `*_seg.nii` / `*_seg.nii.gz`: Segmentation mask (ground truth)

**Important**: This directory is **NEVER modified** by preprocessing scripts. All processed data is written to `data/processed/`.

---

### `data/processed/`
**Purpose**: Contains stage-wise processed data. Each preprocessing stage outputs to its own subdirectory.

#### `stage_0_raw/`
- Reference to raw data (may be empty or contain symlinks)

#### `stage_1_n4/`
**Purpose**: N4 Bias Field Correction
- **Input**: `data/raw/` (original NIfTI files)
- **Output**: Bias-corrected NIfTI files (.nii.gz format)
- **Processing**: Applies N4 bias field correction to reduce intensity inhomogeneity
- **Modalities processed**: t1, t1ce, t2, flair (NOT seg)
- **File naming**: `<patient_id>_<modality>.nii.gz`
- **Directory structure**: `stage_1_n4/train/<class>/<patient_id>/`

#### `stage_2_zscore/`
**Purpose**: Z-score Normalization
- **Input**: `data/processed/stage_1_n4/`
- **Output**: Z-score normalized NIfTI files
- **Processing**: 
  - Computes mean and std **only on brain voxels** (values > 0)
  - Normalizes: `(voxel - mean) / (std + eps)`
  - Preserves background (zeros remain zero)
- **File naming**: `<patient_id>_<modality>.nii.gz`
- **Directory structure**: `stage_2_zscore/train/<class>/<patient_id>/`

#### `stage_3_crop/`
**Purpose**: ROI (Region of Interest) Cropping
- **Input**: `data/processed/stage_2_zscore/`
- **Output**: Cropped volumes (bounding box around brain)
- **Processing**:
  - Computes bounding box from brain mask (all modalities use same bbox per patient)
  - Crops to bounding box with padding (default: 10 voxels)
  - Preserves spatial relationships (updates origin)
- **File naming**: `<patient_id>_<modality>.nii.gz`
- **Directory structure**: `stage_3_crop/train/<class>/<patient_id>/`

#### `stage_4_resize/`
**Purpose**: Resize to Fixed Volume Size
- **Input**: `data/processed/stage_3_crop/`
- **Output**: Resized volumes (128x128x128 voxels)
- **Processing**:
  - Resamples to fixed size using linear interpolation
  - Updates spacing to maintain physical dimensions
  - **This is the final preprocessing stage used for training**
- **File naming**: `<patient_id>_<modality>.nii.gz`
- **Directory structure**: `stage_4_resize/train/<class>/<patient_id>/`
- **Usage**: **This directory is used directly for model training**

#### `stage_5_augmented/` and `stage_6_balanced/`
**Purpose**: Runtime-only stages (intentionally empty directories)
- **No files stored**: These stages are applied dynamically during training
- **Stage 5**: Data augmentation (geometric transforms) - applied in DataLoader
- **Stage 6**: Class balancing (oversampling) - applied via WeightedRandomSampler
- **Rationale**: Prevents data leakage, reduces storage, enables infinite variety

---

### `data/entropy/`
**Purpose**: Stores entropy-based slice informativeness metadata (MIL-only).

**Contents**:
- JSON files: `<patient_id>_entropy.json`
- Each JSON file contains:
  ```json
  {
    "patient_id": "Brats18_TCIA10_103_1",
    "modality": "flair",
    "axis": "axial",
    "entropy_scores": [0.85, 0.92, ..., 0.78],  // One per slice
    "top_k_indices": [45, 67, ..., 23],          // Top-k slice indices
    "top_k": 16,
    "num_slices": 128,
    "stats": {
      "mean": 0.75,
      "std": 0.12,
      "min": 0.45,
      "max": 0.95
    }
  }
  ```

**Usage**: 
- Used **exclusively** by MIL models for slice selection
- Not used by ResNet50-3D or Swin UNETR (they process full 3D volumes)
- Generated by `scripts/analysis/run_entropy_analysis.py`

---

### `data/index/`
**Purpose**: Patient index files for data management.

**Contents**:
- `stage4_index.csv`: Index of all patients after Stage 4 preprocessing
- Columns: `patient_id`, `class`, `class_label`, `path_t1`, `path_t1ce`, `path_t2`, `path_flair`
- Generated by: `scripts/splits/build_stage4_index.py`
- Used by: K-Fold split generation

---

### `models/`
**Purpose**: Model architecture definitions.

#### `models/resnet50_3d_fast/`
- ResNet50-3D architecture for 3D volume classification
- Processes full 3D volumes (128x128x128)

#### `models/swin_unetr/`
- Swin UNETR architecture for 3D volume classification
- Processes full 3D volumes (128x128x128)

#### `models/dual_stream_mil/`
- **`model.py`**: Single-modality MIL model
- **`model_multi_modal.py`**: Multi-modality Dual-Stream MIL model
  - Separate ResNet34 encoders for FLAIR and T1ce
  - Max-pooling + Gated Attention aggregation per modality
  - Fusion at bag (patient) level
  - Binary classification head (HGG vs LGG)

---

### `utils/`
**Purpose**: Utility functions and helper scripts.

#### `utils/augmentations_3d.py`
**Purpose**: 3D geometric data augmentation (Stage 5 - runtime-only).

**Functions**:
- `get_train_transforms_3d()`: Training augmentation pipeline
  - Random rotation (±15 degrees)
  - Random flip (x, y, z axes)
  - Random zoom (±10%)
  - Random translation (±10%)
- `get_val_transforms_3d()`: Validation transforms (no augmentation)
- Model-specific wrappers: `get_mil_transforms_3d()`, `get_resnet3d_transforms_3d()`, etc.

**Usage**: Applied dynamically in DataLoader during training (mode="train")

#### `utils/class_balancing.py`
**Purpose**: Class balancing utilities (Stage 6 - runtime-only).

**Functions**:
- `compute_class_weights()`: Compute class weights from label distribution
- `get_weighted_sampler()`: Create WeightedRandomSampler for balanced sampling
- `get_balanced_dataloader()`: Create DataLoader with class balancing
- `get_class_distribution()`: Get class distribution statistics

**Strategies**:
- `inverse_freq`: Weight = total_samples / (num_classes * class_freq)
- `balanced`: Same as inverse_freq
- `uniform`: All classes have weight 1.0

**Usage**: Applied via WeightedRandomSampler in training DataLoader

#### `utils/entropy_analysis.py`
**Purpose**: Entropy computation for slice informativeness (MIL-only).

**Functions**:
- `compute_slice_entropy()`: Compute Shannon entropy for each 2D slice
- `select_top_k_slices()`: Select top-k most informative slices
- `compute_volume_entropy_stats()`: Compute entropy statistics

**Usage**: Used by `scripts/analysis/run_entropy_analysis.py` to generate entropy JSON files

#### `utils/ldam_loss.py`
**Purpose**: LDAM (Large Margin) loss function for class imbalance.

**Features**:
- Label-Distribution-Aware Margin loss
- Deferred Re-Weighting (DRW) support
- Configurable margin and scaling factor

**Usage**: Used in MIL training scripts

#### `utils/mil_dataset.py` and `utils/mil_dataset_multi_modal.py`
**Purpose**: PyTorch Dataset classes for MIL models.

**Features**:
- Load 3D volumes and extract 2D slices
- Entropy-based slice selection (top-k slices)
- Support for single-modality and multi-modality MIL
- Custom collate functions for bag-of-slices

---

### `scripts/`
**Purpose**: Executable scripts for preprocessing, training, and analysis.

#### `scripts/preprocessing/`
**Purpose**: Data preprocessing pipeline scripts.

**Scripts**:
1. **`run_stage1_n4.py`**: N4 bias field correction
   - Uses SimpleITK N4BiasFieldCorrectionImageFilter
   - Creates brain mask using Otsu thresholding
   - Supports parallel processing and resumability
   - Generates manifest files for tracking

2. **`run_stage2_zscore.py`**: Z-score normalization
   - Computes statistics on brain voxels only
   - Preserves background (zeros)
   - Supports parallel processing and resumability

3. **`run_stage3_crop.py`**: ROI cropping
   - Computes bounding box per patient (all modalities use same bbox)
   - Supports "reference_modality" or "union" bbox modes
   - Adds padding around bounding box

4. **`run_stage4_resize.py`**: Resize to fixed size
   - Resamples to 128x128x128 using linear interpolation
   - Updates spacing to maintain physical dimensions
   - **Final preprocessing stage used for training**

#### `scripts/training/`
**Purpose**: Model training scripts.

**Scripts**:
1. **`train_mil.py`**: Single-modality MIL training
2. **`train_mil_multi_modal.py`**: Multi-modality MIL training (FLAIR + T1ce)
   - Entropy-based slice selection (always enabled)
   - LDAM loss with DRW
   - Class balancing via WeightedRandomSampler
   - Early stopping, mixed precision, gradient clipping
3. **`run_mil_kfold.py`**: K-Fold cross-validation runner

#### `scripts/splits/`
**Purpose**: Data splitting scripts.

**Scripts**:
1. **`build_stage4_index.py`**: Build patient index from Stage 4 outputs
   - Scans `data/processed/stage_4_resize/`
   - Creates `data/index/stage4_index.csv`

2. **`make_kfold_splits.py`**: Generate K-Fold cross-validation splits
   - Uses StratifiedKFold (k=5, seed=42)
   - Patient-level splitting (prevents data leakage)
   - Generates CSV files: `fold_X_train.csv`, `fold_X_val.csv`

#### `scripts/analysis/`
**Purpose**: Analysis and visualization scripts.

**Scripts**:
1. **`run_entropy_analysis.py`**: Compute entropy scores for slices
   - Processes volumes from Stage 4
   - Generates entropy JSON files in `data/entropy/`
   - Supports different modalities and axes (axial, coronal, sagittal)

2. **`visualize_entropy.py`**: Visualize entropy results

---

### `results/`
**Purpose**: Training results, evaluation metrics, and outputs.

**Structure**:
```
results/
├── MIL/
│   ├── runs/
│   │   └── fold_X/
│   │       └── YYYYMMDD_HHMMSS/
│   │           ├── checkpoints/
│   │           │   ├── best.pt      # Best model checkpoint
│   │           │   └── last.pt      # Last epoch checkpoint
│   │           ├── metrics/
│   │           │   ├── metrics.json           # Final evaluation metrics
│   │           │   └── threshold_analysis.json  # Optimal threshold analysis
│   │           ├── plots/
│   │           │   ├── training_curves.png
│   │           │   ├── confusion_matrix.png
│   │           │   └── roc_curve.png
│   │           ├── predictions/
│   │           │   ├── val_probs.npy
│   │           │   ├── val_preds.npy
│   │           │   └── val_labels.npy
│   │           ├── config/
│   │           │   └── config.json  # Training configuration
│   │           └── logs/
│   │               └── training.log
│   └── kfold_summary/
│       └── summary.json  # K-Fold cross-validation summary
```

**Metrics stored**:
- Accuracy, Precision, Recall, F1-score
- AUC-ROC
- Confusion matrix
- Optimal threshold (F1 maximization)
- Training history (loss, accuracy, learning rate)

---

## Content of Key Files

### Preprocessing Scripts

#### `scripts/preprocessing/run_stage1_n4.py`
**Purpose**: Apply N4 bias field correction to reduce intensity inhomogeneity.

**Key Functions**:
- `apply_n4_correction()`: Core N4 correction using SimpleITK
- `create_brain_mask()`: Create brain mask using Otsu thresholding
- `process_single_file()`: Process single NIfTI file with error handling
- `discover_patients()`: Discover patient folders and modality files

**Parameters** (from config):
- `max_iterations`: [40, 40, 30, 20] (per resolution level)
- `num_control_points`: 4 (B-spline control points)
- `convergence_threshold`: 0.001

**Output**: Bias-corrected NIfTI files in `data/processed/stage_1_n4/`

---

#### `scripts/preprocessing/run_stage2_zscore.py`
**Purpose**: Apply Z-score normalization to standardize intensity distributions.

**Key Functions**:
- `apply_zscore_normalization()`: Core normalization logic
  - Computes mean/std **only on brain voxels** (values > 0)
  - Formula: `normalized = (voxel - mean) / (std + eps)`
  - Preserves background (zeros remain zero)

**Parameters**:
- `eps`: 1e-8 (epsilon to avoid division by zero)

**Output**: Normalized NIfTI files in `data/processed/stage_2_zscore/`

---

#### `scripts/preprocessing/run_stage3_crop.py`
**Purpose**: Crop volumes to ROI (bounding box around brain).

**Key Functions**:
- `compute_bounding_box_from_mask()`: Compute bbox from binary mask
- `compute_bbox_from_volume()`: Compute bbox from image volume
- `compute_patient_bbox()`: Compute bbox for all modalities of a patient
  - Supports "reference_modality" (use single modality) or "union" (use all modalities)
- `apply_roi_crop()`: Apply cropping using precomputed bbox
- `update_origin_after_crop()`: Update image origin after cropping

**Parameters**:
- `padding`: 10 voxels (padding around bounding box)
- `eps_mask`: 1e-6 (threshold for brain mask)
- `bbox_mode`: "reference_modality" or "union"
- `reference_modality`: "flair" (if bbox_mode="reference_modality")

**Output**: Cropped NIfTI files in `data/processed/stage_3_crop/`

---

#### `scripts/preprocessing/run_stage4_resize.py`
**Purpose**: Resize volumes to fixed size (128x128x128).

**Key Functions**:
- `apply_resize()`: Core resizing using SimpleITK ResampleImageFilter
  - Linear interpolation (default)
  - Updates spacing to maintain physical dimensions
  - Preserves origin and direction

**Parameters**:
- `target_size`: [128, 128, 128] (x, y, z)
- `interpolation`: "linear" or "nearest"

**Output**: Resized NIfTI files in `data/processed/stage_4_resize/` (**used for training**)

---

### Model Training Scripts

#### `scripts/training/train_mil_multi_modal.py`
**Purpose**: Train Multi-Modality Dual-Stream MIL model (FLAIR + T1ce).

**Key Features**:
- **Entropy-based slice selection**: Always enabled (top-k slices per modality)
- **LDAM loss**: Label-Distribution-Aware Margin loss with DRW
- **Class balancing**: WeightedRandomSampler (inverse frequency)
- **Early stopping**: Monitors AUC, tie-breaker F1
- **Mixed precision**: Automatic Mixed Precision (AMP) for faster training
- **Gradient clipping**: Optional gradient norm clipping

**Training Loop**:
1. Load multi-modality datasets (FLAIR + T1ce)
2. Create DataLoaders with class balancing
3. Forward pass: Encode slices → MIL aggregation → Fusion → Classification
4. Backward pass: LDAM loss with DRW
5. Validation: Compute metrics (accuracy, precision, recall, F1, AUC)
6. Early stopping: Stop if no improvement for N epochs
7. Save best model checkpoint

**Output**: 
- Model checkpoints in `results/MIL/runs/fold_X/.../checkpoints/`
- Metrics in `results/MIL/runs/fold_X/.../metrics/`
- Plots in `results/MIL/runs/fold_X/.../plots/`

---

### Utility Scripts

#### `utils/augmentations_3d.py`
**Purpose**: 3D geometric data augmentation (Stage 5 - runtime-only).

**Augmentations Applied**:
1. **Random Rotation**: ±15 degrees around x, y, z axes (independently)
2. **Random Flip**: 50% probability per axis (x, y, z)
3. **Random Zoom**: ±10% scaling
4. **Random Translation**: ±10% of volume size

**Medical Rationale**:
- Preserves anatomical plausibility
- Mild augmentations to avoid unrealistic transformations
- Applied only during training (validation/test: no augmentation)

**Usage**: Applied dynamically in DataLoader when `mode="train"`

---

#### `utils/class_balancing.py`
**Purpose**: Class balancing via weighted sampling (Stage 6 - runtime-only).

**Implementation**:
- Uses `WeightedRandomSampler` from PyTorch
- Computes class weights: `weight = total_samples / (num_classes * class_count)`
- Minority class (LGG) gets higher weight
- Applied only during training (validation/test: no balancing)

**Usage**: Applied via `sampler` parameter in DataLoader

---

#### `utils/entropy_analysis.py`
**Purpose**: Compute entropy scores for slice informativeness (MIL-only).

**Algorithm**:
1. Extract 2D slices from 3D volume along specified axis (axial, coronal, sagittal)
2. For each slice:
   - Compute histogram (256 bins)
   - Normalize to probabilities
   - Compute Shannon entropy: `H = -Σ p * log2(p)`
3. Select top-k slices with highest entropy

**Output**: Entropy scores and top-k slice indices (stored in JSON)

---

## Directory and File Usage

### During Preprocessing

**Stage 1 (N4)**:
- **Input**: `data/raw/BraTS2018/<class>/<patient_id>/*.nii`
- **Output**: `data/processed/stage_1_n4/train/<class>/<patient_id>/*.nii.gz`

**Stage 2 (Z-score)**:
- **Input**: `data/processed/stage_1_n4/train/<class>/<patient_id>/*.nii.gz`
- **Output**: `data/processed/stage_2_zscore/train/<class>/<patient_id>/*.nii.gz`

**Stage 3 (Crop)**:
- **Input**: `data/processed/stage_2_zscore/train/<class>/<patient_id>/*.nii.gz`
- **Output**: `data/processed/stage_3_crop/train/<class>/<patient_id>/*.nii.gz`

**Stage 4 (Resize)**:
- **Input**: `data/processed/stage_3_crop/train/<class>/<patient_id>/*.nii.gz`
- **Output**: `data/processed/stage_4_resize/train/<class>/<patient_id>/*.nii.gz`
- **Usage**: **This is the final preprocessing stage used for training**

---

### During Training

**For ResNet50-3D and Swin UNETR**:
- **Data source**: `data/processed/stage_4_resize/train/`
- **Input**: Full 3D volumes (128x128x128)
- **Augmentation**: Applied dynamically (Stage 5)
- **Class balancing**: Applied dynamically (Stage 6)
- **No entropy**: These models process full volumes, not slices

**For MIL Models**:
- **Data source**: `data/processed/stage_4_resize/train/`
- **Entropy source**: `data/entropy/<patient_id>_entropy.json`
- **Input**: Top-k slices (2D) selected based on entropy scores
- **Augmentation**: Applied dynamically (Stage 5)
- **Class balancing**: Applied dynamically (Stage 6)
- **Entropy**: **Always enabled** for slice selection

---

### Temporary/Intermediate Files

**Stages 1-3**: Intermediate preprocessing stages
- Can be deleted after Stage 4 is complete (to save disk space)
- Useful for debugging and inspection

**Stages 5-6**: Runtime-only (no files created)
- Directories exist but are intentionally empty
- Applied dynamically during training

---

## Metadata and Results

### Performance Logs and Results

**Location**: `results/MIL/runs/fold_X/YYYYMMDD_HHMMSS/`

**Metrics Files**:
- `metrics/metrics.json`: Final evaluation metrics
  ```json
  {
    "accuracy": 0.85,
    "precision": 0.87,
    "recall": 0.83,
    "f1": 0.85,
    "auc": 0.92,
    "metrics_at_threshold_0.5": {...},
    "val_probs": [...],
    "val_labels": [...]
  }
  ```
- `metrics/threshold_analysis.json`: Optimal threshold analysis
  ```json
  {
    "best_threshold": 0.45,
    "optimization_metric": "f1",
    "tie_breaker": "recall",
    "threshold_sweep": [...],
    "metrics_at_best_threshold": {...}
  }
  ```

**Checkpoints**:
- `checkpoints/best.pt`: Best model (highest validation AUC)
- `checkpoints/last.pt`: Last epoch checkpoint

**Plots**:
- `plots/training_curves.png`: Training/validation loss and accuracy
- `plots/confusion_matrix.png`: Confusion matrix
- `plots/roc_curve.png`: ROC curve

**Predictions**:
- `predictions/val_probs.npy`: Validation probabilities
- `predictions/val_preds.npy`: Validation predictions (at optimal threshold)
- `predictions/val_labels.npy`: Validation ground truth labels

**Logs**:
- `logs/training.log`: Training log file

---

### Preprocessing Logs

**Location**: `logs/preprocessing/`

**Files**:
- `stage1_n4_YYYYMMDD_HHMMSS.log`: Stage 1 processing log
- `stage2_zscore_YYYYMMDD_HHMMSS.log`: Stage 2 processing log
- `stage3_crop_YYYYMMDD_HHMMSS.log`: Stage 3 processing log
- `stage4_resize_YYYYMMDD_HHMMSS.log`: Stage 4 processing log

**Manifest Files**:
- `data/processed/stage_X_*/manifest.jsonl`: Processing manifest (JSONL format)
  - Tracks processing status per file (success, failed, skipped)
  - Enables resumability

---

## Preprocessing Phases

### Stage 1: N4 Bias Field Correction

**Purpose**: Reduce intensity inhomogeneity caused by MRI scanner bias fields.

**Algorithm**:
1. Create brain mask using Otsu thresholding
2. Apply N4 bias field correction filter
3. Clamp negative values to 0
4. Preserve spacing, origin, and direction

**Implementation**: `scripts/preprocessing/run_stage1_n4.py`

**Parameters**:
- `max_iterations`: [40, 40, 30, 20] (per resolution level)
- `num_control_points`: 4
- `convergence_threshold`: 0.001

**Output**: Bias-corrected volumes in `data/processed/stage_1_n4/`

---

### Stage 2: Z-score Normalization

**Purpose**: Standardize intensity distributions across patients and modalities.

**Algorithm**:
1. Extract brain voxels (values > 0)
2. Compute mean and std on brain voxels only
3. Normalize: `(voxel - mean) / (std + eps)`
4. Preserve background (zeros remain zero)

**Implementation**: `scripts/preprocessing/run_stage2_zscore.py`

**Parameters**:
- `eps`: 1e-8 (epsilon to avoid division by zero)

**Output**: Normalized volumes in `data/processed/stage_2_zscore/`

---

### Stage 3: ROI Cropping

**Purpose**: Crop volumes to bounding box around brain to reduce computational cost.

**Algorithm**:
1. Compute bounding box from brain mask (all modalities use same bbox per patient)
2. Add padding (default: 10 voxels)
3. Crop volume to bounding box
4. Update image origin to reflect crop position

**Implementation**: `scripts/preprocessing/run_stage3_crop.py`

**Parameters**:
- `padding`: 10 voxels
- `eps_mask`: 1e-6 (threshold for brain mask)
- `bbox_mode`: "reference_modality" or "union"
- `reference_modality`: "flair" (if bbox_mode="reference_modality")

**Output**: Cropped volumes in `data/processed/stage_3_crop/`

---

### Stage 4: Resize to Fixed Volume

**Purpose**: Resize all volumes to fixed size (128x128x128) for batch processing.

**Algorithm**:
1. Compute new spacing: `new_spacing = old_spacing * (old_size / new_size)`
2. Resample using linear interpolation
3. Preserve origin and direction

**Implementation**: `scripts/preprocessing/run_stage4_resize.py`

**Parameters**:
- `target_size`: [128, 128, 128] (x, y, z)
- `interpolation`: "linear" or "nearest"

**Output**: Resized volumes in `data/processed/stage_4_resize/` (**used for training**)

---

### Stage 5: Data Augmentation (Runtime-Only)

**Purpose**: Apply geometric augmentations to increase data diversity.

**Augmentations**:
- Random rotation (±15 degrees)
- Random flip (x, y, z axes)
- Random zoom (±10%)
- Random translation (±10%)

**Implementation**: `utils/augmentations_3d.py`

**Usage**: Applied dynamically in DataLoader when `mode="train"`

**No disk output**: Augmentation happens in memory during training

---

### Stage 6: Class Balancing (Runtime-Only)

**Purpose**: Address class imbalance (HGG: 210 cases, LGG: 75 cases).

**Method**: WeightedRandomSampler with inverse frequency weighting

**Implementation**: `utils/class_balancing.py`

**Usage**: Applied via `sampler` parameter in training DataLoader

**No disk output**: Balancing happens at DataLoader level

---

### Stage 7: K-Fold Cross-Validation Splits

**Purpose**: Generate patient-level K-Fold splits for cross-validation.

**Algorithm**:
1. Build patient index from Stage 4 outputs (`build_stage4_index.py`)
2. Generate StratifiedKFold splits (k=5, seed=42) (`make_kfold_splits.py`)
3. Save split definitions as CSV files

**Implementation**: 
- `scripts/splits/build_stage4_index.py`
- `scripts/splits/make_kfold_splits.py`

**Output**:
- `data/index/stage4_index.csv`: Patient index
- `splits/fold_X_train.csv`: Training set for fold X
- `splits/fold_X_val.csv`: Validation set for fold X
- `splits/kfold_5fold_seed42.json`: K-Fold summary

**No image modification**: Only creates metadata files (CSV/JSON)

---

## Entropy Processing

### Purpose

Entropy-based slice selection identifies the most informative 2D slices from 3D MRI volumes for MIL models. This reduces computational cost and focuses on slices with high information content.

### Algorithm

1. **Extract Slices**: Extract 2D slices along specified axis (axial, coronal, sagittal)
2. **Compute Entropy**: For each slice:
   - Compute histogram (256 bins)
   - Normalize to probabilities
   - Compute Shannon entropy: `H = -Σ p * log2(p)`
3. **Select Top-K**: Select k slices with highest entropy scores

### Implementation

**Utility Module**: `utils/entropy_analysis.py`
- `compute_slice_entropy()`: Compute entropy for each slice
- `select_top_k_slices()`: Select top-k slices based on entropy

**Runner Script**: `scripts/analysis/run_entropy_analysis.py`
- Processes volumes from Stage 4
- Generates entropy JSON files per patient

### Output Format

**Location**: `data/entropy/<patient_id>_entropy.json`

**Structure**:
```json
{
  "patient_id": "Brats18_TCIA10_103_1",
  "modality": "flair",
  "axis": "axial",
  "entropy_scores": [0.85, 0.92, 0.78, ..., 0.65],
  "top_k_indices": [45, 67, 23, ..., 89],
  "top_k": 16,
  "num_slices": 128,
  "stats": {
    "mean": 0.75,
    "std": 0.12,
    "min": 0.45,
    "max": 0.95,
    "median": 0.72
  }
}
```

### Usage in Training

**MIL Models Only**:
- Entropy-based slice selection is **always enabled** for MIL models
- Each modality (FLAIR, T1ce) uses its own entropy JSON file
- Top-k slices are selected per modality independently
- Slices are extracted from Stage 4 volumes during dataset loading

**Not Used By**:
- ResNet50-3D: Processes full 3D volumes
- Swin UNETR: Processes full 3D volumes

### Why Entropy?

- **Information Content**: High entropy indicates more uniform distribution (more informative)
- **Computational Efficiency**: Reduces number of slices processed (from 128 to top-k, e.g., 16)
- **Focus on Relevant Slices**: Identifies slices with tumor regions or anatomical structures

---

## Training Strategy

### Overall Approach

1. **K-Fold Cross-Validation**: 5-fold stratified cross-validation (patient-level splitting)
2. **Class Balancing**: WeightedRandomSampler (inverse frequency weighting)
3. **Data Augmentation**: Geometric augmentations (rotation, flip, zoom, translation)
4. **Loss Function**: LDAM (Large Margin) loss with DRW (Deferred Re-Weighting)
5. **Early Stopping**: Monitors validation AUC (tie-breaker: F1)

---

### K-Fold Cross-Validation

**Method**: StratifiedKFold (k=5, seed=42)

**Features**:
- **Patient-level splitting**: Prevents data leakage (entire patient in one fold)
- **Stratified**: Preserves class ratio (HGG:LGG) in each fold
- **Reproducible**: Fixed random seed ensures consistent splits

**Implementation**:
1. Build patient index: `scripts/splits/build_stage4_index.py`
2. Generate splits: `scripts/splits/make_kfold_splits.py`
3. Train per fold: `scripts/training/run_mil_kfold.py`

**Output**: 
- `splits/fold_X_train.csv`: Training set for fold X
- `splits/fold_X_val.csv`: Validation set for fold X

---

### Class Balancing

**Method**: WeightedRandomSampler with inverse frequency weighting

**Strategy**: `weight = total_samples / (num_classes * class_count)`

**Example**:
- LGG: 75 cases → weight = 285 / (2 * 75) = 1.9
- HGG: 210 cases → weight = 285 / (2 * 210) = 0.68
- LGG (minority) gets higher weight → sampled more frequently

**Implementation**: `utils/class_balancing.py`

**Usage**: Applied only during training (validation/test: no balancing)

---

### Data Augmentation

**Augmentations**:
1. **Random Rotation**: ±15 degrees (x, y, z axes independently)
2. **Random Flip**: 50% probability per axis
3. **Random Zoom**: ±10% scaling
4. **Random Translation**: ±10% of volume size

**Medical Rationale**:
- Preserves anatomical plausibility
- Mild augmentations to avoid unrealistic transformations

**Implementation**: `utils/augmentations_3d.py`

**Usage**: Applied only during training (validation/test: no augmentation)

---

### Loss Function: LDAM + DRW

**LDAM (Large Margin) Loss**:
- Label-Distribution-Aware Margin loss
- Adds larger margin for minority class (LGG)
- Formula: `loss = -log(exp(s * (z_y - m_y)) / (exp(s * (z_y - m_y)) + Σ exp(s * z_j)))`
  - `m_y`: Class-dependent margin (larger for minority class)
  - `s`: Scaling factor (default: 30)

**DRW (Deferred Re-Weighting)**:
- Starts with uniform weighting
- Switches to class-weighted loss after N epochs (default: 15)
- Helps model learn general features before focusing on minority class

**Implementation**: `utils/ldam_loss.py`

**Parameters**:
- `max_m`: 0.5 (maximum margin)
- `s`: 30 (scaling factor)
- `drw_start_epoch`: 15 (epoch to start DRW)

---

### Early Stopping

**Monitor**: Validation AUC (primary), F1 (tie-breaker)

**Parameters**:
- `patience`: 7 epochs
- `min_epochs`: 10 epochs (minimum training before early stopping)
- `min_delta`: 0.0 (minimum improvement required)

**Implementation**: Custom `EarlyStopping` class in training scripts

---

## Data Handling

### For ResNet50-3D and Swin UNETR

**Input Format**: Full 3D volumes (128x128x128)

**Data Flow**:
1. Load volume from `data/processed/stage_4_resize/train/<class>/<patient_id>/*.nii.gz`
2. Apply augmentation (if training)
3. Convert to tensor: (1, 128, 128, 128) or (C, 128, 128, 128)
4. Feed to model

**No entropy**: These models process full volumes, not slices

---

### For MIL Models

**Input Format**: Bag of 2D slices (top-k slices per modality)

**Data Flow**:
1. Load volume from `data/processed/stage_4_resize/train/<class>/<patient_id>/*.nii.gz`
2. Load entropy JSON from `data/entropy/<patient_id>_entropy.json`
3. Extract top-k slices based on entropy scores
4. Apply augmentation (if training) to slices
5. Create bag: (num_slices, 1, H, W) per modality
6. Feed to model:
   - Encode slices → (num_slices, feature_dim)
   - MIL aggregation → (feature_dim)
   - Fusion (multi-modality) → (fusion_dim)
   - Classification → (num_classes)

**Entropy**: **Always enabled** for slice selection

**Multi-Modality MIL**:
- Separate entropy JSON files per modality (FLAIR, T1ce)
- Each modality selects its own top-k slices
- Fusion happens at bag (patient) level, not slice level

---

### Dataset Classes

**Single-Modality MIL**: `utils/mil_dataset.py`
- Loads single modality (e.g., FLAIR)
- Extracts top-k slices based on entropy

**Multi-Modality MIL**: `utils/mil_dataset_multi_modal.py`
- Loads multiple modalities (FLAIR, T1ce)
- Extracts top-k slices per modality independently
- Returns dictionary: `{modality: bag_tensor}`

---

### DataLoader Configuration

**Training**:
- `batch_size`: 2 (small due to memory constraints)
- `sampler`: WeightedRandomSampler (class balancing)
- `num_workers`: 4-8 (parallel data loading)
- `pin_memory`: True (faster GPU transfer)
- `drop_last`: True (for consistent batch sizes)

**Validation**:
- `batch_size`: 2
- `shuffle`: False
- `num_workers`: 4-8
- `pin_memory`: True
- `drop_last`: False

---

## Summary

This project implements a comprehensive pipeline for brain tumor classification using MRI images:

1. **Preprocessing**: 4 disk-based stages (N4, Z-score, Crop, Resize) + 2 runtime stages (Augmentation, Balancing)
2. **Models**: ResNet50-3D, Swin UNETR, and MIL (single/multi-modality)
3. **Training**: K-Fold cross-validation, LDAM loss, class balancing, early stopping
4. **Entropy**: MIL-specific slice selection based on informativeness
5. **Results**: Comprehensive metrics, plots, and predictions stored in `results/`

The pipeline is designed for reproducibility, scalability, and medical imaging best practices.

