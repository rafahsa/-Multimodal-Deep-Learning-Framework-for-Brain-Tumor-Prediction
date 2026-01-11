# Brain Tumor Classification Project - Complete File & Folder Structure

## Project Overview

This project implements a deep learning pipeline for brain tumor grade classification (LGG vs HGG) using 3D MRI volumes from the BraTS 2018 dataset. The project follows a modular architecture with clear separation between data processing, model training, evaluation, and ensemble methods.

---

## Complete Directory Tree

```
brain_tumor_project/
│
├── 📄 README.md                          # Main project documentation
├── 📄 PROJECT_OVERVIEW.md                # Detailed project overview
├── 📄 PROJECT_STRUCTURE.md               # This file - complete structure guide
├── 📄 .gitignore                         # Git ignore rules
├── 📄 brain_tumor_project.code-workspace # VS Code workspace configuration
├── 📄 VS_CODE_WORKSPACE_SETUP.md        # VS Code setup instructions
│
├── 📁 data/                              # All data (raw and processed)
│   ├── raw/                              # Original, unmodified BraTS 2018 data
│   │   └── BraTS2018/
│   │       ├── HGG/                      # High-Grade Glioma (210 cases)
│   │       └── LGG/                      # Low-Grade Glioma (75 cases)
│   │
│   ├── brats2018/                        # Alternative data location (symlinks/aliases)
│   │
│   ├── processed/                        # Stage-wise processed data
│   │   ├── stage_0_raw/                  # Raw data reference
│   │   ├── stage_1_n4/                   # N4 bias field correction outputs
│   │   ├── stage_2_zscore/               # Z-score normalization outputs
│   │   ├── stage_3_crop/                 # ROI cropping outputs
│   │   ├── stage_4_resize/               # Resized to 128x128x128 (used for training)
│   │   ├── stage_5_augmented/            # (Runtime-only: intentionally empty)
│   │   └── stage_6_balanced/             # (Runtime-only: intentionally empty)
│   │
│   ├── entropy/                          # Entropy-based slice selection metadata (MIL-only)
│   │   └── <patient_id>_entropy.json     # Per-patient entropy scores and top-k slice indices
│   │
│   ├── entropy_results/                  # Entropy analysis results and statistics
│   │
│   ├── index/                            # Patient index files
│   │   └── stage4_index.csv              # Complete index of all patients after Stage 4
│   │
│   ├── kfold_splits/                     # K-Fold split definitions (alternative location)
│   │
│   └── README.md                         # Data directory documentation
│
├── 📁 models/                            # Model architecture definitions
│   ├── resnet50_3d_fast/                 # ResNet50-3D model package
│   │   ├── __init__.py                   # Package initialization
│   │   └── model.py                      # ResNet50-3D architecture implementation
│   │
│   ├── swin_unetr_encoder.py             # Swin UNETR encoder for classification
│   ├── dual_stream_mil.py                # Dual-Stream Multiple Instance Learning model
│   └── __pycache__/                      # Python cache files
│
├── 📁 preprocessing/                     # Preprocessing utilities
│   └── 01_n4_bias.py                     # N4 bias correction utility functions
│
├── 📁 scripts/                           # All executable scripts
│   │
│   ├── preprocessing/                    # Data preprocessing pipeline scripts
│   │   ├── run_stage1_n4.py              # Stage 1: N4 bias field correction
│   │   ├── run_stage2_zscore.py          # Stage 2: Z-score normalization
│   │   ├── run_stage3_crop.py            # Stage 3: ROI cropping
│   │   └── run_stage4_resize.py          # Stage 4: Resize to fixed volume
│   │
│   ├── training/                         # Model training scripts
│   │   ├── train_resnet50_3d.py          # Train ResNet50-3D model
│   │   ├── train_swin_unetr_3d.py        # Train Swin UNETR model
│   │   ├── train_mil.py                  # Train single-modality MIL model
│   │   ├── train_mil_multi_modal.py      # Train multi-modality MIL model
│   │   ├── train_dual_stream_mil.py      # Train dual-stream MIL model
│   │   └── run_mil_kfold.py              # K-Fold cross-validation runner for MIL
│   │
│   ├── evaluation/                       # Model evaluation scripts (empty, to be implemented)
│   │
│   ├── ensemble/                         # Ensemble method scripts
│   │   ├── prepare_oof_predictions.py    # Prepare out-of-fold predictions
│   │   ├── verify_and_merge_oof.py       # Verify and merge OOF predictions
│   │   ├── verify_merged_oof_for_training.py  # Verify merged OOF for meta-learner
│   │   ├── train_meta_learner.py         # Train meta-learner (stacking)
│   │   ├── test_ensemble_on_new_patients.py  # Test ensemble on new patients
│   │   └── generate_visualizations.py    # Generate ensemble visualizations
│   │
│   ├── splits/                           # Data splitting scripts
│   │   ├── build_stage4_index.py         # Build patient index from Stage 4 outputs
│   │   └── make_kfold_splits.py          # Generate K-Fold cross-validation splits
│   │
│   ├── analysis/                         # Analysis and visualization scripts
│   │   ├── run_entropy_analysis.py       # Compute entropy scores for slices
│   │   ├── run_entropy_for_fold.py       # Compute entropy for specific fold
│   │   └── visualize_entropy.py          # Visualize entropy results
│   │
│   ├── utils/                            # Script-specific utility functions
│   │   ├── aggregate_mil_results.py      # Aggregate MIL model results across folds
│   │   └── fix_swinunetr_metrics.py      # Fix Swin UNETR metrics computation
│   │
│   └── README.md                         # Scripts directory documentation
│
├── 📁 utils/                             # Utility functions and helper modules
│   ├── augmentations_3d.py               # 3D geometric data augmentation (Stage 5)
│   ├── augmentations_2d.py               # 2D augmentation for MIL slice processing
│   ├── class_balancing.py                # Class balancing utilities (Stage 6)
│   ├── dataset_3d.py                     # PyTorch Dataset for 3D volumes
│   ├── dataset_3d_multi_modal.py         # PyTorch Dataset for multi-modal 3D volumes
│   ├── mil_dataset.py                    # PyTorch Dataset for single-modality MIL
│   ├── mil_dataset_multi_modal.py        # PyTorch Dataset for multi-modality MIL
│   ├── dataset_mil.py                    # Alternative MIL dataset implementation
│   ├── entropy_analysis.py               # Entropy computation for slice selection
│   ├── ldam_loss.py                      # LDAM (Large Margin) loss function
│   └── __pycache__/                      # Python cache files
│
├── 📁 configs/                           # Configuration files (YAML)
│   ├── stage_1_n4.yaml                   # Stage 1 preprocessing configuration
│   ├── stage_2_zscore.yaml               # Stage 2 preprocessing configuration
│   ├── stage_3_crop.yaml                 # Stage 3 preprocessing configuration
│   ├── stage_4_resize.yaml               # Stage 4 preprocessing configuration
│   └── README.md                         # Configuration files documentation
│
├── 📁 splits/                            # K-Fold cross-validation split definitions
│   ├── kfold_5fold_seed42.json           # K-Fold summary (5 folds, seed=42)
│   ├── fold_0_train.csv                  # Training set for fold 0
│   ├── fold_0_val.csv                    # Validation set for fold 0
│   ├── fold_1_train.csv                  # Training set for fold 1
│   ├── fold_1_val.csv                    # Validation set for fold 1
│   ├── fold_2_train.csv                  # Training set for fold 2
│   ├── fold_2_val.csv                    # Validation set for fold 2
│   ├── fold_3_train.csv                  # Training set for fold 3
│   ├── fold_3_val.csv                    # Validation set for fold 3
│   ├── fold_4_train.csv                  # Training set for fold 4
│   └── fold_4_val.csv                    # Validation set for fold 4
│
├── 📁 pretrained/                        # Pretrained model weights
│   ├── medicalnet_resnet50_3d.pth        # MedicalNet pretrained ResNet50-3D weights
│   └── README.md                         # Pretrained models documentation
│
├── 📁 ensemble/                          # Ensemble method outputs and models
│   ├── models/                           # Trained meta-learner models
│   ├── oof_predictions/                  # Out-of-fold predictions from base models
│   ├── results/                          # Ensemble evaluation results
│   ├── visualizations/                   # Ensemble visualizations
│   └── README.md                         # Ensemble documentation
│
├── 📁 results/                           # Training results and evaluation outputs
│   ├── ResNet50-3D/                      # ResNet50-3D experiment results
│   │   └── runs/
│   │       └── fold_X/                   # Per-fold results
│   │           └── YYYYMMDD_HHMMSS/
│   │               ├── checkpoints/      # Model checkpoints
│   │               ├── metrics/          # Evaluation metrics (JSON)
│   │               ├── plots/            # Training curves, confusion matrices
│   │               └── predictions/      # Prediction outputs
│   │
│   ├── Swin_UNETR/                       # Swin UNETR experiment results
│   ├── SwinUNETR-3D/                     # Alternative Swin UNETR results
│   ├── MIL/                              # Single-modality MIL results
│   │   └── runs/
│   │       └── fold_X/                   # Per-fold results
│   │           └── YYYYMMDD_HHMMSS/
│   │               ├── checkpoints/
│   │               ├── metrics/
│   │               ├── plots/
│   │               └── predictions/
│   │
│   ├── DualStreamMIL-3D/                 # Dual-Stream MIL results
│   └── entropy_visualization/            # Entropy analysis visualizations
│
├── 📁 experiments/                       # Experiment tracking
│   ├── resnet50_3d/                      # ResNet50-3D experiments
│   ├── swin_unetr/                       # Swin UNETR experiments
│   ├── mil/                              # MIL experiments
│   └── README.md                         # Experiments documentation
│
├── 📁 logs/                              # Training and preprocessing logs
│   ├── preprocessing/                    # Preprocessing stage logs
│   │   └── stageX_*.log                  # Per-stage processing logs
│   ├── training/                         # Training logs
│   │   └── *.log                         # Training run logs
│   ├── evaluation/                       # Evaluation logs
│   └── README.md                         # Logs directory documentation
│
├── 📁 docs/                              # Project documentation
│   ├── stage1_n4_preprocessing.md        # Stage 1 documentation
│   ├── stage2_zscore_preprocessing.md    # Stage 2 documentation
│   ├── stage3_crop_preprocessing.md      # Stage 3 documentation
│   ├── stage4_resize_preprocessing.md    # Stage 4 documentation
│   ├── stage5_augmentation.md            # Stage 5 documentation
│   ├── stage6_class_balancing.md         # Stage 6 documentation
│   ├── stage7_kfold.md                   # Stage 7 (K-Fold) documentation
│   ├── stage_entropy_mil.md              # Entropy analysis documentation
│   │
│   ├── resnet50_3d_training.md           # ResNet50-3D training guide
│   ├── resnet50_3d_optimizations.md      # ResNet50-3D optimizations
│   ├── resnet50_3d_training_fixes.md     # ResNet50-3D bug fixes
│   ├── resnet50_3d_multimodal.md         # ResNet50-3D multimodal usage
│   │
│   ├── swin_unetr_classification_proposal.md  # Swin UNETR proposal
│   │
│   ├── mil_training.md                   # MIL training guide
│   ├── mil_slice_selection_analysis.md   # MIL slice selection analysis
│   ├── mil_performance_analysis.md       # MIL performance analysis
│   ├── mil_overfitting_analysis_and_solution.md  # MIL overfitting solutions
│   ├── mil_anti_overfitting_implementation_summary.md  # MIL anti-overfitting
│   ├── mil_optimal_solution_implementation.md  # MIL optimal solution
│   ├── mil_final_training_guide.md       # MIL final training guide
│   ├── dual_stream_mil_design.md         # Dual-Stream MIL design
│   ├── dual_stream_mil_implementation_summary.md  # Dual-Stream MIL implementation
│   ├── dual_stream_mil_loss_analysis.md  # Dual-Stream MIL loss analysis
│   │
│   ├── ensemble_stacking_methodology.md  # Ensemble stacking methodology
│   ├── ensemble_stacking_methodology منهجية الدمج.md  # Ensemble (Arabic)
│   ├── ensemble_implementation_plan.md   # Ensemble implementation plan
│   ├── ensemble_implementation_summary.md  # Ensemble implementation summary
│   ├── ensemble_training_readiness_report.md  # Ensemble training readiness
│   ├── ensemble_meta_learner_training_report.md  # Meta-learner training report
│   ├── ensemble_oof_verification_report.md  # OOF verification report
│   ├── ensemble_test_patients_guide.md   # Ensemble testing guide
│   ├── ensemble_test_script_fixes.md     # Ensemble test fixes
│   ├── ensemble_visualizations_summary.md  # Ensemble visualizations
│   │
│   ├── training_journey_summary ملخص اول موديل.md  # Training journey (Arabic)
│   ├── training_journey_summary_SwinUNETR-3Dملخص تاني موديل.md  # Swin UNETR journey
│   ├── training_journey_summary_DualStreamMIL-3Dملخص ثالث موديل.md  # Dual-Stream MIL journey
│   │
│   ├── entropy_visualization.md          # Entropy visualization guide
│   ├── medicalnet_integration.md         # MedicalNet integration guide
│   └── training_strategy_analysis.md     # Training strategy analysis
│
├── 📁 notebooks/                         # Jupyter notebooks (minimal, for exploration)
│
├── 📁 test/                              # Test data and scripts
│   └── DATA_FOR_TEST/                    # Test patient data
│
└── 📁 __pycache__/                       # Python cache files (if any in root)
```

---

## Directory Descriptions

### 🗂️ Root Level Files

#### `README.md`
Main project documentation containing:
- Project overview and goals
- Dataset information (BraTS 2018)
- Model architectures (ResNet50-3D, Swin UNETR, MIL)
- Preprocessing pipeline stages
- Usage instructions
- Reproducibility notes

#### `PROJECT_OVERVIEW.md`
Comprehensive technical overview with:
- Detailed project structure
- Preprocessing phase descriptions
- Training strategy explanations
- Data handling workflows
- Model-specific documentation

#### `PROJECT_STRUCTURE.md`
This file - complete file and folder structure reference guide.

#### `.gitignore`
Git ignore rules excluding:
- Large data files (`data/raw/`, `data/processed/`)
- Logs and checkpoints
- Python cache files
- IDE configurations

---

### 📁 `data/` - Data Directory

**Purpose**: Centralized location for all data (raw, processed, and metadata).

#### `data/raw/`
- **Contents**: Original BraTS 2018 dataset (never modified)
- **Structure**: Organized by tumor grade (HGG/LGG), then by patient ID
- **Format**: NIfTI files (.nii/.nii.gz) with 4 modalities (T1, T1CE, T2, FLAIR) + segmentation masks

#### `data/processed/`
- **Contents**: Stage-wise processed data outputs
- **Stage 1-4**: Disk-based preprocessing (persistent files)
  - `stage_1_n4/`: N4 bias field correction outputs
  - `stage_2_zscore/`: Z-score normalized volumes
  - `stage_3_crop/`: ROI-cropped volumes
  - `stage_4_resize/`: Final preprocessed volumes (128x128x128) - **used for training**
- **Stage 5-6**: Runtime-only stages (directories intentionally empty)
  - `stage_5_augmented/`: Placeholder (augmentation applied in-memory during training)
  - `stage_6_balanced/`: Placeholder (class balancing applied via sampling)

#### `data/entropy/`
- **Purpose**: Entropy-based slice informativeness metadata (MIL models only)
- **Format**: JSON files per patient containing entropy scores and top-k slice indices
- **Usage**: Used by MIL models to select most informative 2D slices from 3D volumes

#### `data/index/`
- **Contents**: Patient index files
- **File**: `stage4_index.csv` - Complete index of all patients after Stage 4 preprocessing
- **Columns**: patient_id, class, class_label, path_t1, path_t1ce, path_t2, path_flair

---

### 📁 `models/` - Model Architectures

**Purpose**: Model architecture definitions implemented in PyTorch.

#### `models/resnet50_3d_fast/`
- **Purpose**: ResNet50-3D model package
- **File**: `model.py` - 3D ResNet50 implementation for full volume classification
- **Input**: Full 3D volumes (128x128x128)
- **Pretrained**: Can load MedicalNet pretrained weights

#### `models/swin_unetr_encoder.py`
- **Purpose**: Swin UNETR encoder adapted for classification
- **Input**: Full 3D volumes (128x128x128)
- **Architecture**: Transformer-based encoder with patch embedding

#### `models/dual_stream_mil.py`
- **Purpose**: Dual-Stream Multiple Instance Learning model
- **Features**: Separate encoders per modality (FLAIR, T1ce), attention-based aggregation, fusion at bag level
- **Input**: Top-k 2D slices per modality (selected via entropy)

---

### 📁 `scripts/` - Executable Scripts

**Purpose**: All runnable scripts organized by functionality.

#### `scripts/preprocessing/`
Preprocessing pipeline scripts executed sequentially:
1. **`run_stage1_n4.py`**: N4 bias field correction using SimpleITK
2. **`run_stage2_zscore.py`**: Z-score normalization (brain voxels only)
3. **`run_stage3_crop.py`**: ROI cropping with bounding box computation
4. **`run_stage4_resize.py`**: Resize to fixed volume (128x128x128)

Each script:
- Reads configuration from `configs/`
- Supports parallel processing
- Generates manifest files for tracking
- Creates logs in `logs/preprocessing/`

#### `scripts/training/`
Model training scripts:
- **`train_resnet50_3d.py`**: Train ResNet50-3D model
- **`train_swin_unetr_3d.py`**: Train Swin UNETR model
- **`train_mil.py`**: Train single-modality MIL model
- **`train_mil_multi_modal.py`**: Train multi-modality MIL model
- **`train_dual_stream_mil.py`**: Train dual-stream MIL model
- **`run_mil_kfold.py`**: K-Fold cross-validation runner for MIL

#### `scripts/ensemble/`
Ensemble method implementation:
- **`prepare_oof_predictions.py`**: Prepare out-of-fold predictions from base models
- **`verify_and_merge_oof.py`**: Verify and merge OOF predictions across folds
- **`train_meta_learner.py`**: Train meta-learner (stacking) using OOF predictions
- **`test_ensemble_on_new_patients.py`**: Test ensemble on new patient data
- **`generate_visualizations.py`**: Generate ensemble performance visualizations

#### `scripts/splits/`
Data splitting utilities:
- **`build_stage4_index.py`**: Build comprehensive patient index from Stage 4 outputs
- **`make_kfold_splits.py`**: Generate stratified K-Fold splits (k=5, seed=42)

#### `scripts/analysis/`
Analysis and visualization:
- **`run_entropy_analysis.py`**: Compute entropy scores for slice selection
- **`run_entropy_for_fold.py`**: Compute entropy for specific fold
- **`visualize_entropy.py`**: Visualize entropy analysis results

---

### 📁 `utils/` - Utility Modules

**Purpose**: Reusable utility functions used across the project.

#### Core Utilities:
- **`augmentations_3d.py`**: 3D geometric augmentation transforms (Stage 5)
  - Random rotation, flip, zoom, translation
  - Medical-safe augmentations using MONAI
  
- **`augmentations_2d.py`**: 2D augmentation for MIL slice processing

- **`class_balancing.py`**: Class balancing utilities (Stage 6)
  - WeightedRandomSampler implementation
  - Inverse frequency weighting

- **`entropy_analysis.py`**: Entropy computation for slice informativeness
  - Shannon entropy calculation per slice
  - Top-k slice selection

- **`ldam_loss.py`**: LDAM (Large Margin) loss function
  - Label-Distribution-Aware Margin loss
  - Deferred Re-Weighting (DRW) support

#### Dataset Classes:
- **`dataset_3d.py`**: PyTorch Dataset for 3D volume loading
- **`dataset_3d_multi_modal.py`**: PyTorch Dataset for multi-modal 3D volumes
- **`mil_dataset.py`**: PyTorch Dataset for single-modality MIL
- **`mil_dataset_multi_modal.py`**: PyTorch Dataset for multi-modality MIL

---

### 📁 `configs/` - Configuration Files

**Purpose**: YAML configuration files for preprocessing stages.

- **`stage_1_n4.yaml`**: N4 bias correction parameters
- **`stage_2_zscore.yaml`**: Z-score normalization parameters
- **`stage_3_crop.yaml`**: ROI cropping parameters (padding, bbox mode)
- **`stage_4_resize.yaml`**: Resize parameters (target size, interpolation)

All configs are human-readable YAML format for easy parameter tuning.

---

### 📁 `splits/` - K-Fold Split Definitions

**Purpose**: Patient-level K-Fold cross-validation split definitions.

**Files**:
- **`kfold_5fold_seed42.json`**: Summary of K-Fold configuration
- **`fold_X_train.csv`**: Training set for fold X (X = 0-4)
- **`fold_X_val.csv`**: Validation set for fold X

**Features**:
- Stratified splitting (preserves class ratio)
- Patient-level (prevents data leakage)
- Reproducible (seed=42)

---

### 📁 `pretrained/` - Pretrained Models

**Purpose**: Pretrained model weights for transfer learning.

- **`medicalnet_resnet50_3d.pth`**: MedicalNet pretrained ResNet50-3D weights
- Used for initializing ResNet50-3D models

---

### 📁 `results/` - Training Results

**Purpose**: Training results, checkpoints, metrics, and visualizations.

**Structure** (per model type):
```
results/
└── <ModelType>/
    └── runs/
        └── fold_X/
            └── YYYYMMDD_HHMMSS/
                ├── checkpoints/      # Model checkpoints (best.pt, last.pt)
                ├── metrics/          # Evaluation metrics (JSON)
                ├── plots/            # Training curves, confusion matrices, ROC
                ├── predictions/      # Prediction outputs (numpy arrays)
                └── logs/             # Training logs
```

**Model Types**:
- `ResNet50-3D/`
- `Swin_UNETR/` or `SwinUNETR-3D/`
- `MIL/`
- `DualStreamMIL-3D/`

---

### 📁 `ensemble/` - Ensemble Outputs

**Purpose**: Ensemble method outputs and meta-learner models.

- **`models/`**: Trained meta-learner models
- **`oof_predictions/`**: Out-of-fold predictions from base models
- **`results/`**: Ensemble evaluation results
- **`visualizations/`**: Ensemble performance visualizations

---

### 📁 `experiments/` - Experiment Tracking

**Purpose**: Experiment tracking and organization (optional structure for experiment management).

- `resnet50_3d/`: ResNet50-3D experiment outputs
- `swin_unetr/`: Swin UNETR experiment outputs
- `mil/`: MIL experiment outputs

---

### 📁 `logs/` - Logs

**Purpose**: Training and preprocessing logs.

**Structure**:
- `preprocessing/`: Preprocessing stage logs (e.g., `stage1_n4_YYYYMMDD_HHMMSS.log`)
- `training/`: Training run logs
- `evaluation/`: Evaluation logs

---

### 📁 `docs/` - Documentation

**Purpose**: Comprehensive project documentation organized by topic.

**Categories**:
- **Preprocessing**: Stage-by-stage preprocessing documentation
- **Model Training**: Training guides for each model type
- **MIL Models**: Extensive MIL documentation (overfitting, optimizations, etc.)
- **Ensemble**: Ensemble methodology and implementation guides
- **Training Journeys**: Summaries of training experiences (some in Arabic)
- **Technical Guides**: Entropy analysis, MedicalNet integration, etc.

---

## Key Workflows

### 1. Data Preprocessing Workflow

```
data/raw/BraTS2018/
  ↓ (Stage 1: N4)
data/processed/stage_1_n4/
  ↓ (Stage 2: Z-score)
data/processed/stage_2_zscore/
  ↓ (Stage 3: Crop)
data/processed/stage_3_crop/
  ↓ (Stage 4: Resize)
data/processed/stage_4_resize/  ← Used for training
```

### 2. Training Workflow

```
Stage 4 Data
  ↓
K-Fold Splits (splits/)
  ↓
Training Scripts (scripts/training/)
  ↓
Results (results/<ModelType>/)
```

### 3. MIL-Specific Workflow

```
Stage 4 Data
  ↓
Entropy Analysis (scripts/analysis/run_entropy_analysis.py)
  ↓
data/entropy/ (JSON files)
  ↓
MIL Training (scripts/training/train_mil*.py)
  ↓
Results (results/MIL/)
```

### 4. Ensemble Workflow

```
Base Model Results (results/)
  ↓
OOF Predictions (scripts/ensemble/prepare_oof_predictions.py)
  ↓
Merge OOF (scripts/ensemble/verify_and_merge_oof.py)
  ↓
Train Meta-Learner (scripts/ensemble/train_meta_learner.py)
  ↓
Test Ensemble (scripts/ensemble/test_ensemble_on_new_patients.py)
  ↓
Ensemble Results (ensemble/results/)
```

---

## File Naming Conventions

### Data Files
- **Raw data**: `<patient_id>_<modality>.nii` or `.nii.gz`
- **Processed data**: `<patient_id>_<modality>.nii.gz`
- **Entropy files**: `<patient_id>_entropy.json`

### Model Files
- **Checkpoints**: `best.pt`, `last.pt`
- **Pretrained**: `<model_name>.pth`

### Split Files
- **CSV splits**: `fold_X_train.csv`, `fold_X_val.csv`
- **JSON summary**: `kfold_<k>fold_seed<seed>.json`

### Log Files
- **Preprocessing**: `stageX_<description>_YYYYMMDD_HHMMSS.log`
- **Training**: `training_YYYYMMDD_HHMMSS.log`

### Result Files
- **Metrics**: `metrics.json`, `threshold_analysis.json`
- **Plots**: `training_curves.png`, `confusion_matrix.png`, `roc_curve.png`
- **Predictions**: `val_probs.npy`, `val_preds.npy`, `val_labels.npy`

---

## Important Notes

### Runtime-Only Stages
- **Stage 5 (Augmentation)** and **Stage 6 (Balancing)** do NOT create files on disk
- They are applied dynamically during DataLoader iteration
- Directories exist but are intentionally empty

### Data Persistence
- All data is stored on persistent volume (`/workspace/`) that survives pod restarts
- Raw data is NEVER modified (read-only)
- All processed outputs go to `data/processed/`

### Model Input Formats
- **ResNet50-3D & Swin UNETR**: Full 3D volumes (128x128x128)
- **MIL Models**: Top-k 2D slices (selected via entropy) per modality

### Reproducibility
- Fixed random seeds (seed=42 for K-Fold)
- Configuration files for all preprocessing stages
- Deterministic augmentation when applicable

---

## Getting Started

1. **Preprocessing**: Run scripts in `scripts/preprocessing/` sequentially (Stage 1 → 4)
2. **Generate Splits**: Run `scripts/splits/build_stage4_index.py` then `scripts/splits/make_kfold_splits.py`
3. **Train Models**: Run appropriate training script from `scripts/training/`
4. **Evaluate**: Check results in `results/<ModelType>/`
5. **Ensemble** (optional): Follow ensemble workflow in `scripts/ensemble/`

For detailed instructions, see individual README files in each directory and documentation in `docs/`.

