# ROI-Based MIL Experiment: Implementation Summary

## Overview

This document describes the minimal changes needed to test ROI-based sampling in the MIL model using BraTS segmentation masks.

## Changes Made

### 1. Fold CSV Files with Segmentation Paths ✅

**Script:** `scripts/prepare_folds_with_seg.py`

**Output:** New CSV files in `splits/` with `_with_seg` suffix:
- `fold_0_train_with_seg.csv` through `fold_4_val_with_seg.csv`

**What it does:**
- Adds `path_seg` column pointing to `<class>/<patient_id>/<patient_id>_seg.nii.gz`
- Preserves all existing patient assignments (same folds)

### 2. ROI-Enabled MIL Dataset ✅

**File:** `utils/dataset_mil_roi.py`

**Key Features:**
- Extends `MILSliceDataset` with ROI sampling support
- New sampling strategy: `'roi'`
- 70% of instances from tumor region (seg > 0)
- 30% from context (near-tumor or whole brain)

**Usage:**
```python
from utils.dataset_mil_roi import MILSliceDatasetROI

dataset = MILSliceDatasetROI(
    data_root='data/processed/stage_4_resize/train',
    split_file='splits/fold_0_train_with_seg.csv',
    bag_size=32,  # Use same bag size as baseline
    sampling_strategy='roi',  # NEW: ROI-based sampling
    seg_data_root='data/raw/BraTS2018',  # Where seg masks are
    roi_tumor_ratio=0.7  # 70% from tumor
)
```

## Next Steps: Training

### Option 1: Minimal Wrapper Script

Create a wrapper that modifies the existing training script to use ROI dataset:

```python
# scripts/training/train_dual_stream_mil_roi.py
# (Minimal modification of train_dual_stream_mil.py)

# Change this line:
from utils.dataset_mil import MILSliceDataset

# To this:
from utils.dataset_mil_roi import MILSliceDatasetROI as MILSliceDataset

# Change split file paths:
train_split_file = splits_dir / f'fold_{current_fold}_train_with_seg.csv'
val_split_file = splits_dir / f'fold_{current_fold}_val_with_seg.csv'

# Change sampling strategy for training:
train_dataset = MILSliceDataset(
    ...
    sampling_strategy='roi',  # Use ROI sampling
    ...
)
```

### Option 2: Direct Modification

Modify `scripts/training/train_dual_stream_mil.py`:
1. Import ROI dataset: `from utils.dataset_mil_roi import MILSliceDatasetROI`
2. Use `_with_seg.csv` files
3. Set `sampling_strategy='roi'` for training dataset

## Training Command

```bash
python scripts/training/train_dual_stream_mil.py \
    --data-root data/processed/stage_4_resize/train \
    --splits-dir splits \
    --output-dir runs/mil_roi_experiment \
    --bag-size 32 \
    --sampling-strategy roi \
    --epochs 60 \
    --batch-size 4 \
    --lr 5e-5
```

**Note:** You'll need to modify the training script to:
- Use `fold_X_train_with_seg.csv` files
- Import `MILSliceDatasetROI`
- Set `sampling_strategy='roi'`

## Generating OOF Predictions

After training all 5 folds, generate OOF predictions using the existing script:

```bash
python scripts/ensemble/prepare_oof_predictions.py --model-name dualstream_mil_3d
```

This will create `ensemble/oof_predictions/dualstream_mil_3d_oof.csv` with ROI-guided MIL probabilities.

## Re-training Meta-Learner

1. Merge OOF predictions (if needed):
```bash
python scripts/ensemble/verify_and_merge_oof.py
```

2. Re-train meta-learner:
```bash
python scripts/ensemble/train_meta_learner.py --threshold 0.22
```

3. Compare coefficients:
```bash
python scripts/analysis/analyze_ensemble_contributions.py
```

## Expected Results

### Success Criteria:
- ✅ FP decreases by ≥1 (mean FP)
- ✅ MIL coefficient increases (from ~0.09 to higher)
- ✅ Recall ≥ 0.92 (no drop)
- ✅ FN does not increase significantly

### Comparison Metrics:
- Baseline MIL coefficient: ~0.09 (from deployed model)
- Target: MIL coefficient > 0.15 (indicating stronger contribution)

## Files Modified/Created

1. ✅ `scripts/prepare_folds_with_seg.py` - Adds seg paths to fold CSVs
2. ✅ `utils/dataset_mil_roi.py` - ROI-enabled MIL dataset
3. ⏳ `scripts/training/train_dual_stream_mil.py` - Needs modification to use ROI dataset
4. ⏳ Training runs - Generate OOF predictions
5. ⏳ Meta-learner re-training - Compare coefficients

## Notes

- Segmentation masks are loaded from `data/raw/BraTS2018/`
- ROI sampling falls back to random if segmentation not available
- Same bag size (32) and all other hyperparameters unchanged
- Same patient-level folds (no data leakage)

