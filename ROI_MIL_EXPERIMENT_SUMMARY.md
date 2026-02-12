# ROI-Based MIL Experiment: Implementation Summary

**Status:** ✅ **Infrastructure Ready** - Ready for training

---

## What Has Been Completed

### 1. ✅ Fold CSV Files with Segmentation Paths

**Script:** `scripts/prepare_folds_with_seg.py`  
**Output:** 10 new CSV files in `splits/`:
- `fold_0_train_with_seg.csv` through `fold_4_val_with_seg.csv`
- Each file has a `path_seg` column pointing to segmentation masks
- **Same patient assignments** as original folds (no data leakage)

**Verification:**
```bash
# Check that files were created
ls splits/*_with_seg.csv

# Verify segmentation paths
head -3 splits/fold_0_train_with_seg.csv
```

### 2. ✅ ROI-Enabled MIL Dataset

**File:** `utils/dataset_mil_roi.py`

**Features:**
- Extends `MILSliceDataset` with ROI sampling
- New `'roi'` sampling strategy:
  - **70%** of instances from tumor region (seg > 0)
  - **30%** from context (near-tumor or whole brain)
- Falls back to random sampling if segmentation not available
- **Same bag size and all other parameters** as baseline

**Key Methods:**
- `_load_segmentation_mask()` - Loads seg mask from path_seg
- `_get_roi_indices()` - Identifies tumor vs context slices
- `_sample_slices_roi()` - Implements 70/30 sampling strategy

---

## What Needs To Be Done

### Step 3: Train MIL with ROI Sampling

**Option A: Direct Modification (Recommended)**

Modify `scripts/training/train_dual_stream_mil.py`:

1. **Change import** (line ~30):
```python
# OLD:
from utils.dataset_mil import MILSliceDataset

# NEW:
from utils.dataset_mil_roi import MILSliceDatasetROI as MILSliceDataset
```

2. **Change split file paths** (lines ~1158-1159):
```python
# OLD:
train_split_file = splits_dir / f'fold_{current_fold}_train.csv'
val_split_file = splits_dir / f'fold_{current_fold}_val.csv'

# NEW:
train_split_file = splits_dir / f'fold_{current_fold}_train_with_seg.csv'
val_split_file = splits_dir / f'fold_{current_fold}_val_with_seg.csv'
```

3. **Set ROI sampling for training** (line ~1175):
```python
# Change:
sampling_strategy=args.sampling_strategy,

# To (or add --sampling-strategy roi to command):
sampling_strategy='roi',  # Force ROI sampling for this experiment
```

**Training Command:**
```bash
python scripts/training/train_dual_stream_mil.py \
    --data-root data/processed/stage_4_resize/train \
    --splits-dir splits \
    --output-dir runs/mil_roi_experiment \
    --bag-size 32 \
    --sampling-strategy roi \
    --epochs 60 \
    --batch-size 4 \
    --lr 5e-5 \
    --instance-encoder-backbone resnet18 \
    --instance-encoder-input-size 224
```

**Train all 5 folds** (remove `--single-fold` flag or train each fold separately).

### Step 4: Generate OOF Predictions

After training completes for all 5 folds:

```bash
python scripts/ensemble/prepare_oof_predictions.py --model-name dualstream_mil_3d
```

This will create/update `ensemble/oof_predictions/dualstream_mil_3d_oof.csv` with ROI-guided MIL probabilities.

**Note:** The script automatically picks the latest run per fold, so your ROI experiment runs should be selected.

### Step 5: Re-train Meta-Learner

1. **Merge OOF predictions** (if needed):
```bash
python scripts/ensemble/verify_and_merge_oof.py
```

2. **Re-train meta-learner**:
```bash
python scripts/ensemble/train_meta_learner.py --threshold 0.22
```

3. **Extract coefficients**:
```bash
python scripts/analysis/analyze_ensemble_contributions.py
```

This will show the new MIL coefficient and compare to baseline.

### Step 6: Compare Results

Fill in `reports/ROI_MIL_EXPERIMENT_REPORT_TEMPLATE.md` with:

1. **Ensemble metrics** from nested CV evaluation
2. **Meta-learner coefficients** (especially MIL coefficient)
3. **Comparison** vs baseline

---

## Success Criteria

| Criterion | Target | How to Check |
|-----------|--------|--------------|
| **FP Reduction** | Decrease by ≥1 (mean) | Compare FP_mean in nested CV results |
| **MIL Coefficient** | Increase (target >0.15) | Compare from `analyze_ensemble_contributions.py` |
| **Recall** | ≥ 0.92 (no drop) | Compare Recall_mean in nested CV results |
| **FN** | No significant increase | Compare FN_mean in nested CV results |

---

## Files Created/Modified

### ✅ Completed:
1. `scripts/prepare_folds_with_seg.py` - Adds seg paths to fold CSVs
2. `utils/dataset_mil_roi.py` - ROI-enabled MIL dataset
3. `splits/*_with_seg.csv` - Fold CSVs with segmentation paths
4. `docs/ROI_MIL_EXPERIMENT.md` - Detailed implementation guide
5. `reports/ROI_MIL_EXPERIMENT_REPORT_TEMPLATE.md` - Results template

### ⏳ To Do:
1. Modify `scripts/training/train_dual_stream_mil.py` (3 small changes)
2. Train MIL model (5 folds)
3. Generate OOF predictions
4. Re-train meta-learner
5. Compare results

---

## Quick Start

1. **Modify training script** (3 lines as shown above)

2. **Train one fold** (for testing):
```bash
python scripts/training/train_dual_stream_mil.py \
    --data-root data/processed/stage_4_resize/train \
    --splits-dir splits \
    --output-dir runs/mil_roi_test \
    --bag-size 32 \
    --sampling-strategy roi \
    --single-fold \
    --fold 0 \
    --epochs 10  # Quick test
```

3. **Verify ROI sampling works** (check logs for "ROI sampling" messages)

4. **Train all folds** (remove `--single-fold --fold 0`)

5. **Generate OOF and re-train meta-learner** (steps 4-5 above)

---

## Expected Timeline

- **Training (5 folds):** ~2-4 hours (depending on GPU)
- **OOF generation:** ~5 minutes
- **Meta-learner re-training:** ~1 minute
- **Analysis:** ~5 minutes

**Total:** ~3-5 hours for complete experiment

---

## Notes

- **No architecture changes** - Only sampling strategy changed
- **Same hyperparameters** - Bag size, learning rate, etc. unchanged
- **Same folds** - Patient-level splits preserved (no leakage)
- **Minimal code changes** - Only 3 lines in training script

---

*Ready for execution. See `docs/ROI_MIL_EXPERIMENT.md` for detailed technical notes.*

