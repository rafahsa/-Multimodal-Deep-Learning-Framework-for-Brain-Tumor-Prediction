# MIL Attention Analysis Plan

## Overview

This document outlines the plan for analyzing and comparing MIL attention behavior between baseline and ROI variants.

## Step 1: Model Variant Identification

### Baseline MIL
- **Checkpoint location**: `results/DualStreamMIL-3D/runs/fold_X/run_*/checkpoints/best*.pt`
- **Sampling strategy**: `random`, `entropy`, or `sequential`
- **Dataset**: Uses `MILSliceDataset` with standard sampling
- **Split files**: `splits/fold_X_train.csv`

### ROI MIL
- **Checkpoint location**: `results/DualStreamMIL-3D/runs/fold_X/run_*/checkpoints/best*.pt`
- **Sampling strategy**: `roi` (70% tumor, 30% context)
- **Dataset**: Uses `MILSliceDatasetROI` with ROI-guided sampling
- **Split files**: `splits/fold_X_train_with_seg.csv`

**Note**: Both variants are stored in the same directory structure. They are distinguished by the `sampling_strategy` field in `config.json`.

## Step 2: Attention Extraction

### Script: `scripts/analysis/extract_mil_attention.py`

**Features**:
- Loads MIL model checkpoints for both variants
- Extracts attention weights for each patient
- Computes slice-level attention rankings
- Optionally computes tumor overlap statistics

**Outputs per patient**:
- `attention_weights.csv`: Slice-level attention data
  - Columns: `slice_index`, `attention_weight`, `selection_weight`, `rank`, `z_coordinate`, `has_tumor`
- `metadata.json`: Patient-level metadata
  - Prediction info, attention statistics, overlap metrics

**Output structure**:
```
ensemble/results/interpretability/mil_attention/
├── baseline/
│   └── <patient_id>/
│       ├── attention_weights.csv
│       └── metadata.json
├── roi/
│   └── <patient_id>/
│       ├── attention_weights.csv
│       └── metadata.json
└── summary.json
```

## Step 3: Comparison & Visualization

### Metrics to Compare

1. **Attention Concentration**:
   - Mean attention weight
   - Attention entropy (diversity measure)
   - Top-k attention concentration

2. **Spatial Focus**:
   - Center of mass of attention
   - Attention spread (standard deviation)
   - Number of "active" slices (attention > threshold)

3. **Tumor Overlap** (if masks available):
   - % of top-k slices that contain tumor
   - Average attention on tumor vs non-tumor slices

### Visualization Script (TODO)

Create `scripts/analysis/visualize_mil_attention.py`:
- Attention bar plots (slice vs weight)
- Overlay attention on sample slices
- Comparison plots (baseline vs ROI)
- Tumor overlap heatmaps

## Step 4: Interpretation

### Questions to Answer

1. **Does ROI increase spatial focus?**
   - Compare attention entropy between variants
   - Check if ROI variant has more concentrated attention

2. **Does attention become more concentrated?**
   - Compare top-k attention weights
   - Check effective number of attended slices

3. **Is noise reduced?**
   - Compare attention on tumor vs non-tumor slices
   - Check overlap statistics

## Usage

### Extract Attention for Both Variants

```bash
python scripts/analysis/extract_mil_attention.py \
    --variant both \
    --fold 0 \
    --patient_ids_file data/selected_patients.txt \
    --output_dir ensemble/results/interpretability/mil_attention \
    --bag_size 64
```

### Extract for Single Variant

```bash
# Baseline only
python scripts/analysis/extract_mil_attention.py \
    --variant baseline \
    --fold 0 \
    --patient_ids_file data/selected_patients.txt

# ROI only
python scripts/analysis/extract_mil_attention.py \
    --variant roi \
    --fold 0 \
    --patient_ids_file data/selected_patients.txt
```

## Next Steps

1. ✅ Create attention extraction script
2. ⏳ Test on selected patients
3. ⏳ Create visualization script
4. ⏳ Generate comparison report
5. ⏳ Document findings

