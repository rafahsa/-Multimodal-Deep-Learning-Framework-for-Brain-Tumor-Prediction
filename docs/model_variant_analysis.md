# Model Variant Analysis for Grad-CAM Comparison

## Investigation Results

### ResNet50-3D Checkpoints Found

**Location**: `results/ResNet50-3D/runs/fold_X/run_YYYYMMDD_HHMMSS/checkpoints/`

**Structure**:
- Standard ResNet50-3D architecture
- Trained on: `data/processed/stage_4_resize/train/`
- Input: 4-channel multi-modal volumes (T1, T1ce, T2, FLAIR)
- Output: Binary classification (LGG vs HGG)

### ROI Experiment Context

**ROI MIL Experiment**: Only affects MIL model, not ResNet50-3D
- ROI MIL uses segmentation masks for slice selection
- ResNet50-3D remains unchanged in ROI ensemble
- Ensemble comparison shows identical ResNet50-3D probabilities

### Current Status

**Finding**: Only ONE ResNet50-3D model variant found
- Single checkpoint location: `results/ResNet50-3D/`
- Same model used in both baseline and ROI ensembles
- No evidence of separate "mri" vs "roi_mri" ResNet50-3D variants

## Assumptions for Implementation

Since only one ResNet50-3D variant exists, the implementation will:

1. **Support variant selection** for future extensibility
2. **Use same ResNet50-3D** for both "mri" and "roi_mri" variants
3. **Separate output directories** for comparison purposes
4. **Allow different checkpoint paths** if variants are added later

## Variant Configuration Table

| variant_name | checkpoint_path | input_type | notes |
|--------------|----------------|------------|-------|
| mri | results/ResNet50-3D/ | Full 3D volumes (128³) | Baseline variant |
| roi_mri | results/ResNet50-3D/ | Full 3D volumes (128³) | Same as mri (ROI only affects MIL) |

**Note**: Both variants use the same ResNet50-3D model. The "roi_mri" designation is for organizational purposes (separate output directories) to compare results in ROI ensemble context vs baseline context.

