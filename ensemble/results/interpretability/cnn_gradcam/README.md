# 3D Grad-CAM Heatmaps for ResNet50-3D

This directory contains Grad-CAM heatmaps generated for ResNet50-3D model predictions.

## What is Grad-CAM?

Gradient-weighted Class Activation Mapping (Grad-CAM) is a technique for visualizing which spatial regions in the input volume are most important for the model's prediction.

**How it works:**
1. Captures activations from the last convolutional layer (`layer4`) before global pooling
2. Computes gradients of the target class logit with respect to these activations
3. Computes channel-wise weights as the mean of gradients over spatial dimensions
4. Generates a weighted activation map (CAM)
5. Applies ReLU and normalizes to [0, 1]
6. Upsamples to input resolution (128×128×128) if needed

**Reference:** Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks"

## Usage

### Basic Usage

```bash
python scripts/analysis/generate_cnn_gradcam_3d.py \
    --checkpoint "AUTO" \
    --patient_ids Brats18_TCIA10_103_1 Brats18_TCIA10_104_1 \
    --output_dir "ensemble/results/interpretability/cnn_gradcam" \
    --target_class "pred" \
    --num_slices 12
```

### Using Patient IDs File

```bash
# Create a text file with patient IDs (one per line)
echo "Brats18_TCIA10_103_1" > data/selected_patients.txt
echo "Brats18_TCIA10_104_1" >> data/selected_patients.txt

# Run script
python scripts/analysis/generate_cnn_gradcam_3d.py \
    --checkpoint "AUTO" \
    --patient_ids_file "data/selected_patients.txt" \
    --output_dir "ensemble/results/interpretability/cnn_gradcam" \
    --target_class "pred" \
    --num_slices 12 \
    --fold 0
```

### Manual Checkpoint Path

```bash
python scripts/analysis/generate_cnn_gradcam_3d.py \
    --checkpoint "results/ResNet50-3D/runs/fold_0/run_20240101_120000/checkpoints/best.pt" \
    --patient_ids Brats18_TCIA10_103_1 \
    --output_dir "ensemble/results/interpretability/cnn_gradcam"
```

## Arguments

- `--checkpoint`: Checkpoint path or "AUTO" to find latest (default: AUTO)
- `--fold`: Fold number for AUTO checkpoint (default: 0)
- `--patient_ids`: List of patient IDs (space-separated)
- `--patient_ids_file`: Path to text file with patient IDs (one per line)
- `--output_dir`: Output directory (default: ensemble/results/interpretability/cnn_gradcam)
- `--target_class`: Target class for Grad-CAM: "pred" (predicted), "0" (LGG), or "1" (HGG) (default: pred)
- `--num_slices`: Number of slices per montage (default: 12)
- `--device`: Device to use (default: cuda if available, else cpu)

## Output Structure

For each patient, the following files are generated:

```
ensemble/results/interpretability/cnn_gradcam/
├── checkpoint_info.json          # Checkpoint metadata
├── summary.json                  # Summary of all patients
└── <patient_id>/
    ├── gradcam_3d.nii.gz         # 3D heatmap as NIfTI file
    ├── axial_overlay.png         # Axial plane montage
    ├── coronal_overlay.png        # Coronal plane montage
    ├── sagittal_overlay.png       # Sagittal plane montage
    ├── summary.png                # All three planes in one image
    └── metadata.json              # Patient prediction info
```

## Output Files

### `gradcam_3d.nii.gz`
- 3D Grad-CAM heatmap as NIfTI file
- Shape: (128, 128, 128)
- Values: [0, 1] (normalized)
- Can be loaded in medical imaging viewers (e.g., ITK-SNAP, 3D Slicer)

### `*_overlay.png`
- Montage showing multiple slices with Grad-CAM overlay
- Background: T1ce modality (grayscale)
- Overlay: Grad-CAM heatmap (colored, semi-transparent)
- Shows evenly spaced slices across the volume

### `summary.png`
- Single image with all three planes (axial, coronal, sagittal)
- 9 slices per plane
- Includes patient ID and prediction info in title

### `metadata.json`
Contains:
- Patient ID
- Class (LGG/HGG)
- Predicted class and probabilities
- Target class used for Grad-CAM

## Interpretation

- **Red/Yellow regions**: High importance (model focuses here)
- **Blue/Dark regions**: Low importance
- **Background**: T1ce modality (grayscale)
- **Overlay**: Grad-CAM heatmap (colored, 50% transparency)

**Important Notes:**
- Grad-CAM shows correlation, not causation
- Heatmaps are relative (normalized to [0, 1])
- Only positive contributions are shown (ReLU applied)
- Heatmap resolution matches input (128×128×128)

## Memory Considerations

- Batch size is fixed to 1 (memory-safe)
- 3D volumes are 128×128×128 (4 channels) = ~67 MB per volume
- Grad-CAM computation requires additional memory for gradients
- If GPU memory is limited, use `--device cpu` (slower but more memory-efficient)

## Troubleshooting

### Checkpoint Not Found
- Ensure model is trained: `results/ResNet50-3D/runs/fold_X/.../checkpoints/best.pt`
- Check fold number: `--fold 0` (default) to `--fold 4`
- Use manual path: `--checkpoint <full_path>`

### Patient Not Found
- Check patient ID format: `Brats18_TCIA10_103_1`
- Verify data exists: `data/processed/stage_4_resize/train/<class>/<patient_id>/`
- Check class name: LGG or HGG

### CUDA Out of Memory
- Use CPU: `--device cpu`
- Process fewer patients at a time
- Reduce `--num_slices` (fewer slices per montage)

## Example Output Paths

After running on patient `Brats18_TCIA10_103_1`:

```
ensemble/results/interpretability/cnn_gradcam/
├── checkpoint_info.json
├── summary.json
└── Brats18_TCIA10_103_1/
    ├── gradcam_3d.nii.gz
    ├── axial_overlay.png
    ├── coronal_overlay.png
    ├── sagittal_overlay.png
    ├── summary.png
    └── metadata.json
```

