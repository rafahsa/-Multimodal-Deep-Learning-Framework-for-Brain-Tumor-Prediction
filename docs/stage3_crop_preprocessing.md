# Stage 3: ROI Cropping

## Overview

Stage 3 of the BraTS2018 preprocessing pipeline applies ROI (Region of Interest) cropping to 3D NIfTI volumes from Stage 2 outputs (Z-score normalized). This step reduces computational load by focusing on brain regions while maintaining spatial alignment across modalities.

**Key Features:**
- Processes imaging modalities only (t1, t1ce, t2, flair)
- All modalities for the same patient use the SAME bounding box
- Configurable bounding box computation (reference modality or union)
- Correctly updates image origin after cropping
- Resumable processing with manifest tracking
- Parallel processing with capped worker count
- Comprehensive logging and error handling
- Persistent outputs survive pod restarts

## Requirements

- Python 3.7+
- SimpleITK
- NumPy
- PyYAML
- Standard library: multiprocessing, pathlib, logging

Install dependencies:
```bash
pip install SimpleITK numpy PyYAML
```

## Directory Structure

```
/workspace/brain_tumor_project/
├── data/
│   ├── processed/
│   │   ├── stage_2_zscore/              # Input (Stage 2 outputs)
│   │   │   └── train/
│   │   │       ├── HGG/
│   │   │       │   └── <patient_id>/
│   │   │       │       ├── <patient_id>_t1.nii.gz
│   │   │       │       ├── <patient_id>_t1ce.nii.gz
│   │   │       │       ├── <patient_id>_t2.nii.gz
│   │   │       │       └── <patient_id>_flair.nii.gz
│   │   │       └── LGG/
│   │   └── stage_3_crop/                # Output directory (persistent)
│   │       ├── manifest.jsonl           # Processing manifest
│   │       └── train/
│   │           ├── HGG/
│   │           │   └── <patient_id>/
│   │           │       └── <patient_id>_<modality>.nii.gz
│   │           └── LGG/
│   │               └── <patient_id>/
│   │                   └── <patient_id>_<modality>.nii.gz
├── logs/
│   └── preprocessing/
│       └── stage_3_crop/                # Log files
│           └── stage3_crop_YYYYMMDD_HHMMSS.log
├── configs/
│   └── stage_3_crop.yaml                # Configuration file
└── scripts/
    └── preprocessing/
        └── run_stage3_crop.py           # Main script
```

## Usage

### Basic Usage

Process all classes (HGG and LGG) with automatic capped worker detection:

```bash
python scripts/preprocessing/run_stage3_crop.py --split train --classes HGG LGG --workers auto
```

### Process Specific Classes

Process only HGG patients:

```bash
python scripts/preprocessing/run_stage3_crop.py --split train --classes HGG --workers 8
```

### Dry-Run Mode

Preview what will be processed without actually running:

```bash
python scripts/preprocessing/run_stage3_crop.py --split train --classes HGG LGG --dry-run
```

### Custom Worker Count

Specify the number of parallel workers:

```bash
python scripts/preprocessing/run_stage3_crop.py --split train --classes HGG LGG --workers 8
```

### Custom Configuration

Use a different config file:

```bash
python scripts/preprocessing/run_stage3_crop.py --config configs/custom_stage3.yaml --split train --classes HGG LGG
```

## Resumability

The pipeline is fully resumable:

1. **Manifest Tracking**: All processed files are recorded in `data/processed/stage_3_crop/manifest.jsonl` (JSONL format).

2. **Automatic Skip**: On rerun, the script automatically skips patients where:
   - All modalities are already listed in the manifest with status `success`
   - All output files exist and are readable

3. **Resume After Interruption**: If the process is interrupted (pod restart, crash, etc.):
   ```bash
   # Simply rerun the same command - it will continue from where it left off
   python scripts/preprocessing/run_stage3_crop.py --split train --classes HGG LGG --workers 8
   ```

4. **Manifest Format**: Each entry in `manifest.jsonl` contains:
   - `patient_id`: Patient identifier
   - `modality`: Modality name (t1, t1ce, t2, flair)
   - `input_path`: Input file path (from Stage 2)
   - `output_path`: Output file path
   - `status`: Processing status (success, failed, skipped)
   - `timestamp`: ISO format timestamp
   - `error`: Error message (if failed)

## Configuration

Edit `configs/stage_3_crop.yaml` to adjust:

- **Crop Parameters**:
  - `padding`: Padding in voxels to add around bounding box (default: 10)
  - `eps_mask`: Threshold for brain mask `abs(image) > eps_mask` (default: 1e-6)
  - `bbox_mode`: Bounding box computation mode:
    - `"reference_modality"`: Use single reference modality (faster, default)
    - `"union"`: Use union mask across all modalities (more robust)
  - `reference_modality`: Modality to use if `bbox_mode="reference_modality"` (default: flair)

- **Paths**: All paths are relative to `project_root` (set to `/workspace/brain_tumor_project`)

- **Processing**:
  - `modalities`: List of modalities to process (default: [t1, t1ce, t2, flair])
  - `output_format`: Output file format (default: .nii.gz)

- **Worker Count**: Auto workers are capped: `min(max(1, cpu_count()//4), 16)`

## Algorithm Details

### ROI Cropping Algorithm

1. **Input**: 3D NIfTI volume from Stage 2 (Z-score normalized)

2. **Bounding Box Computation** (per patient, applied to all modalities):
   
   **Mode: reference_modality** (default, faster):
   - Load reference modality (default: flair)
   - Create brain mask: `mask = abs(image_array) > eps_mask`
   - Find bounding box of mask
   - Apply padding (default: 10 voxels per axis)
   
   **Mode: union** (more robust):
   - Load all modalities
   - Create mask for each: `mask = abs(image_array) > eps_mask`
   - Find bounding box for each modality
   - Compute union: `(min(z_mins), max(z_maxs)), (min(y_mins), max(y_maxs)), (min(x_mins), max(x_maxs))`
   - Apply padding to union bbox

3. **Cropping** (applied to each modality):
   - Extract region using SimpleITK ExtractImageFilter
   - Update origin based on crop start position:
     ```
     new_origin = old_origin + direction @ (spacing * crop_start_xyz)
     ```
   - Preserve spacing and direction
   - Save as .nii.gz

4. **Critical Constraint**: All 4 modalities for the same patient use the **SAME bounding box** to maintain spatial alignment.

### Why Same Bounding Box Per Patient?

- **Spatial Alignment**: Ensures all modalities remain spatially aligned for multi-modal analysis
- **Model Training**: Deep learning models (MIL, ResNet3D, Swin UNETR) require aligned multi-modal inputs
- **Consistency**: Prevents misalignment issues in downstream stages

### Origin Update

The origin is correctly updated after cropping to maintain physical coordinates:
- Physical position = origin + direction × (spacing × voxel_index)
- New origin accounts for the crop start position in physical space
- Spacing and direction matrices are preserved

## Performance

- **Parallelization**: Uses multiprocessing with capped worker count
  - Auto: `min(max(1, cpu_count()//4), 16)`
  - Manual: Specify `--workers N`
- **Memory**: Processes one patient at a time per worker
- **CPU**: Bounding box computation is fast; cropping is I/O bound
- **I/O**: Reads from Stage 2 outputs (.nii.gz), writes cropped volumes (.nii.gz)

## Output Verification

The script automatically verifies outputs:
- Checks that output file exists and is readable
- Validates using SimpleITK ReadImage
- Logs verification failures

## Logging

Logs are written to:
- **File**: `logs/preprocessing/stage_3_crop/stage3_crop_YYYYMMDD_HHMMSS.log`
- **Console**: INFO level and above

Log entries include:
- Processing start/completion
- Bounding box computation for each patient
- Success/failure per file
- Skip notifications for already processed patients
- Error messages with full stack traces
- Summary statistics
- Processing duration

## Summary Output

After processing, the script prints:
- Status breakdown (success, failed, skipped counts)
- Class breakdown (HGG, LGG counts)
- Total files processed
- Processing duration
- Input and output directory paths
- Manifest file path

Example:
```
================================================================================
PROCESSING SUMMARY
================================================================================

Status breakdown:
  success     :  1140
  skipped     :     0
  failed      :     0

Class breakdown:
  HGG         :   840
  LGG         :   300

Total processed: 1140 files
Processing time: 0:25:15.123456
Input directory: /workspace/brain_tumor_project/data/processed/stage_2_zscore
Output directory: /workspace/brain_tumor_project/data/processed/stage_3_crop
Manifest file: /workspace/brain_tumor_project/data/processed/stage_3_crop/manifest.jsonl
================================================================================
```

## Medical Imaging Context

### Importance for Deep Learning

ROI cropping is critical for:
- **Computational Efficiency**: Reduces memory and compute requirements
- **Model Training**: Focuses learning on relevant brain regions
- **Multi-modal Alignment**: Ensures all modalities are spatially consistent
- **Downstream Processing**: Prepares for resizing to standard dimensions (e.g., 128×128×128)

### Pipeline Integration

Stage 3 outputs feed directly into:
- **Stage 4 (if implemented)**: Resize to (128, 128, 128)
- **Model training**: Direct input to classification models (MIL, ResNet3D, Swin UNETR)

## Troubleshooting

### Common Issues

1. **Input files not found**:
   - Verify Stage 2 has been completed successfully
   - Check that `data/processed/stage_2_zscore/` contains expected structure
   - Verify input path in config matches actual Stage 2 output location

2. **Memory errors**:
   - Reduce worker count: `--workers 4`
   - Process one class at a time

3. **Permission errors**:
   - Ensure write permissions to output directory
   - Check `/workspace` is mounted and writable

4. **Bounding box computation fails**:
   - Check that reference modality exists for the patient
   - Verify input files are valid NIfTI volumes
   - Try `bbox_mode: union` if reference modality fails

5. **Origin update issues**:
   - Verify input images have valid spacing/origin/direction metadata
   - Check logs for specific error messages

### Debug Mode

For verbose logging, modify `log_level` in config to `DEBUG`, or edit the script to use DEBUG level.

## Next Steps

After Stage 3 completion:
- Verify output volumes are correctly cropped
- Check that bounding boxes are reasonable (not too small/large)
- Verify spatial alignment across modalities for sample patients
- Proceed to Stage 4 (if implemented): Resizing to (128, 128, 128)
- Use outputs for model training

## References

- SimpleITK Documentation: https://simpleitk.org/
- NumPy Documentation: https://numpy.org/doc/
- BraTS2018 Dataset: https://www.med.upenn.edu/cbica/brats2018/
- ROI Cropping in Medical Imaging: Standard practice for reducing computational load

## Academic Reproducibility

This pipeline is designed for:
- Academic thesis work
- Medical AI research publications
- Public GitHub repositories

All parameters are configurable, code is modular and well-documented, and the pipeline is fully reproducible with proper logging and manifest tracking.

