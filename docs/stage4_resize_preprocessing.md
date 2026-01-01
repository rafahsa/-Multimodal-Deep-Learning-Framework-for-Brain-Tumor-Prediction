# Stage 4: Resize

## Overview

Stage 4 of the BraTS2018 preprocessing pipeline resizes 3D NIfTI volumes from Stage 3 outputs to a fixed target size (128, 128, 128) using SimpleITK resampling with linear interpolation. This standardization is essential for deep learning models that require fixed input dimensions.

**Key Features:**
- Resizes all volumes to fixed size (128, 128, 128)
- Uses SimpleITK ResampleImageFilter with linear interpolation
- Preserves direction and origin metadata
- Correctly updates spacing based on size change
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
│   │   ├── stage_3_crop/                # Input (Stage 3 outputs)
│   │   │   └── train/
│   │   │       ├── HGG/
│   │   │       │   └── <patient_id>/
│   │   │       │       ├── <patient_id>_t1.nii.gz
│   │   │       │       ├── <patient_id>_t1ce.nii.gz
│   │   │       │       ├── <patient_id>_t2.nii.gz
│   │   │       │       └── <patient_id>_flair.nii.gz
│   │   │       └── LGG/
│   │   └── stage_4_resize/              # Output directory (persistent)
│   │       ├── manifest.jsonl           # Processing manifest
│   │       └── train/
│   │           ├── HGG/
│   │           │   └── <patient_id>/
│   │           │       └── <patient_id>_<modality>.nii.gz (all 128×128×128)
│   │           └── LGG/
│   │               └── <patient_id>/
│   │                   └── <patient_id>_<modality>.nii.gz
├── logs/
│   └── preprocessing/
│       └── stage_4_resize/              # Log files
│           └── stage4_resize_YYYYMMDD_HHMMSS.log
├── configs/
│   └── stage_4_resize.yaml              # Configuration file
└── scripts/
    └── preprocessing/
        └── run_stage4_resize.py         # Main script
```

## Usage

### Basic Usage

Process all classes (HGG and LGG) with automatic capped worker detection:

```bash
python scripts/preprocessing/run_stage4_resize.py --split train --classes HGG LGG --workers auto
```

### Process Specific Classes

Process only HGG patients:

```bash
python scripts/preprocessing/run_stage4_resize.py --split train --classes HGG --workers 8
```

### Dry-Run Mode

Preview what will be processed without actually running:

```bash
python scripts/preprocessing/run_stage4_resize.py --split train --classes HGG LGG --dry-run
```

### Custom Worker Count

Specify the number of parallel workers:

```bash
python scripts/preprocessing/run_stage4_resize.py --split train --classes HGG LGG --workers 8
```

### Custom Configuration

Use a different config file:

```bash
python scripts/preprocessing/run_stage4_resize.py --config configs/custom_stage4.yaml --split train --classes HGG LGG
```

## Resumability

The pipeline is fully resumable:

1. **Manifest Tracking**: All processed files are recorded in `data/processed/stage_4_resize/manifest.jsonl` (JSONL format).

2. **Automatic Skip**: On rerun, the script automatically skips files that:
   - Are already listed in the manifest with status `success`
   - Have existing output files that are readable and have the correct size (128×128×128)

3. **Resume After Interruption**: If the process is interrupted (pod restart, crash, etc.):
   ```bash
   # Simply rerun the same command - it will continue from where it left off
   python scripts/preprocessing/run_stage4_resize.py --split train --classes HGG LGG --workers 8
   ```

4. **Manifest Format**: Each entry in `manifest.jsonl` contains:
   - `patient_id`: Patient identifier
   - `modality`: Modality name (t1, t1ce, t2, flair)
   - `input_path`: Input file path (from Stage 3)
   - `output_path`: Output file path
   - `status`: Processing status (success, failed, skipped)
   - `timestamp`: ISO format timestamp
   - `error`: Error message (if failed)

## Configuration

Edit `configs/stage_4_resize.yaml` to adjust:

- **Resize Parameters**:
  - `target_size`: Target size as [x, y, z] (default: [128, 128, 128])
  - `interpolation`: Interpolation method (default: "linear", also supports "nearest")

- **Paths**: All paths are relative to `project_root` (set to `/workspace/brain_tumor_project`)

- **Processing**:
  - `modalities`: List of modalities to process (default: [t1, t1ce, t2, flair])
  - `output_format`: Output file format (default: .nii.gz)

- **Worker Count**: Auto workers are capped: `min(max(1, cpu_count()//4), 16)`

## Algorithm Details

### Resize Algorithm

1. **Input**: 3D NIfTI volume from Stage 3 (ROI cropped)

2. **Resampling**:
   - Read input image using SimpleITK
   - Get current size and spacing (xyz order)
   - Compute new spacing:
     ```
     new_spacing[i] = old_spacing[i] * (old_size[i] / new_size[i])
     ```
   - Set up ResampleImageFilter:
     - Target size: (128, 128, 128) in xyz order
     - New spacing: computed from size ratio
     - Origin: preserved from original image
     - Direction: preserved from original image
     - Interpolator: `sitk.sitkLinear` (linear interpolation)
   
3. **Output**: Resized volume as .nii.gz with:
   - Fixed size: (128, 128, 128)
   - Updated spacing (maintains physical scale)
   - Preserved origin and direction
   - Linear interpolation for smooth intensity transitions

### Why Fixed Size?

- **Model Input**: Deep learning models require fixed input dimensions
- **Batch Processing**: Enables efficient batch processing in training
- **Standardization**: Ensures consistent representation across all patients
- **Memory Efficiency**: Predictable memory requirements for model inference

### Spacing Update

The spacing is correctly updated to maintain physical scale:
- Physical distance = spacing × voxel_count
- When resizing, we maintain physical distance by adjusting spacing
- Formula: `new_spacing = old_spacing * (old_size / new_size)`

### Metadata Preservation

- **Direction**: Preserved (spatial orientation matrix)
- **Origin**: Preserved (physical position of first voxel)
- **Spacing**: Updated based on size change (maintains physical scale)

## Performance

- **Parallelization**: Uses multiprocessing with capped worker count
  - Auto: `min(max(1, cpu_count()//4), 16)`
  - Manual: Specify `--workers N`
- **Memory**: Processes one volume at a time per worker
- **CPU**: Resampling is computationally efficient; parallelization speeds up processing
- **I/O**: Reads from Stage 3 outputs (.nii.gz), writes resized volumes (.nii.gz)

## Output Verification

The script automatically verifies outputs:
- Checks that output file exists and is readable
- Validates output size matches target (128, 128, 128)
- Validates using SimpleITK ReadImage
- Logs verification failures

## Logging

Logs are written to:
- **File**: `logs/preprocessing/stage_4_resize/stage4_resize_YYYYMMDD_HHMMSS.log`
- **Console**: INFO level and above

Log entries include:
- Processing start/completion
- Success/failure per file
- Skip notifications for already processed files
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
Processing time: 0:10:30.456789
Input directory: /workspace/brain_tumor_project/data/processed/stage_3_crop
Output directory: /workspace/brain_tumor_project/data/processed/stage_4_resize
Manifest file: /workspace/brain_tumor_project/data/processed/stage_4_resize/manifest.jsonl
================================================================================
```

## Verification Commands

After processing, verify the outputs:

```bash
# Count files (expected: 1140)
find data/processed/stage_4_resize -type f -name "*.nii.gz" | wc -l

# Manifest check
python3 -c "import json; f=open('data/processed/stage_4_resize/manifest.jsonl'); lines=f.readlines(); s=sum(1 for l in lines if json.loads(l)['status']=='success'); print(f'Success {s}/{len(lines)}')"

# Check sample volume shape, spacing, origin, direction
python3 << 'EOF'
import SimpleITK as sitk
import glob
p='data/processed/stage_4_resize/train/HGG'
f=glob.glob(p+'/*/*_flair.nii.gz')
print('Sample:', f[0])
img=sitk.ReadImage(f[0])
arr=sitk.GetArrayFromImage(img)
print('Shape:', arr.shape, '(expected: (128, 128, 128) for z,y,x)')
print('Spacing:', img.GetSpacing(), '(x, y, z)')
print('Origin:', img.GetOrigin(), '(x, y, z)')
print('Direction:', img.GetDirection())
EOF
```

**Expected output:**
- Shape: `(128, 128, 128)` (numpy arrays are z,y,x order)
- Spacing: Updated based on original size
- Origin: Preserved from Stage 3
- Direction: Preserved from Stage 3

## Medical Imaging Context

### Importance for Deep Learning

Fixed-size resizing is critical for:
- **Model Architecture**: Most deep learning models require fixed input dimensions
- **Batch Processing**: Enables efficient GPU batch processing
- **Standardization**: Ensures consistent representation for training
- **Inference Speed**: Predictable memory and compute requirements

### Pipeline Integration

Stage 4 outputs are ready for:
- **Model Training**: Direct input to classification models (MIL, ResNet3D, Swin UNETR)
- **Batch Loading**: Can be efficiently batched for training
- **Inference**: Fixed-size inputs simplify deployment

## Troubleshooting

### Common Issues

1. **Input files not found**:
   - Verify Stage 3 has been completed successfully
   - Check that `data/processed/stage_3_crop/` contains expected structure
   - Verify input path in config matches actual Stage 3 output location

2. **Memory errors**:
   - Reduce worker count: `--workers 4`
   - Process one class at a time

3. **Permission errors**:
   - Ensure write permissions to output directory
   - Check `/workspace` is mounted and writable

4. **Size mismatch errors**:
   - Verify target_size in config is correct
   - Check logs for specific error messages

5. **Interpolation artifacts**:
   - Linear interpolation is recommended for MRI intensities
   - If needed, can switch to nearest neighbor (but not recommended)

### Debug Mode

For verbose logging, modify `log_level` in config to `DEBUG`, or edit the script to use DEBUG level.

## Next Steps

After Stage 4 completion:
- Verify all output volumes are exactly (128, 128, 128)
- Check that spacing is correctly updated
- Verify origin and direction are preserved
- Use outputs for model training (MIL, ResNet3D, Swin UNETR)
- Load data in batches for training pipelines

## References

- SimpleITK Documentation: https://simpleitk.org/
- SimpleITK ResampleImageFilter: https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1ResampleImageFilter.html
- NumPy Documentation: https://numpy.org/doc/
- BraTS2018 Dataset: https://www.med.upenn.edu/cbica/brats2018/
- Image Resampling in Medical Imaging: Standard practice for deep learning pipelines

## Academic Reproducibility

This pipeline is designed for:
- Academic thesis work
- Medical AI research publications
- Public GitHub repositories

All parameters are configurable, code is modular and well-documented, and the pipeline is fully reproducible with proper logging and manifest tracking.

