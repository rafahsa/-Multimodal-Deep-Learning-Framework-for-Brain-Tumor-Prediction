# Stage 1: N4 Bias Field Correction

## Overview

Stage 1 of the BraTS2018 preprocessing pipeline applies N4 bias field correction to 3D NIfTI volumes. This step corrects intensity non-uniformity (bias field) in MRI images, which is essential for consistent intensity-based analysis.

**Key Features:**
- Processes imaging modalities only (t1, t1ce, t2, flair)
- Preserves segmentation masks (seg) unchanged
- Resumable processing with manifest tracking
- Parallel processing with automatic CPU detection
- Comprehensive logging and error handling
- Persistent outputs survive pod restarts

## Requirements

- Python 3.7+
- SimpleITK
- PyYAML
- Standard library: multiprocessing, pathlib, logging

Install dependencies:
```bash
pip install SimpleITK PyYAML
```

## Directory Structure

```
/workspace/brain_tumor_project/
├── data/
│   ├── brats2018/                          # Input (raw data, not modified)
│   │   └── MICCAI_BraTS_2018_Data_Training/
│   │       ├── HGG/
│   │       │   └── <patient_id>/
│   │       │       ├── <patient_id>_t1.nii
│   │       │       ├── <patient_id>_t1ce.nii
│   │       │       ├── <patient_id>_t2.nii
│   │       │       ├── <patient_id>_flair.nii
│   │       │       └── <patient_id>_seg.nii
│   │       └── LGG/
│   └── processed/
│       └── stage_1_n4/                     # Output directory (persistent)
│           ├── manifest.jsonl              # Processing manifest
│           └── train/
│               ├── HGG/
│               │   └── <patient_id>/
│               │       └── <patient_id>_<modality>.nii.gz
│               └── LGG/
│                   └── <patient_id>/
│                       └── <patient_id>_<modality>.nii.gz
├── logs/
│   └── preprocessing/
│       └── stage_1_n4/                     # Log files
│           └── stage1_n4_YYYYMMDD_HHMMSS.log
├── configs/
│   └── stage_1_n4.yaml                     # Configuration file
└── scripts/
    └── preprocessing/
        └── run_stage1_n4.py                # Main script
```

## Usage

### Basic Usage

Process all classes (HGG and LGG) with automatic worker detection:

```bash
python scripts/preprocessing/run_stage1_n4.py --split train --classes HGG LGG --workers auto
```

### Process Specific Classes

Process only HGG patients:

```bash
python scripts/preprocessing/run_stage1_n4.py --split train --classes HGG --workers auto
```

### Dry-Run Mode

Preview what will be processed without actually running:

```bash
python scripts/preprocessing/run_stage1_n4.py --split train --classes HGG LGG --dry-run
```

### Custom Worker Count

Specify the number of parallel workers:

```bash
python scripts/preprocessing/run_stage1_n4.py --split train --classes HGG LGG --workers 8
```

### Custom Configuration

Use a different config file:

```bash
python scripts/preprocessing/run_stage1_n4.py --config configs/custom_stage1.yaml --split train --classes HGG LGG
```

## Resumability

The pipeline is fully resumable:

1. **Manifest Tracking**: All processed files are recorded in `data/processed/stage_1_n4/manifest.jsonl` (JSONL format).

2. **Automatic Skip**: On rerun, the script automatically skips files that:
   - Are already listed in the manifest with status `success`
   - Have existing output files that are readable

3. **Resume After Interruption**: If the process is interrupted (pod restart, crash, etc.):
   ```bash
   # Simply rerun the same command - it will continue from where it left off
   python scripts/preprocessing/run_stage1_n4.py --split train --classes HGG LGG --workers auto
   ```

4. **Manifest Format**: Each entry in `manifest.jsonl` contains:
   - `patient_id`: Patient identifier
   - `modality`: Modality name (t1, t1ce, t2, flair)
   - `input_path`: Original input file path
   - `output_path`: Output file path
   - `status`: Processing status (success, failed, skipped)
   - `timestamp`: ISO format timestamp
   - `error`: Error message (if failed)

## Configuration

Edit `configs/stage_1_n4.yaml` to adjust:

- **N4 Parameters**:
  - `max_iterations`: Iterations per resolution level `[40, 40, 30, 20]`
  - `num_control_points`: B-spline control points (default: 4)
  - `convergence_threshold`: Convergence threshold (default: 0.001)

- **Paths**: All paths are relative to `project_root` (set to `/workspace/brain_tumor_project`)

- **Processing**:
  - `modalities`: List of modalities to process (default: [t1, t1ce, t2, flair])
  - `output_format`: Output file format (default: .nii.gz)

## Algorithm Details

### N4 Bias Field Correction

1. **Input**: 3D NIfTI volume (t1, t1ce, t2, or flair)
2. **Mask Creation**: Otsu thresholding to create brain mask
3. **N4 Correction**: Multi-resolution B-spline fitting with:
   - Maximum iterations: [40, 40, 30, 20] per level
   - 4x4x4 control point grid
   - Convergence threshold: 0.001
4. **Post-processing**:
   - Clamp negative values to 0
   - Preserve spacing, origin, and direction metadata
   - Convert to float32 for precision
5. **Output**: Compressed NIfTI (.nii.gz) with bias field corrected

### What is NOT Processed

- Segmentation masks (`*_seg.nii`) are **not** processed by N4
- Validation dataset is left untouched
- Raw input data is never modified

## Performance

- **Parallelization**: Uses multiprocessing with worker count `max(1, cpu_count() - 2)` by default
- **Memory**: Processes one volume at a time per worker to avoid RAM overload
- **CPU**: N4 correction is CPU-intensive; parallelization significantly speeds up processing

## Output Verification

The script automatically verifies outputs:
- Checks that output file exists and is readable
- Validates using SimpleITK ReadImage
- Logs verification failures

## Logging

Logs are written to:
- **File**: `logs/preprocessing/stage_1_n4/stage1_n4_YYYYMMDD_HHMMSS.log`
- **Console**: INFO level and above

Log entries include:
- Processing start/completion
- Success/failure per file
- Skip notifications for already processed files
- Error messages with full stack traces
- Summary statistics

## Summary Output

After processing, the script prints:
- Status breakdown (success, failed, skipped counts)
- Total files processed
- Output directory path
- Manifest file path

Example:
```
================================================================================
PROCESSING SUMMARY
================================================================================

Status breakdown:
  success     :  850
  skipped     :   50
  failed      :    0

Total processed: 900
Output directory: /workspace/brain_tumor_project/data/processed/stage_1_n4
Manifest file: /workspace/brain_tumor_project/data/processed/stage_1_n4/manifest.jsonl
================================================================================
```

## Troubleshooting

### Common Issues

1. **File not found errors**:
   - Verify input data path in config matches actual dataset location
   - Check that patient folders contain expected modality files

2. **Memory errors**:
   - Reduce worker count: `--workers 4`
   - Process one class at a time

3. **Permission errors**:
   - Ensure write permissions to output directory
   - Check `/workspace` is mounted and writable

4. **Corrupted output files**:
   - Delete corrupted files from output directory
   - Remove corresponding entries from manifest.jsonl (or let script detect and reprocess)

### Debug Mode

For verbose logging, modify `log_level` in config to `DEBUG`, or edit the script to use DEBUG level.

## Next Steps

After Stage 1 completion:
- Verify output volumes are correct
- Proceed to Stage 2 (if implemented): Intensity normalization (z-score)
- Proceed to Stage 3 (if implemented): Cropping/padding
- Proceed to Stage 4 (if implemented): Resizing

## References

- N4 Bias Field Correction: Tustison et al., "N4ITK: Improved N3 Bias Correction", IEEE TMI, 2010
- SimpleITK Documentation: https://simpleitk.org/
- BraTS2018 Dataset: https://www.med.upenn.edu/cbica/brats2018/

