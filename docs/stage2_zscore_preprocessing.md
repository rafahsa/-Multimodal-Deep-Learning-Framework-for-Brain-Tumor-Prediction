# Stage 2: Z-score Normalization

## Overview

Stage 2 of the BraTS2018 preprocessing pipeline applies Z-score normalization to 3D NIfTI volumes from Stage 1 outputs (N4 bias field corrected). This step standardizes intensity distributions across patients and modalities, which is critical for deep learning model training stability and convergence.

**Key Features:**
- Processes imaging modalities only (t1, t1ce, t2, flair)
- Normalizes only brain voxels (values > 0), preserves background (zeros)
- Resumable processing with manifest tracking
- Parallel processing with automatic CPU detection
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
│   │   ├── stage_1_n4/                  # Input (Stage 1 outputs)
│   │   │   └── train/
│   │   │       ├── HGG/
│   │   │       │   └── <patient_id>/
│   │   │       │       ├── <patient_id>_t1.nii.gz
│   │   │       │       ├── <patient_id>_t1ce.nii.gz
│   │   │       │       ├── <patient_id>_t2.nii.gz
│   │   │       │       └── <patient_id>_flair.nii.gz
│   │   │       └── LGG/
│   │   └── stage_2_zscore/              # Output directory (persistent)
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
│       └── stage_2_zscore/               # Log files
│           └── stage2_zscore_YYYYMMDD_HHMMSS.log
├── configs/
│   └── stage_2_zscore.yaml               # Configuration file
└── scripts/
    └── preprocessing/
        └── run_stage2_zscore.py          # Main script
```

## Usage

### Basic Usage

Process all classes (HGG and LGG) with automatic worker detection:

```bash
python scripts/preprocessing/run_stage2_zscore.py --split train --classes HGG LGG --workers auto
```

### Process Specific Classes

Process only HGG patients:

```bash
python scripts/preprocessing/run_stage2_zscore.py --split train --classes HGG --workers auto
```

### Dry-Run Mode

Preview what will be processed without actually running:

```bash
python scripts/preprocessing/run_stage2_zscore.py --split train --classes HGG LGG --dry-run
```

### Custom Worker Count

Specify the number of parallel workers:

```bash
python scripts/preprocessing/run_stage2_zscore.py --split train --classes HGG LGG --workers 8
```

### Custom Configuration

Use a different config file:

```bash
python scripts/preprocessing/run_stage2_zscore.py --config configs/custom_stage2.yaml --split train --classes HGG LGG
```

## Resumability

The pipeline is fully resumable:

1. **Manifest Tracking**: All processed files are recorded in `data/processed/stage_2_zscore/manifest.jsonl` (JSONL format).

2. **Automatic Skip**: On rerun, the script automatically skips files that:
   - Are already listed in the manifest with status `success`
   - Have existing output files that are readable

3. **Resume After Interruption**: If the process is interrupted (pod restart, crash, etc.):
   ```bash
   # Simply rerun the same command - it will continue from where it left off
   python scripts/preprocessing/run_stage2_zscore.py --split train --classes HGG LGG --workers auto
   ```

4. **Manifest Format**: Each entry in `manifest.jsonl` contains:
   - `patient_id`: Patient identifier
   - `modality`: Modality name (t1, t1ce, t2, flair)
   - `input_path`: Input file path (from Stage 1)
   - `output_path`: Output file path
   - `status`: Processing status (success, failed, skipped)
   - `timestamp`: ISO format timestamp
   - `error`: Error message (if failed)

## Configuration

Edit `configs/stage_2_zscore.yaml` to adjust:

- **Z-score Parameters**:
  - `eps`: Epsilon value to avoid division by zero (default: 1e-8)

- **Paths**: All paths are relative to `project_root` (set to `/workspace/brain_tumor_project`)

- **Processing**:
  - `modalities`: List of modalities to process (default: [t1, t1ce, t2, flair])
  - `output_format`: Output file format (default: .nii.gz)

## Algorithm Details

### Z-score Normalization

The normalization is computed **ONLY on brain voxels** (values > 0), preserving background (zeros):

1. **Input**: 3D NIfTI volume from Stage 1 (N4 corrected)
2. **Brain Voxel Extraction**: `brain_voxels = image_array[image_array > 0]`
3. **Statistics Computation**:
   - `mean = np.mean(brain_voxels)`
   - `std = np.std(brain_voxels)`
4. **Normalization**:
   - `normalized = (image_array - mean) / (std + eps)`
   - `normalized[image_array == 0] = 0` (preserve background)
5. **Output**: Compressed NIfTI (.nii.gz) with Z-score normalized intensities
6. **Metadata Preservation**: Spacing, origin, and direction are preserved

### Why Normalize Only Brain Voxels?

- **Background preservation**: Background voxels (zeros) represent empty space and should remain zero
- **Biological relevance**: Normalization should reflect tissue intensity distributions, not background
- **Model training**: Standardizing brain tissue intensities improves training stability and convergence
- **Comparability**: Enables intensity-based features to be comparable across patients and modalities

### What is NOT Processed

- Segmentation masks are **not** processed (they remain in Stage 1 outputs if needed)
- Background voxels (zeros) remain zero after normalization
- Raw input data (Stage 1 outputs) are never modified

## Performance

- **Parallelization**: Uses multiprocessing with worker count `max(1, cpu_count() - 2)` by default
- **Memory**: Processes one volume at a time per worker to avoid RAM overload
- **CPU**: Z-score normalization is computationally efficient; parallelization significantly speeds up processing
- **I/O**: Reads from Stage 1 outputs (.nii.gz), writes normalized volumes (.nii.gz)

## Output Verification

The script automatically verifies outputs:
- Checks that output file exists and is readable
- Validates using SimpleITK ReadImage
- Logs verification failures

## Logging

Logs are written to:
- **File**: `logs/preprocessing/stage_2_zscore/stage2_zscore_YYYYMMDD_HHMMSS.log`
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
  success     :  850
  skipped     :   50
  failed      :    0

Class breakdown:
  HGG         :  650
  LGG         :  250

Total processed: 900
Processing time: 0:15:32.456789
Input directory: /workspace/brain_tumor_project/data/processed/stage_1_n4
Output directory: /workspace/brain_tumor_project/data/processed/stage_2_zscore
Manifest file: /workspace/brain_tumor_project/data/processed/stage_2_zscore/manifest.jsonl
================================================================================
```

## Medical Imaging Context

### Importance for Deep Learning

Z-score normalization is critical for:
- **Training stability**: Standardized inputs prevent gradient explosion/vanishing
- **Convergence speed**: Normalized features enable faster optimization
- **Model performance**: Especially important for:
  - Multi-Instance Learning (MIL) models with attention mechanisms
  - ResNet3D architectures
  - Swin UNETR models
- **Feature comparability**: Enables intensity-based features to be meaningful across patients

### Pipeline Integration

Stage 2 outputs feed directly into:
- **Stage 3 (if implemented)**: ROI cropping
- **Stage 4 (if implemented)**: Resize to (128, 128, 128)
- **Model training**: Direct input to classification models

## Troubleshooting

### Common Issues

1. **Input files not found**:
   - Verify Stage 1 has been completed successfully
   - Check that `data/processed/stage_1_n4/` contains expected structure
   - Verify input path in config matches actual Stage 1 output location

2. **Memory errors**:
   - Reduce worker count: `--workers 4`
   - Process one class at a time

3. **Permission errors**:
   - Ensure write permissions to output directory
   - Check `/workspace` is mounted and writable

4. **Corrupted output files**:
   - Delete corrupted files from output directory
   - Remove corresponding entries from manifest.jsonl (or let script detect and reprocess)

5. **All zeros in output**:
   - Check that input files contain non-zero voxels
   - Verify Stage 1 outputs are valid

### Debug Mode

For verbose logging, modify `log_level` in config to `DEBUG`, or edit the script to use DEBUG level.

## Next Steps

After Stage 2 completion:
- Verify output volumes have normalized intensity distributions
- Check that background voxels remain zero
- Proceed to Stage 3 (if implemented): ROI cropping
- Proceed to Stage 4 (if implemented): Resizing to (128, 128, 128)
- Use outputs for model training

## References

- SimpleITK Documentation: https://simpleitk.org/
- NumPy Documentation: https://numpy.org/doc/
- BraTS2018 Dataset: https://www.med.upenn.edu/cbica/brats2018/
- Normalization in Medical Imaging: Standard practice for deep learning pipelines

## Academic Reproducibility

This pipeline is designed for:
- Academic thesis work
- Medical AI research publications
- Public GitHub repositories

All parameters are configurable, code is modular and well-documented, and the pipeline is fully reproducible with proper logging and manifest tracking.

