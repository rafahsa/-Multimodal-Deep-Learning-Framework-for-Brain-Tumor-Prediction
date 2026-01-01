# Data Directory

This directory contains all raw and processed data for the brain tumor classification project.

## Structure

- `raw/`: Original, unmodified data from BraTS 2018
  - `BraTS2018/`: Organized by tumor grade
    - `HGG/`: High-Grade Glioma cases (210 patients)
    - `LGG/`: Low-Grade Glioma cases (75 patients)
  - `miccai-brats2018-original-dataset.zip`: Original dataset archive (kept for reference)

- `processed/`: Stage-wise processed data
  - Each preprocessing stage outputs to its own directory
  - Raw data is NEVER modified
  - Intermediate results can be inspected at each stage

## Data Format

- **File Format**: NIfTI (.nii / .nii.gz)
- **Modalities per patient**:
  - `*_t1.nii`: T1-weighted MRI
  - `*_t1ce.nii`: T1-weighted contrast-enhanced MRI
  - `*_t2.nii`: T2-weighted MRI
  - `*_flair.nii`: FLAIR MRI
  - `*_seg.nii`: Segmentation mask (ground truth)

## Important Notes

- **DO NOT modify files in `raw/`**: This is the source of truth
- All preprocessing operations create new files in `processed/` directories
- Each preprocessing stage is independent and can be re-run if needed

