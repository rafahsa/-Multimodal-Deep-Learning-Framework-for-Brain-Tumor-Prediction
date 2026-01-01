# Entropy Visualization for MIL Slice Selection

## Overview

This document explains the entropy-based slice selection visualization tool, which creates publication-ready figures for understanding how Multiple Instance Learning (MIL) models select informative slices from 3D MRI volumes.

## Purpose

The visualization tool (`scripts/analysis/visualize_entropy.py`) generates comprehensive figures that:
1. **Illustrate entropy distribution** across all slices in a 3D volume
2. **Highlight selected slices** that are used for MIL training
3. **Compare slice informativeness** by showing high-entropy vs low-entropy slices
4. **Support model interpretability** by making slice selection criteria transparent

## What is Entropy?

### Shannon Entropy in Medical Imaging

Shannon entropy quantifies the **information content** or **uncertainty** in a slice:

**Formula**: H(X) = -Σ p(x) × log₂(p(x))

Where:
- `p(x)` is the probability of intensity value `x` in the slice
- Higher entropy = more diverse intensity distribution = more informative
- Lower entropy = more uniform distribution = less informative

### Entropy in Brain MRI Context

In brain MRI volumes:

- **High-entropy slices** contain:
  - Diverse tissue types (gray matter, white matter, CSF)
  - Tumor regions with heterogeneous intensity patterns
  - Anatomical structures with varying contrast
  - **→ Rich information for classification**

- **Low-entropy slices** contain:
  - Mostly background (zeros or near-zero values)
  - Uniform tissue regions with minimal variation
  - Out-of-brain regions
  - **→ Less discriminative information**

## Why High-Entropy Slices are Informative for MIL

### Multiple Instance Learning (MIL) Context

MIL models process 3D volumes by:
1. Extracting 2D slices from the volume
2. Treating each slice as an "instance" in a "bag" (the patient)
3. Learning which instances (slices) are most relevant for classification
4. Aggregating instance-level predictions to make a patient-level prediction

### Information Content and Discriminative Power

High-entropy slices are informative for MIL because:

1. **Feature Richness**: High-entropy slices contain diverse intensity patterns that encode anatomical and pathological information

2. **Discriminative Signals**: Tumor regions, tissue boundaries, and anatomical structures appear as intensity variations that contribute to entropy

3. **Computational Efficiency**: Focusing on high-entropy slices reduces computational cost while preserving discriminative information

4. **Model Performance**: MIL models benefit from training on informative slices rather than uniform background regions

### Example Interpretation

Consider two slices:
- **Slice A** (High entropy, H=4.5): Contains brain tissue with tumor, showing intensity variations from 0 to 255
- **Slice B** (Low entropy, H=0.3): Mostly background (zeros) with minimal intensity variation

For LGG vs HGG classification:
- Slice A provides discriminative features (tumor characteristics, tissue contrast)
- Slice B provides little to no useful information
- MIL model should focus on Slice A (and similar high-entropy slices)

## Visualization Components

### 1. Entropy Curve Plot

**Top panel** shows:
- **Blue line**: Entropy values for all slices (x-axis: slice index, y-axis: entropy)
- **Red markers**: Top-k slices selected for MIL training
- **Grid and labels**: Clear axis labels and grid for readability

**Interpretation**:
- Peaks indicate slices with high information content
- Valleys indicate slices with low information content
- Red markers show which slices are used for training

### 2. High-Entropy Slices

**Middle panel** shows 2-3 slices with **highest entropy**:
- Visual representation of what "informative" slices look like
- Typically show brain tissue with diverse structures
- May contain tumor regions, anatomical boundaries, or complex tissue patterns

### 3. Low-Entropy Slices

**Bottom panel** shows 2-3 slices with **lowest entropy**:
- Visual representation of "less informative" slices
- Typically show background regions, uniform tissue, or out-of-brain areas
- Demonstrate why these slices are not selected for MIL training

## How This Visualization Supports Model Interpretability

### Understanding Model Decisions

1. **Transparency**: Visualizations make slice selection criteria explicit and interpretable

2. **Validation**: Researchers can verify that selected slices contain meaningful anatomical/pathological information

3. **Debugging**: Low-performing models can be analyzed by checking if entropy selection is working correctly

4. **Communication**: Visualizations help explain MIL slice selection to clinicians and reviewers

### Academic Reporting

For research publications:

- **Figure quality**: High-resolution (300 DPI) publication-ready figures
- **Clear labeling**: All axes, legends, and titles are clearly labeled
- **Reproducibility**: Deterministic visualization ensures consistent results
- **Interpretability**: Visual comparison helps readers understand entropy-based selection

## Usage

### Basic Usage

```bash
# Generate visualization for a patient
python scripts/analysis/visualize_entropy.py \
    --patient-id Brats18_2013_10_1 \
    --modality flair
```

### Advanced Options

```bash
# Customize visualization
python scripts/analysis/visualize_entropy.py \
    --patient-id Brats18_2013_10_1 \
    --modality flair \
    --num-slices 3 \
    --output-dir results/entropy_visualization
```

### Parameters

- `--patient-id`: Patient identifier (required)
- `--modality`: Modality to visualize (t1, t1ce, t2, flair, default: flair)
- `--num-slices`: Number of example slices to show for high/low entropy (default: 3)
- `--entropy-dir`: Directory containing entropy JSON files (default: data/entropy)
- `--data-root`: Path to Stage 4 data (default: data/processed/stage_4_resize/train)
- `--output-dir`: Output directory for figures (default: results/entropy_visualization)

### Output

**Location**: `results/entropy_visualization/<patient_id>_entropy.png`

**Format**: PNG image, 300 DPI, publication-ready

**Structure**: 
- Top panel: Entropy curve with top-k slices highlighted
- Middle panel: High-entropy slices (most informative)
- Bottom panel: Low-entropy slices (least informative)

## Example Workflow

### 1. Generate Entropy Analysis (if not done)

```bash
# Run entropy analysis first
python scripts/analysis/run_entropy_analysis.py \
    --modality flair \
    --axis axial \
    --top-k 16
```

### 2. Create Visualizations

```bash
# Generate visualization for a specific patient
python scripts/analysis/visualize_entropy.py \
    --patient-id Brats18_2013_10_1 \
    --modality flair
```

### 3. Batch Processing (optional)

```bash
# Generate visualizations for multiple patients
for patient in $(ls data/entropy/*.json | sed 's/.*\///; s/_entropy.json//'); do
    python scripts/analysis/visualize_entropy.py \
        --patient-id "$patient" \
        --modality flair
done
```

## Medical and ML Justification

### Why Visualize Entropy Selection?

1. **Validation**: Verify that entropy-based selection captures meaningful slices
2. **Interpretability**: Understand why certain slices are selected for MIL training
3. **Debugging**: Identify issues in slice selection (e.g., if low-entropy slices are incorrectly selected)
4. **Communication**: Explain MIL slice selection to clinical collaborators and reviewers

### Clinical Relevance

For medical imaging applications:

- **Anatomical plausibility**: High-entropy slices should correspond to anatomically meaningful regions
- **Tumor visibility**: Tumor-containing slices typically have higher entropy due to intensity heterogeneity
- **Quality assurance**: Visualizations help ensure preprocessing and selection are working correctly

### Research Reproducibility

Visualizations support reproducibility by:

- **Documentation**: Providing visual evidence of slice selection criteria
- **Transparency**: Making selection methodology explicit and interpretable
- **Validation**: Allowing reviewers to verify entropy-based selection quality

## Integration with MIL Pipeline

### Workflow

1. **Preprocessing**: Stages 1-4 prepare 3D volumes (normalization, resizing)
2. **Entropy Analysis**: `run_entropy_analysis.py` computes entropy and selects top-k slices
3. **Visualization**: `visualize_entropy.py` creates figures for reporting (optional)
4. **Training**: MIL Dataset uses entropy metadata to extract top-k slices during training

### Visualization vs Training

- **Visualization**: For analysis, reporting, and interpretability (this tool)
- **Training**: Entropy metadata is used in MIL Dataset class to extract slices (separate code)

## Best Practices

### For Academic Reporting

1. **Select Representative Patients**: Visualize patients from both classes (HGG and LGG)
2. **Multiple Modalities**: Create visualizations for different modalities to show consistency
3. **Consistent Formatting**: Use default parameters for consistent figure style
4. **Include in Methods**: Reference entropy visualization in methodology sections

### For Model Development

1. **Validate Selection**: Check that high-entropy slices contain meaningful information
2. **Debug Issues**: Use visualizations to identify problems in slice selection
3. **Parameter Tuning**: Visualize different top-k values to understand selection impact
4. **Quality Assurance**: Verify entropy analysis is working correctly across the dataset

## Troubleshooting

### Missing Entropy Files

**Error**: `Entropy file not found`

**Solution**: Run entropy analysis first:
```bash
python scripts/analysis/run_entropy_analysis.py --modality flair
```

### Missing Volume Files

**Error**: `Volume file not found`

**Solution**: Verify Stage 4 outputs exist:
```bash
ls data/processed/stage_4_resize/train/HGG/<patient_id>/
```

### Matplotlib Issues

**Error**: Display-related errors

**Solution**: Use non-interactive backend (script handles this automatically):
```python
import matplotlib
matplotlib.use('Agg')
```

### Figure Quality

**Issue**: Low-resolution figures

**Solution**: Default DPI is 300 (publication-ready). For higher quality:
- Modify `dpi` parameter in `plt.savefig()` (default: 300)

## References

- Shannon Entropy: Information theory measure of uncertainty
- Multiple Instance Learning: Learning from sets of instances (slices)
- Medical Image Visualization: Best practices for publication-ready figures
- BraTS2018 Dataset: Brain tumor segmentation and classification challenge

## Summary

The entropy visualization tool provides a clear, interpretable way to understand how MIL models select informative slices from 3D MRI volumes. By visualizing entropy distributions and comparing high-entropy vs low-entropy slices, researchers can:

- **Validate** slice selection quality
- **Understand** why certain slices are selected
- **Communicate** MIL methodology to reviewers and clinicians
- **Debug** issues in entropy-based selection

This visualization supports model interpretability and academic reporting by making slice selection criteria transparent and visually accessible.

