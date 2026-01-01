# Entropy-based Slice Selection (MIL-only)

## Overview

This stage provides entropy-based slice informativeness analysis for Multiple Instance Learning (MIL) models. It computes Shannon entropy for each 2D slice in 3D MRI volumes to identify the most informative slices for training.

**Key Principles:**
- **MIL-specific**: Used exclusively with slice-based MIL models
- **Metadata-only**: Only creates JSON metadata files, no image modification
- **Runtime metadata**: Analysis results stored for use during training
- **Not a preprocessing stage**: Does not modify Stages 1-4 outputs

## Why Entropy for Slice Selection?

### Informativeness Measure

Shannon entropy quantifies the information content in a slice:
- **High entropy**: More diverse intensity distribution → more informative
- **Low entropy**: Uniform or sparse distribution → less informative

**Formula**: H(X) = -Σ p(x) × log₂(p(x))

### Medical Imaging Rationale

In brain MRI:
- **High-entropy slices**: Contain brain tissue with varying intensities (gray matter, white matter, CSF, lesions)
- **Low-entropy slices**: Mostly background (zeros) or uniform regions
- **Tumor regions**: Typically have high entropy due to intensity heterogeneity

### MIL Model Integration

MIL models process 2D slices extracted from 3D volumes:
- **Problem**: Processing all 128 slices is computationally expensive
- **Solution**: Select top-k most informative slices based on entropy
- **Benefit**: Reduces computation while focusing on informative regions

## Why MIL-Only?

This entropy analysis is **exclusive to MIL models** because:

1. **MIL Architecture**: MIL models are designed to work with 2D slices/instances
2. **ResNet50-3D**: Processes full 3D volumes directly, no slice selection needed
3. **Swin UNETR**: Processes full 3D volumes with attention, no slice selection needed
4. **Slice-Based Processing**: Only MIL benefits from pre-selecting informative slices

## Why Entropy Is Used Only for MIL

### Architectural Differences

**MIL Models (Entropy-Enabled)**:
- Process 2D slices extracted from 3D volumes
- Treat each slice as an "instance" in a "bag" (the patient)
- Learn which instances (slices) are most relevant for classification
- Benefit from pre-selecting informative slices to reduce computation and focus on discriminative regions
- Entropy-based selection helps identify slices with high information content

**ResNet50-3D (Entropy NOT Used)**:
- Processes **full 3D volumes** directly (shape: 128×128×128)
- Uses 3D convolutional operations that capture spatial relationships across all dimensions
- Requires complete volume information to learn 3D spatial features
- **Slice selection would destroy 3D spatial context** essential for 3D CNNs
- Entropy-based slicing would discard important 3D anatomical information

**Swin UNETR (Entropy NOT Used)**:
- Processes **full 3D volumes** using transformer-based attention mechanisms
- Uses windowed attention to capture long-range dependencies in 3D space
- Requires complete volume to compute attention across spatial dimensions
- **Slice selection would break 3D attention patterns** that are core to the model
- Entropy-based slicing would remove critical 3D structural information

### Medical Imaging Justification

**Why Entropy Works for MIL**:
- MIL models operate on 2D slices, so selecting informative 2D slices is natural
- High-entropy slices contain diverse tissue patterns that are discriminative
- Reducing slice count (128 → 16) improves computational efficiency without losing key information
- Slice-level attention in MIL can focus on selected high-entropy slices

**Why Entropy Does NOT Work for 3D CNNs**:
- 3D CNNs learn features from **3D spatial context** (e.g., tumor shape, spatial relationships)
- Removing slices breaks spatial continuity and destroys 3D anatomical structure
- 3D convolutions require **complete volume** to learn meaningful 3D features
- Entropy-based selection would introduce **artificial spatial gaps** that harm performance
- 3D models benefit from all available spatial information, not just high-entropy regions

### Design Principle

**Separation of Concerns**:
- **MIL**: Slice-based processing → entropy selection is beneficial
- **3D CNNs**: Volume-based processing → entropy selection is harmful
- Each model type uses data in its optimal format
- Entropy is a MIL-specific optimization, not a general preprocessing step

### Implementation Note

The `MILDataset` class (`utils/mil_dataset.py`) includes entropy support with `use_entropy=True`, while ResNet50-3D and Swin UNETR datasets use full 3D volumes without any slice selection. This design ensures:
- MIL models can leverage entropy-based slice selection for efficiency
- 3D CNNs receive complete volumes with full spatial context
- No cross-contamination between model-specific data loading strategies

## Why Metadata-Only?

This stage creates **metadata files only** (no image modification) because:

1. **No Preprocessing**: Entropy is computed but not applied to images
2. **Training-Time Usage**: Slice selection happens during DataLoader iteration
3. **Flexibility**: Can adjust top-k without reprocessing
4. **Storage Efficiency**: Small JSON files vs. large processed volumes

## Requirements

- Python 3.7+
- PyTorch >= 1.8.0
- SimpleITK
- NumPy

Install dependencies:
```bash
pip install torch SimpleITK numpy
```

## Implementation Location

**Utility Module**: `utils/entropy_analysis.py`
- `compute_slice_entropy()`: Compute entropy per slice
- `select_top_k_slices()`: Select top-k most informative slices

**Runner Script**: `scripts/analysis/run_entropy_analysis.py`
- Processes all volumes from Stage 4
- Generates JSON metadata files

## Usage

### Run Entropy Analysis

```bash
# Basic usage (default: flair modality, axial axis, top-16 slices)
python scripts/analysis/run_entropy_analysis.py

# Custom parameters
python scripts/analysis/run_entropy_analysis.py \
    --modality flair \
    --axis axial \
    --top-k 16 \
    --num-bins 256 \
    --normalize \
    --use-cuda
```

**Parameters**:
- `--modality`: Modality to analyze (t1, t1ce, t2, flair)
- `--axis`: Slice axis (axial, coronal, sagittal)
- `--top-k`: Number of top slices to select (default: 16)
- `--num-bins`: Histogram bins for entropy computation (default: 256)
- `--normalize`: Normalize entropy to [0, 1] range
- `--use-cuda`: Use GPU if available
- `--classes`: Classes to process (default: HGG LGG)

### Output Format

Each patient generates a JSON file: `data/entropy/<patient_id>_entropy.json`

**Example JSON**:
```json
{
  "patient_id": "Brats18_2013_10_1",
  "axis": "axial",
  "entropy_per_slice": [2.45, 2.67, 3.12, ...],
  "top_k": 16,
  "top_k_slices": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57],
  "num_slices": 128
}
```

## Integration with MIL Training

### Loading Entropy Metadata

```python
import json
from pathlib import Path

def load_entropy_metadata(patient_id, entropy_dir='data/entropy'):
    """Load entropy metadata for a patient."""
    entropy_file = Path(entropy_dir) / f"{patient_id}_entropy.json"
    with open(entropy_file, 'r') as f:
        return json.load(f)

# Load metadata
metadata = load_entropy_metadata("Brats18_2013_10_1")
top_slices = metadata['top_k_slices']
```

### Using in MIL Dataset

```python
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np

class MILDataset(Dataset):
    def __init__(self, split_csv, entropy_dir='data/entropy'):
        self.entropy_dir = Path(entropy_dir)
        # ... load patients from split_csv ...
    
    def __getitem__(self, idx):
        patient = self.patients[idx]
        patient_id = patient['patient_id']
        
        # Load volume
        volume_path = self.data_root / patient['path_flair']
        volume = sitk.ReadImage(str(volume_path))
        volume_array = sitk.GetArrayFromImage(volume)  # Shape: (D, H, W)
        
        # Load entropy metadata
        entropy_file = self.entropy_dir / f"{patient_id}_entropy.json"
        with open(entropy_file, 'r') as f:
            entropy_data = json.load(f)
        
        # Extract top-k slices
        top_slice_indices = entropy_data['top_k_slices']
        slices = [volume_array[i, :, :] for i in top_slice_indices]
        
        # Stack slices
        slice_stack = np.stack(slices, axis=0)  # Shape: (k, H, W)
        
        return torch.tensor(slice_stack), label
```

## Algorithm Details

### Entropy Computation

1. **Slice Extraction**: Extract 2D slices along specified axis (axial/coronal/sagittal)
2. **Histogram Computation**: Compute intensity histogram for each slice
3. **Probability Estimation**: Normalize histogram to get probability distribution
4. **Entropy Calculation**: H(X) = -Σ p(x) × log₂(p(x))
5. **Slice Ranking**: Rank slices by entropy (highest = most informative)

### Top-K Selection

1. **Sort by Entropy**: Sort slices by entropy score (descending)
2. **Select Top-K**: Take k slices with highest entropy
3. **Return Indices**: Return slice indices in original order (ascending)

### Axes

- **Axial** (z-axis): Most common for brain MRI (horizontal slices)
- **Coronal** (y-axis): Front-to-back slices
- **Sagittal** (x-axis): Left-to-right slices

## Input Data

**Source**: `data/processed/stage_4_resize/train/`

**Format**:
- Shape: (128, 128, 128)
- Single channel (grayscale MRI)
- Modalities: t1, t1ce, t2, flair (separate files)
- Already normalized and resized (from Stages 1-4)

## Output

**Location**: `data/entropy/`

**Format**: JSON files (one per patient)
- File naming: `<patient_id>_entropy.json`
- Small files (~3KB each)
- Contains entropy scores and top-k slice indices

**Example structure**:
```
data/entropy/
├── Brats18_2013_10_1_entropy.json
├── Brats18_2013_11_1_entropy.json
└── ...
```

## Medical and ML Justification

### Why Entropy Works for Slice Selection

1. **Information Content**: Entropy quantifies information diversity
2. **Tumor Regions**: High entropy indicates heterogeneous tissue (tumor characteristics)
3. **Background Filtering**: Low entropy slices (mostly background) are filtered out
4. **Computational Efficiency**: Focuses computation on informative regions

### Why Not Use All Slices?

1. **Computational Cost**: Processing 128 slices per volume is expensive
2. **Redundant Information**: Many slices contain similar or less informative content
3. **MIL Efficiency**: MIL models benefit from focusing on informative instances
4. **Training Speed**: Fewer slices = faster training iterations

### Why Not Intensity-Based Selection?

1. **Distribution Matters**: Entropy considers intensity distribution, not just values
2. **Robustness**: Less sensitive to absolute intensity values (already normalized)
3. **Information Theory**: Entropy is a principled measure of information content

## Verification

### Check Generated Files

```bash
# Count entropy files
find data/entropy -name "*.json" | wc -l
# Should match number of patients processed

# Inspect a sample file
cat data/entropy/Brats18_2013_10_1_entropy.json | python3 -m json.tool
```

### Validate JSON Structure

```python
import json
import glob

# Check all JSON files are valid
for json_file in glob.glob('data/entropy/*.json'):
    with open(json_file, 'r') as f:
        data = json.load(f)
        assert 'patient_id' in data
        assert 'axis' in data
        assert 'entropy_per_slice' in data
        assert 'top_k_slices' in data
        assert len(data['top_k_slices']) == data['top_k']
```

## Troubleshooting

### CUDA Out of Memory

- Disable `--use-cuda` flag
- Process fewer patients at a time
- Reduce batch processing

### Slow Processing

- Use `--use-cuda` if GPU available
- Process specific classes: `--classes HGG`
- Reduce `--num-bins` (faster but less precise)

### Missing JSON Files

- Verify Stage 4 volumes exist
- Check file permissions
- Verify modality files are present

## Integration Checklist

✅ **Entropy analysis complete**
- All patients have entropy JSON files
- Top-k slices identified for each patient

✅ **MIL Dataset Integration**
- Load entropy metadata in Dataset class
- Extract top-k slices during `__getitem__()`
- Use slices for MIL model training

✅ **Training Loop**
- MIL model receives top-k slices per patient
- Slices are processed as instances
- Model learns from informative slices

## Next Steps

After entropy analysis:
- Integrate entropy metadata into MIL Dataset
- Extract top-k slices during training
- Train MIL model with informative slices
- Evaluate model performance

## References

- Shannon Entropy: Information theory measure of uncertainty
- Multiple Instance Learning: Learning from sets of instances
- BraTS2018 Dataset: https://www.med.upenn.edu/cbica/brats2018/
- Slice Selection in Medical Imaging: Common practice for 2D MIL models

## Academic Reproducibility

This entropy analysis pipeline is designed for:
- Academic thesis work
- Medical AI research publications
- Public GitHub repositories

All parameters are configurable, code is modular and well-documented, and the pipeline follows medical imaging ML best practices for MIL models.

