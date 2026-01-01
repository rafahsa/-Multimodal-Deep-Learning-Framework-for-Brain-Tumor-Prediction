# Stage 5: Geometric Data Augmentation

## Overview

Stage 5 provides **on-the-fly geometric data augmentation** for 3D MRI volumes during training. Unlike previous preprocessing stages, Stage 5 does **NOT** create new files on disk. Instead, augmentation is applied dynamically within the DataLoader during training only.

**Key Principles:**
- **On-the-fly**: Applied in DataLoader, no disk storage
- **Train-only**: No augmentation during validation or testing
- **Medical-safe**: Preserves anatomical plausibility
- **Compatible**: Works with MIL (slice-based), ResNet50-3D, Swin UNETR

## Why On-the-Fly Augmentation?

### Memory Efficiency
- **No disk storage**: Augmented volumes are not saved, saving terabytes of disk space
- **Dynamic generation**: Each epoch sees different augmented versions
- **Infinite variety**: Same volume → different augmentations per epoch

### Correctness
- **Training only**: Validation/test sets remain unchanged (critical for fair evaluation)
- **Reproducibility**: Deterministic seeds can be set for reproducibility
- **No data leakage**: Augmentation boundaries are clear (train vs. val/test)

### Medical Imaging Best Practices
- **Anatomical preservation**: Augmentation is applied in a controlled, medical-safe manner
- **No artifacts**: On-the-fly augmentation avoids introducing file corruption or metadata issues
- **Flexibility**: Easy to adjust augmentation parameters without reprocessing data

## Requirements

- Python 3.7+
- MONAI (Medical Open Network for AI)
- PyTorch
- NumPy

Install dependencies:
```bash
pip install monai torch numpy
```

## Implementation Location

**File**: `utils/augmentations_3d.py`

**Functions**:
- `get_train_transforms_3d()`: Training augmentation pipeline
- `get_val_transforms_3d()`: Validation/test pipeline (no augmentation)
- `get_transforms_3d(mode)`: Convenience function
- Model-specific functions: `get_mil_transforms_3d()`, `get_resnet3d_transforms_3d()`, `get_swin_unetr_transforms_3d()`

## Augmentation Strategy

### Allowed Transforms (Medical-Safe)

1. **Random Rotation** (±15 degrees)
   - Rotates around x, y, z axes independently
   - Range: ±15 degrees (≈ 0.26 radians)
   - Preserves anatomical structure
   - Uses bilinear interpolation

2. **Random Flip** (x, y, z axes)
   - Flips along each axis independently
   - Probability: 50% per axis
   - Anatomically plausible (brain symmetry)

3. **Random Zoom** (±10%)
   - Zoom range: 0.9 to 1.1 (90% to 110%)
   - Maintains relative proportions
   - Uses trilinear interpolation

4. **Random Translation** (±10% of volume size)
   - Translates along x, y, z axes
   - Keeps brain within field of view
   - Combined with rotation for realism

### NOT Allowed

❌ **Elastic deformation**: Can introduce non-physiological artifacts  
❌ **Intensity distortion**: Already normalized in Stage 2  
❌ **Random cropping**: Volumes already cropped to ROI in Stage 3  
❌ **Non-linear warping**: Preserves anatomical realism  

## Usage

### Basic Usage in DataLoader

```python
from utils.augmentations_3d import get_transforms_3d
import torch
from torch.utils.data import Dataset, DataLoader

class BraTSDataset(Dataset):
    def __init__(self, data_path, mode="train"):
        self.data_path = data_path
        self.mode = mode
        # Get appropriate transforms
        self.transforms = get_transforms_3d(mode=mode)
        # ... load data paths ...
    
    def __getitem__(self, idx):
        # Load volume (shape: 128×128×128)
        volume = load_volume(self.data_paths[idx])
        
        # Prepare dict for MONAI transforms
        data_dict = {"image": volume}
        
        # Apply transforms (augmentation if train, passthrough if val)
        data_dict = self.transforms(data_dict)
        
        # Extract tensor
        volume_tensor = data_dict["image"]
        
        return volume_tensor, label
```

### Model-Specific Usage

```python
# For MIL (slice-based)
from utils.augmentations_3d import get_mil_transforms_3d
train_transforms = get_mil_transforms_3d("train")

# For ResNet50-3D
from utils.augmentations_3d import get_resnet3d_transforms_3d
train_transforms = get_resnet3d_transforms_3d("train")

# For Swin UNETR
from utils.augmentations_3d import get_swin_unetr_transforms_3d
train_transforms = get_swin_unetr_transforms_3d("train")
```

### Custom Parameters

```python
from utils.augmentations_3d import get_train_transforms_3d

# Custom augmentation parameters
transforms = get_train_transforms_3d(
    rotation_range=(0.2, 0.2, 0.2),  # ±11.5 degrees
    flip_prob=0.5,
    zoom_range=(0.95, 1.05),  # ±5%
    translation_range=(0.05, 0.05, 0.05),  # ±5%
    prob=0.5
)
```

## Integration with Training Loop

### Example: PyTorch Training

```python
import torch
from torch.utils.data import DataLoader
from utils.augmentations_3d import get_transforms_3d

# Create datasets
train_dataset = BraTSDataset(data_path, mode="train")
val_dataset = BraTSDataset(data_path, mode="val")

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Training loop
model.train()  # Enable training mode
for epoch in range(num_epochs):
    for batch in train_loader:
        volumes, labels = batch
        # Volumes are augmented here (different each epoch)
        outputs = model(volumes)
        loss = criterion(outputs, labels)
        # ... backprop ...

# Validation loop
model.eval()  # Disable training mode
with torch.no_grad():
    for batch in val_loader:
        volumes, labels = batch
        # Volumes are NOT augmented (original data)
        outputs = model(volumes)
        # ... evaluation ...
```

## Medical Justification

### Why These Augmentations?

1. **Rotation (±15°)**: 
   - Represents natural head positioning variation
   - Small enough to preserve anatomical structure
   - Common in clinical imaging

2. **Flip**:
   - Brain has approximate bilateral symmetry
   - Common in medical imaging augmentation
   - Preserves anatomical plausibility

3. **Zoom (±10%)**:
   - Represents field-of-view variation
   - Maintains relative proportions
   - Realistic scanner variation

4. **Translation (±10%)**:
   - Represents patient positioning variation
   - Keeps brain within field of view
   - Realistic clinical variation

### Why NOT Other Augmentations?

- **Elastic deformation**: Can introduce non-physiological artifacts
- **Intensity augmentation**: Already normalized in Stage 2
- **Random cropping**: Volumes already cropped to ROI in Stage 3
- **Non-linear warping**: Preserves anatomical realism

## Compatibility

### MIL (Multiple Instance Learning)

MIL models work with 2D slices extracted from 3D volumes. Stage 5 augmentation:
- Can be applied to 3D volumes before slicing
- Or adapted for slice-level augmentation
- Function: `get_mil_transforms_3d()`

### ResNet50-3D

ResNet50-3D processes full 3D volumes:
- Standard 3D augmentation applies directly
- Function: `get_resnet3d_transforms_3d()`

### Swin UNETR

Swin UNETR also processes full 3D volumes:
- Standard 3D augmentation applies directly
- Function: `get_swin_unetr_transforms_3d()`

## Verification Checklist

✅ **No new files created on disk**
- Augmentation is applied in DataLoader only
- No files written to `data/processed/stage_5_*`

✅ **Same input volume → different augmented outputs per epoch**
- Each epoch sees different augmented versions
- Random seed can be set for reproducibility

✅ **Validation samples remain unchanged**
- Validation/test sets use `get_val_transforms_3d()` (no augmentation)
- Original volumes are preserved

✅ **Shape remains (128, 128, 128)**
- All transforms maintain output size
- `keep_size=True` in MONAI transforms

✅ **Augmentation disabled when `model.eval()`**
- Training mode: augmentation applied
- Evaluation mode: no augmentation

## Input Data

**Source**: `data/processed/stage_4_resize/`

**Format**:
- Shape: (128, 128, 128)
- Single channel (grayscale MRI)
- Modalities: t1, t1ce, t2, flair (separate files)
- Spacing & metadata: Preserved from Stage 4

**Example structure**:
```
data/processed/stage_4_resize/
└── train/
    ├── HGG/
    │   └── <patient_id>/
    │       ├── <patient_id>_t1.nii.gz
    │       ├── <patient_id>_t1ce.nii.gz
    │       ├── <patient_id>_t2.nii.gz
    │       └── <patient_id>_flair.nii.gz
    └── LGG/
        └── <patient_id>/
            └── <patient_id>_<modality>.nii.gz
```

## Output

**NO files are created on disk.**

Augmentation happens:
- **In memory**: During DataLoader iteration
- **On-the-fly**: Different each epoch
- **Training only**: Not applied to validation/test

## Reproducibility

To ensure reproducibility:

```python
import random
import numpy as np
import torch

# Set seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# MONAI transforms will use these seeds
```

## Troubleshooting

### MONAI Not Installed

```bash
pip install monai
```

### Shape Mismatch Errors

- Ensure input volumes are (128, 128, 128)
- Check that `EnsureChannelFirst` is applied
- Verify MONAI transform compatibility

### Augmentation Too Strong/Weak

Adjust parameters in `get_train_transforms_3d()`:
- `rotation_range`: Reduce for less rotation
- `zoom_range`: Narrow for less zoom variation
- `prob`: Reduce for less frequent augmentation

### Validation Data Being Augmented

Ensure validation dataset uses:
```python
val_transforms = get_val_transforms_3d()  # No augmentation
```

## Next Steps

After Stage 5 implementation:
- Integrate into DataLoader
- Test with training loop
- Verify augmentation is train-only
- Monitor training with/without augmentation
- Proceed to model training (MIL, ResNet50-3D, Swin UNETR)

## References

- MONAI Documentation: https://docs.monai.io/
- MONAI Transforms: https://docs.monai.io/en/stable/transforms.html
- Medical Image Augmentation: Best practices for 3D medical imaging
- BraTS2018 Dataset: https://www.med.upenn.edu/cbica/brats2018/

## Academic Reproducibility

This augmentation pipeline is designed for:
- Academic thesis work
- Medical AI research publications
- Public GitHub repositories

All parameters are configurable, code is modular and well-documented, and the pipeline follows medical imaging best practices.

