# Stage 6: Class Balancing (Oversampling)

## Overview

Stage 6 provides **runtime class balancing** for handling class imbalance (HGG vs LGG) in the BraTS2018 dataset. Unlike previous preprocessing stages, Stage 6 does **NOT** create new files on disk. Instead, class balancing is applied dynamically within the DataLoader during training only via weighted sampling.

**Key Principles:**
- **Runtime-only**: Applied in DataLoader, no disk storage
- **Training-only**: Never applied to validation or test sets
- **Sampling-based**: Uses WeightedRandomSampler (no image synthesis)
- **Deterministic**: Supports reproducible sampling with seeds
- **Patient-safe**: Maintains patient-level atomicity

## Why Runtime-Only Class Balancing?

### Prevents Data Leakage
- **Validation/test integrity**: Validation and test sets remain unchanged, ensuring fair model evaluation
- **No contamination**: Training balancing does not affect evaluation metrics
- **Clear boundaries**: Separation between training and evaluation data is maintained

### Efficiency
- **No disk storage**: Avoids creating duplicate samples or modified files
- **Dynamic balancing**: Can adjust balancing strategy without reprocessing data
- **Memory efficient**: Sampling happens on-the-fly during DataLoader iteration

### Medical Imaging Best Practices
- **Preserves data integrity**: Original volumes remain unmodified
- **Reproducible**: Deterministic seeds ensure consistent sampling
- **Flexible**: Easy to experiment with different balancing strategies

## Requirements

- Python 3.7+
- PyTorch >= 1.8.0
- NumPy

Install dependencies:
```bash
pip install torch numpy
```

## Implementation Location

**File**: `utils/class_balancing.py`

**Functions**:
- `compute_class_weights()`: Compute class weights from label distribution
- `get_weighted_sampler()`: Create WeightedRandomSampler for balanced sampling
- `get_balanced_dataloader()`: Convenience function to create balanced DataLoader
- `get_class_distribution()`: Get class distribution statistics

## Class Imbalance in BraTS2018

The BraTS2018 dataset exhibits class imbalance:
- **HGG (High-Grade Glioma)**: ~210 cases (majority class)
- **LGG (Low-Grade Glioma)**: ~75 cases (minority class)
- **Imbalance ratio**: ~2.8:1 (HGG:LGG)

This imbalance can lead to:
- Model bias toward majority class
- Poor generalization on minority class
- Misleading accuracy metrics

## Balancing Strategy: WeightedRandomSampler

### Why WeightedRandomSampler?

We use **WeightedRandomSampler** (sampling-based approach) rather than image synthesis methods (SMOTE, ADASYN) for the following reasons:

#### 1. **Medical Image Quality**
- **No synthetic artifacts**: Sampling preserves original image quality
- **Anatomical accuracy**: Real medical images maintain anatomical plausibility
- **Clinical validity**: Synthetic images may introduce non-physiological features

#### 2. **3D Volume Complexity**
- **High dimensionality**: 3D volumes (128×128×128) are complex
- **Spatial relationships**: SMOTE/ADASYN struggle with 3D spatial structure
- **Multi-modal data**: BraTS has 4 modalities per patient, making synthesis complex

#### 3. **Patient-Level Integrity**
- **Patient atomicity**: Sampling maintains patient-level units (no mixing)
- **Reproducibility**: Deterministic sampling ensures consistent training
- **Memory efficiency**: No need to store synthetic samples

#### 4. **Training Efficiency**
- **On-the-fly balancing**: No preprocessing overhead
- **Flexible**: Easy to adjust weights without reprocessing
- **Compatible**: Works seamlessly with PyTorch DataLoader

### Weight Computation Strategies

1. **Inverse Frequency** (default):
   ```
   weight[i] = total_samples / (num_classes * class_count[i])
   ```
   - Most common approach
   - Minority class gets higher weight

2. **Balanced**:
   - Similar to inverse frequency
   - Normalized to ensure equal contribution

3. **Uniform**:
   - All classes have equal weight (no balancing)

## Usage

### Basic Usage in Training Pipeline

```python
from utils.class_balancing import get_balanced_dataloader
import torch
from torch.utils.data import Dataset, DataLoader

class BraTSDataset(Dataset):
    def __init__(self, data_path, mode="train"):
        self.data_path = data_path
        self.mode = mode
        # ... load data paths and labels ...
    
    def get_all_labels(self):
        """Return array of all labels in dataset."""
        return np.array([self[i][1] for i in range(len(self))])
    
    def __getitem__(self, idx):
        # ... load volume and label ...
        return volume, label

# Create datasets
train_dataset = BraTSDataset(data_path, mode="train")
val_dataset = BraTSDataset(data_path, mode="val")

# Get labels for training dataset
train_labels = train_dataset.get_all_labels()

# Create balanced training DataLoader
train_loader = get_balanced_dataloader(
    dataset=train_dataset,
    labels=train_labels,  # Enables class balancing
    batch_size=8,
    num_workers=4,
    strategy="inverse_freq",
    seed=42  # For reproducibility
)

# Create validation DataLoader (no balancing)
val_loader = get_balanced_dataloader(
    dataset=val_dataset,
    labels=None,  # No balancing for validation
    batch_size=8,
    num_workers=4,
    shuffle=False
)
```

### Manual Sampler Creation

```python
from utils.class_balancing import get_weighted_sampler, compute_class_weights
from torch.utils.data import DataLoader

# Get labels
train_labels = train_dataset.get_all_labels()

# Compute class weights
weights = compute_class_weights(train_labels, strategy="inverse_freq")
print(f"Class weights: {weights}")  # [weight_LGG, weight_HGG]

# Create sampler
sampler = get_weighted_sampler(
    labels=train_labels,
    strategy="inverse_freq",
    num_samples=len(train_dataset),
    replacement=True,
    generator=torch.Generator().manual_seed(42)
)

# Create DataLoader with sampler
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    sampler=sampler,  # Uses weighted sampling
    num_workers=4
)
```

### Class Distribution Analysis

```python
from utils.class_balancing import get_class_distribution

# Get distribution statistics
stats = get_class_distribution(
    train_labels,
    class_names=["LGG", "HGG"]
)

print(f"LGG: {stats['LGG']['count']} samples ({stats['LGG']['percentage']:.1f}%)")
print(f"HGG: {stats['HGG']['count']} samples ({stats['HGG']['percentage']:.1f}%)")
print(f"Imbalance ratio: {stats['ratio']:.3f}")
print(f"LGG weight: {stats['LGG']['weight']:.4f}")
print(f"HGG weight: {stats['HGG']['weight']:.4f}")
```

## Integration with Training Loop

### Example: PyTorch Training with Class Balancing

```python
import torch
from utils.class_balancing import get_balanced_dataloader

# Create datasets
train_dataset = BraTSDataset(data_path, mode="train")
val_dataset = BraTSDataset(data_path, mode="val")

# Get training labels
train_labels = train_dataset.get_all_labels()

# Create balanced training DataLoader
train_loader = get_balanced_dataloader(
    train_dataset,
    labels=train_labels,
    batch_size=8,
    seed=42
)

# Create validation DataLoader (no balancing)
val_loader = get_balanced_dataloader(
    val_dataset,
    labels=None,  # No balancing
    batch_size=8,
    shuffle=False
)

# Training loop
model.train()
for epoch in range(num_epochs):
    for batch in train_loader:
        volumes, labels = batch
        # Labels are now balanced (more LGG samples per epoch)
        outputs = model(volumes)
        loss = criterion(outputs, labels)
        # ... backprop ...

# Validation loop (no balancing)
model.eval()
with torch.no_grad():
    for batch in val_loader:
        volumes, labels = batch
        # Labels maintain original distribution
        outputs = model(volumes)
        # ... evaluation ...
```

## Why NOT SMOTE/ADASYN for 3D Medical Images?

### 1. **Image Synthesis Quality**
- **Synthetic artifacts**: SMOTE/ADASYN can introduce unrealistic features in 3D medical images
- **Anatomical plausibility**: Generated images may violate anatomical constraints
- **Clinical validity**: Synthetic samples may not represent real pathology

### 2. **High Dimensionality**
- **3D complexity**: 128×128×128 = 2,097,152 voxels per volume
- **Spatial relationships**: SMOTE struggles with preserving 3D spatial structure
- **Multi-modal**: BraTS has 4 modalities (t1, t1ce, t2, flair), making synthesis complex

### 3. **Patient-Level Integrity**
- **Atomic units**: Patients should remain atomic (no mixing between patients)
- **Reproducibility**: Synthetic samples may introduce variability
- **Memory**: Storing synthetic samples requires significant disk space

### 4. **Medical Imaging Best Practices**
- **Data integrity**: Original medical images should remain unmodified
- **Regulatory compliance**: Synthetic data may raise regulatory concerns
- **Publication standards**: Many medical imaging publications prefer real data

## Medical and ML Justification

### Medical Justification

1. **Preserves Clinical Validity**: 
   - Original medical images maintain their clinical significance
   - No introduction of synthetic artifacts that could mislead the model

2. **Anatomical Accuracy**:
   - Real MRI volumes preserve anatomical relationships
   - Synthetic samples may violate spatial constraints

3. **Regulatory Compliance**:
   - Real data maintains traceability and auditability
   - Synthetic data may complicate regulatory approval

### ML Justification

1. **Prevents Overfitting to Synthetic Data**:
   - Models trained on synthetic samples may not generalize to real data
   - Weighted sampling uses only real data

2. **Computational Efficiency**:
   - Sampling is O(n) complexity, synthesis is often O(n²) or higher
   - No preprocessing overhead

3. **Flexibility**:
   - Easy to experiment with different balancing strategies
   - Can adjust weights without reprocessing data

4. **Standard Practice**:
   - WeightedRandomSampler is the standard approach in medical imaging ML
   - Widely used in research publications

## Compatibility

### MIL (Multiple Instance Learning)
- Works with slice-based datasets
- Sampling applies to patient-level units
- Compatible with MIL's instance-level processing

### ResNet50-3D
- Works with full 3D volume datasets
- Sampling maintains batch structure
- Compatible with 3D CNN architectures

### Swin UNETR
- Works with transformer-based architectures
- Sampling preserves sequence structure
- Compatible with attention mechanisms

## Verification Checklist

✅ **No files created on disk**
- Class balancing is applied in DataLoader only
- No files written to `data/processed/stage_6_balanced/`

✅ **Training data is balanced**
- Minority class (LGG) is sampled more frequently
- Class distribution in batches is more balanced

✅ **Validation/test data is NOT balanced**
- Validation/test sets maintain original distribution
- No WeightedRandomSampler used for validation/test

✅ **Deterministic with seed**
- Same seed produces same sampling sequence
- Reproducible across training runs

✅ **Patient-level atomicity maintained**
- No mixing between patients
- Each sample is a complete patient volume

## Input Data

**Source**: `data/processed/stage_4_resize/`

**Format**:
- Shape: (128, 128, 128)
- Single channel (grayscale MRI)
- Modalities: t1, t1ce, t2, flair (separate files)
- Labels: 0 (LGG) or 1 (HGG)

**Class Distribution**:
- HGG: ~210 cases (majority)
- LGG: ~75 cases (minority)
- Imbalance ratio: ~2.8:1

## Output

**NO files are created on disk.**

Class balancing happens:
- **In memory**: During DataLoader iteration
- **On-the-fly**: Different sampling each epoch (with seed)
- **Training only**: Not applied to validation/test

## Reproducibility

To ensure reproducibility:

```python
import torch
import numpy as np
import random

# Set seeds
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Create sampler with seed
sampler = get_weighted_sampler(
    labels=train_labels,
    generator=torch.Generator().manual_seed(seed)
)
```

## Troubleshooting

### Class Imbalance Still Present

- Check that `labels` parameter is provided to `get_balanced_dataloader()`
- Verify `strategy` parameter (try "inverse_freq" or "balanced")
- Check class distribution with `get_class_distribution()`

### Validation Data Being Balanced

Ensure validation DataLoader uses:
```python
val_loader = get_balanced_dataloader(
    val_dataset,
    labels=None,  # No balancing
    shuffle=False
)
```

### Memory Issues

- Reduce `num_samples` if using custom sampling
- Use smaller `batch_size`
- Reduce `num_workers` if memory-constrained

## Next Steps

After Stage 6 implementation:
- Integrate into training DataLoader
- Monitor class distribution in batches
- Verify validation/test remain unbalanced
- Evaluate model performance with/without balancing
- Proceed to model training (MIL, ResNet50-3D, Swin UNETR)

## References

- PyTorch WeightedRandomSampler: https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler
- Class Imbalance in Medical Imaging: Common challenge in medical ML
- BraTS2018 Dataset: https://www.med.upenn.edu/cbica/brats2018/
- Medical Image Classification: Best practices for handling class imbalance

## Academic Reproducibility

This class balancing pipeline is designed for:
- Academic thesis work
- Medical AI research publications
- Public GitHub repositories

All parameters are configurable, code is modular and well-documented, and the pipeline follows medical imaging ML best practices.

