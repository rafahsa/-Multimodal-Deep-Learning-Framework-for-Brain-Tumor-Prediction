# Stage 7: K-Fold Cross-Validation Split

## Overview

Stage 7 generates patient-level K-Fold cross-validation splits for the BraTS2018 dataset. Unlike preprocessing stages 1-4 that process image data, Stage 7 creates **metadata files only** (CSV and JSON) that define train/validation splits. No image files are moved, copied, or modified.

**Key Principles:**
- **Patient-level splitting**: Prevents data leakage (entire patient stays in one fold)
- **Stratified**: Preserves class ratio (HGG:LGG) in each fold
- **Reproducible**: Fixed random seed (42) ensures consistent splits
- **Metadata-only**: Only creates split definition files, no data movement
- **Git-friendly**: Split files are small and can be version-controlled

## Requirements

- Python 3.7+
- scikit-learn (for StratifiedKFold)
- NumPy

Install dependencies:
```bash
pip install scikit-learn numpy
```

## Implementation

### Stage 7 consists of two scripts:

1. **Index Builder**: `scripts/splits/build_stage4_index.py`
   - Scans Stage 4 output directory
   - Creates patient index file with metadata
   - Output: `data/index/stage4_index.csv`

2. **Split Generator**: `scripts/splits/make_kfold_splits.py`
   - Generates K-Fold splits using StratifiedKFold
   - Creates split definition files (CSV + JSON)
   - Output: `splits/` directory with split files

## Usage

### Step 1: Build Patient Index

```bash
python scripts/splits/build_stage4_index.py
```

This scans `data/processed/stage_4_resize/train/` and creates:
- `data/index/stage4_index.csv`: CSV file with patient metadata

**Output format**:
- `patient_id`: Patient identifier
- `class`: Class label (HGG or LGG)
- `class_label`: Numeric label (1=HGG, 0=LGG)
- `path_t1`, `path_t1ce`, `path_t2`, `path_flair`: Relative paths to modality files

### Step 2: Generate K-Fold Splits

```bash
python scripts/splits/make_kfold_splits.py --k 5 --seed 42
```

**Parameters**:
- `--index`: Path to index file (default: `data/index/stage4_index.csv`)
- `--k`: Number of folds (default: 5)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output-dir`: Output directory (default: `splits`)
- `--output-json`: JSON summary file (default: `splits/kfold_5fold_seed42.json`)

**Output files**:
- `splits/kfold_5fold_seed42.json`: JSON summary of all splits
- `splits/fold_0_train.csv`: Training patients for fold 0
- `splits/fold_0_val.csv`: Validation patients for fold 0
- ... (repeated for folds 1-4)

## Output Files

### JSON Summary

`splits/kfold_5fold_seed42.json` contains:
```json
{
  "k": 5,
  "seed": 42,
  "total_patients": 285,
  "folds": [
    {
      "fold": 0,
      "train_patients": [...],
      "val_patients": [...],
      "train_count": 228,
      "val_count": 57
    },
    ...
  ]
}
```

### CSV Files per Fold

Each fold has two CSV files:
- `fold_X_train.csv`: Training set for fold X
- `fold_X_val.csv`: Validation set for fold X

Each CSV contains the same columns as the index file:
- `patient_id`, `class`, `class_label`, `path_t1`, `path_t1ce`, `path_t2`, `path_flair`

## Verification

The split generator automatically verifies:

1. **No Patient Overlap**: Each patient appears in either train OR val, never both
2. **Class Distribution**: Class ratio (HGG:LGG) is preserved in each fold
3. **Patient Counts**: Total patients in train + val = total patients

Example verification output:
```
============================================================
SPLIT VERIFICATION
============================================================
✓ Fold 0: No patient overlap
  Train: 228 patients (HGG: 168, LGG: 60)
  Val:   57 patients (HGG: 42, LGG: 15)
  Train HGG ratio: 0.737, Val HGG ratio: 0.737
...
✓ All verifications passed!
```

## Integration with Training

### Loading Splits in Python

```python
import csv
from pathlib import Path

def load_split(csv_path):
    """Load a split CSV file."""
    patients = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patients.append(row)
    return patients

# Load fold 0 training set
train_patients = load_split('splits/fold_0_train.csv')
val_patients = load_split('splits/fold_0_val.csv')

print(f"Train: {len(train_patients)} patients")
print(f"Val: {len(val_patients)} patients")
```

### Using with Dataset Class

```python
from torch.utils.data import Dataset
import SimpleITK as sitk

class BraTSDataset(Dataset):
    def __init__(self, split_csv, data_root):
        self.data_root = Path(data_root)
        # Load split
        self.patients = []
        with open(split_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.patients.append(row)
    
    def __getitem__(self, idx):
        patient = self.patients[idx]
        # Build full path
        flair_path = self.data_root / patient['path_flair']
        # Load volume
        volume = sitk.ReadImage(str(flair_path))
        # ... process ...
        return volume, int(patient['class_label'])

# Create datasets for fold 0
data_root = 'data/processed/stage_4_resize/train'
train_dataset = BraTSDataset('splits/fold_0_train.csv', data_root)
val_dataset = BraTSDataset('splits/fold_0_val.csv', data_root)
```

### Cross-Validation Loop

```python
from pathlib import Path

k_folds = 5
for fold in range(k_folds):
    print(f"\n=== Fold {fold} ===")
    
    train_csv = f'splits/fold_{fold}_train.csv'
    val_csv = f'splits/fold_{fold}_val.csv'
    
    train_dataset = BraTSDataset(train_csv, data_root)
    val_dataset = BraTSDataset(val_csv, data_root)
    
    # Train model
    model = train_model(train_dataset)
    
    # Evaluate
    metrics = evaluate_model(model, val_dataset)
    print(f"Fold {fold} metrics: {metrics}")
```

## Why Patient-Level Splitting?

### Prevents Data Leakage

- **Patient atomicity**: All data from the same patient stays together
- **No information leakage**: Training cannot see validation patients
- **Realistic evaluation**: Mimics real-world deployment scenario

### Medical Imaging Best Practices

- **Clinical validity**: Patients are independent units in medical imaging
- **Regulatory compliance**: Ensures fair evaluation of model performance
- **Publication standards**: Required by most medical imaging journals

## Stratified Splitting

### Why Stratified?

Stratified K-Fold ensures that the class ratio (HGG:LGG) is preserved in each fold:

- **Original distribution**: HGG: ~73.7%, LGG: ~26.3%
- **Each fold**: Maintains similar ratio
- **Balanced evaluation**: Prevents bias due to imbalanced validation sets

### Class Distribution Example

For 5-fold split with 285 patients (210 HGG, 75 LGG):
- Each fold: ~57 patients (42 HGG, 15 LGG) in validation
- Each fold: ~228 patients (168 HGG, 60 LGG) in training
- Ratio preserved: ~73.7% HGG in both train and val

## Reproducibility

### Fixed Random Seed

The default seed (42) ensures:
- **Consistent splits**: Same seed produces same splits
- **Reproducible experiments**: Results can be replicated
- **Comparable baselines**: Different models use same data splits

### Using Different Seeds

```bash
# Generate splits with different seed
python scripts/splits/make_kfold_splits.py --seed 123 --output-json splits/kfold_5fold_seed123.json
```

## File Structure

```
/workspace/brain_tumor_project/
├── data/
│   ├── index/
│   │   └── stage4_index.csv          # Patient index (285 patients)
│   └── processed/
│       └── stage_4_resize/           # Stage 4 outputs (input to Stage 7)
├── splits/
│   ├── kfold_5fold_seed42.json       # Split summary (JSON)
│   ├── fold_0_train.csv              # Fold 0 training set
│   ├── fold_0_val.csv                # Fold 0 validation set
│   ├── fold_1_train.csv
│   ├── fold_1_val.csv
│   └── ... (folds 2-4)
└── scripts/
    └── splits/
        ├── build_stage4_index.py     # Index builder
        └── make_kfold_splits.py      # Split generator
```

## Important Notes

1. **No Data Movement**: Stage 7 does NOT move, copy, or modify any NIfTI files
2. **Metadata Only**: Only creates CSV/JSON files defining splits
3. **Relative Paths**: Paths in CSV files are relative to `data/processed/stage_4_resize/train/`
4. **Git-Friendly**: Split files are small and can be committed to version control
5. **Regenerable**: Splits can be regenerated from index file if needed

## Troubleshooting

### Index File Not Found

```bash
# First build the index
python scripts/splits/build_stage4_index.py
```

### scikit-learn Not Installed

```bash
pip install scikit-learn
```

### Verification Failures

If verification fails:
- Check that index file is correct
- Ensure no duplicate patient IDs
- Verify class labels are correct (0=LGG, 1=HGG)

## Next Steps

After Stage 7 completion:
- Use split files to create train/val datasets
- Implement cross-validation training loop
- Train models on each fold
- Aggregate results across folds
- Proceed with model training (MIL, ResNet50-3D, Swin UNETR)

## References

- scikit-learn StratifiedKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
- Cross-Validation in Medical Imaging: Best practices for ML evaluation
- BraTS2018 Dataset: https://www.med.upenn.edu/cbica/brats2018/
- Patient-Level Splitting: Critical for preventing data leakage in medical ML

## Academic Reproducibility

This split generation pipeline is designed for:
- Academic thesis work
- Medical AI research publications
- Public GitHub repositories

All parameters are configurable, splits are reproducible with fixed seeds, and the pipeline follows medical imaging ML best practices for cross-validation.

