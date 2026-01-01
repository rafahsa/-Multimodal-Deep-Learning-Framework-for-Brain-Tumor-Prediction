"""
MIL Dataset for BraTS2018 Brain Tumor Classification

This module provides a PyTorch Dataset class for Multiple Instance Learning (MIL)
on 3D MRI volumes. It supports entropy-based slice selection for MIL models.

IMPORTANT: This dataset is designed EXCLUSIVELY for MIL models.
- ResNet50-3D and Swin UNETR should NOT use this dataset
- Entropy-based slice selection is MIL-specific by design
- 3D CNNs process full volumes, not slices

Author: Medical Imaging Pipeline
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class MILDataset(Dataset):
    """
    Multiple Instance Learning Dataset for BraTS2018 3D MRI volumes.
    
    This dataset extracts 2D slices from 3D volumes to create "bags" of instances
    for MIL training. It optionally uses entropy-based slice selection to focus
    on informative slices.
    
    IMPORTANT DESIGN NOTES:
    - This dataset is MIL-ONLY. Do not use for ResNet50-3D or Swin UNETR.
    - Entropy-based selection is optional (use_entropy=False uses all slices)
    - Entropy metadata must be pre-computed using run_entropy_analysis.py
    - Slice selection is deterministic (same slices selected each time)
    
    Args:
        split_csv: Path to CSV file with patient data (columns: patient_id, class, path_*)
        data_root: Root directory containing processed data (e.g., data/processed/stage_4_resize/train)
        modality: Modality to use ('t1', 't1ce', 't2', 'flair', default: 'flair')
        use_entropy: If True, use entropy-based slice selection (default: False)
        entropy_dir: Directory containing entropy JSON files (default: 'data/entropy')
        transform: Optional transform to apply to slices
    """
    
    def __init__(
        self,
        split_csv: str,
        data_root: str,
        modality: str = 'flair',
        use_entropy: bool = False,
        entropy_dir: str = 'data/entropy',
        transform: Optional[object] = None
    ):
        self.data_root = Path(data_root)
        self.modality = modality
        self.use_entropy = use_entropy
        self.entropy_dir = Path(entropy_dir)
        self.transform = transform
        
        # Load patient data from CSV
        import csv
        self.patients = []
        with open(split_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Store path for the selected modality
                path_key = f'path_{modality}'
                if path_key not in row:
                    raise ValueError(f"CSV missing required column: {path_key}")
                
                self.patients.append({
                    'patient_id': row['patient_id'],
                    'class': row['class'],
                    'class_label': int(row['class_label']),  # HGG=1, LGG=0
                    'path': row[path_key]  # Relative path to modality file
                })
        
        # If using entropy, verify entropy files exist
        if self.use_entropy:
            missing_entropy = []
            for patient in self.patients:
                entropy_file = self.entropy_dir / f"{patient['patient_id']}_entropy.json"
                if not entropy_file.exists():
                    missing_entropy.append(patient['patient_id'])
            
            if missing_entropy:
                raise FileNotFoundError(
                    f"Entropy files not found for {len(missing_entropy)} patients. "
                    f"Run: python scripts/analysis/run_entropy_analysis.py\n"
                    f"Missing: {missing_entropy[:5]}..."
                )
    
    def __len__(self) -> int:
        return len(self.patients)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a patient's slice bag and label.
        
        Returns:
            (slice_bag, label) where:
            - slice_bag: Tensor of shape (num_slices, H, W) - bag of slices
            - label: Tensor of shape (1,) - class label (HGG=1, LGG=0)
        """
        patient = self.patients[idx]
        patient_id = patient['patient_id']
        label = torch.tensor(patient['class_label'], dtype=torch.long)
        
        # Load volume using path from CSV
        # Path from CSV is relative to data_root (e.g., "HGG/patient_id/patient_id_flair.nii.gz")
        volume_path = self.data_root / patient['path']
        
        # Try .nii.gz first, then .nii if needed
        if not volume_path.exists():
            volume_path = volume_path.with_suffix('.nii')
        
        if not volume_path.exists():
            raise FileNotFoundError(
                f"Volume not found: {volume_path} (patient {patient_id}, modality {self.modality})"
            )
        
        # Read volume
        image = sitk.ReadImage(str(volume_path))
        volume_array = sitk.GetArrayFromImage(image)  # Shape: (D, H, W) for axial slices
        
        # Select slices based on entropy or use all slices
        if self.use_entropy:
            # ENTROPY-BASED SLICE SELECTION (MIL-ONLY)
            # Load entropy metadata
            # Naming convention: <patient_id>_entropy.json
            # This matches the output from run_entropy_analysis.py and run_entropy_for_fold.py
            entropy_file = self.entropy_dir / f"{patient_id}_entropy.json"
            with open(entropy_file, 'r') as f:
                entropy_data = json.load(f)
            
            # Get top-k slice indices
            top_k_indices = entropy_data['top_k_slices']
            
            # Verify axis matches (should be 'axial' for standard MIL)
            axis = entropy_data.get('axis', 'axial')
            if axis != 'axial':
                # For non-axial, we'd need different slice extraction
                # For now, we assume axial (most common for MIL)
                pass
            
            # Extract selected slices
            # Note: top_k_indices are already sorted (ascending) from entropy analysis
            slice_indices = sorted(top_k_indices)  # Ensure sorted for deterministic behavior
            slices = [volume_array[i, :, :] for i in slice_indices]
            
        else:
            # USE ALL SLICES (original behavior, no entropy selection)
            num_slices = volume_array.shape[0]
            slices = [volume_array[i, :, :] for i in range(num_slices)]
        
        # Stack slices into bag
        slice_bag = np.stack(slices, axis=0)  # Shape: (num_slices, H, W)
        
        # Convert to tensor
        slice_bag = torch.from_numpy(slice_bag).float()
        
        # Apply transforms if provided
        if self.transform is not None:
            slice_bag = self.transform(slice_bag)
        
        # Add channel dimension if needed (some models expect (C, H, W) per slice)
        # For MIL, typically we keep as (num_slices, H, W) or (num_slices, 1, H, W)
        if slice_bag.dim() == 3:
            # Add channel dimension: (num_slices, H, W) -> (num_slices, 1, H, W)
            slice_bag = slice_bag.unsqueeze(1)
        
        return slice_bag, label
    
    def get_patient_id(self, idx: int) -> str:
        """Get patient ID for a given index."""
        return self.patients[idx]['patient_id']
    
    def get_class_label(self, idx: int) -> int:
        """Get class label for a given index."""
        return self.patients[idx]['class_label']


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of MILDataset with and without entropy.
    """
    print("MIL Dataset Example")
    print("=" * 60)
    
    # Example: Without entropy (use all slices)
    print("\n1. Dataset WITHOUT entropy (all slices):")
    print("   dataset = MILDataset(")
    print("       split_csv='splits/fold_1_train.csv',")
    print("       data_root='data/processed/stage_4_resize/train',")
    print("       modality='flair',")
    print("       use_entropy=False")
    print("   )")
    
    # Example: With entropy (top-k slices only)
    print("\n2. Dataset WITH entropy (top-k slices):")
    print("   dataset = MILDataset(")
    print("       split_csv='splits/fold_1_train.csv',")
    print("       data_root='data/processed/stage_4_resize/train',")
    print("       modality='flair',")
    print("       use_entropy=True,")
    print("       entropy_dir='data/entropy'")
    print("   )")
    
    print("\n" + "=" * 60)
    print("IMPORTANT NOTES:")
    print("  - This dataset is MIL-ONLY (not for ResNet50-3D or Swin UNETR)")
    print("  - Entropy selection requires pre-computed entropy JSON files")
    print("  - Slice selection is deterministic (patient-level consistency)")
    print("  - Entropy must match the modality used for analysis")
    print("=" * 60)

