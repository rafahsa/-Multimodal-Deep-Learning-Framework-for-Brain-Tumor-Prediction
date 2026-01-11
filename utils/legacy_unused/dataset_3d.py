"""
3D Volume Dataset for ResNet50-3D and Swin UNETR

This module provides PyTorch Dataset classes for loading 3D MRI volumes
from preprocessed Stage 4 data (128x128x128 volumes).

Author: Medical Imaging Pipeline
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
import SimpleITK as sitk

try:
    from monai.transforms import Compose
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("Warning: MONAI not installed. Transforms will be limited.")


class Volume3DDataset(Dataset):
    """
    Dataset for loading 3D MRI volumes for ResNet50-3D and Swin UNETR.
    
    This dataset loads full 3D volumes (128x128x128) from Stage 4 preprocessed data.
    No entropy-based slice selection is applied (unlike MIL datasets).
    
    Args:
        data_root: Root directory containing preprocessed data (stage_4_resize/train/)
        split_file: Path to CSV file with patient IDs and labels (fold_X_train.csv or fold_X_val.csv)
        modality: Modality to load ('t1', 't1ce', 't2', 'flair')
        transform: Optional transform pipeline (from augmentations_3d.py)
        class_to_idx: Dictionary mapping class names to indices (default: {'LGG': 0, 'HGG': 1})
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        split_file: Union[str, Path],
        modality: str = 'flair',
        transform: Optional[Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        self.data_root = Path(data_root)
        self.split_file = Path(split_file)
        self.modality = modality.lower()
        self.transform = transform
        
        if class_to_idx is None:
            self.class_to_idx = {'LGG': 0, 'HGG': 1}
        else:
            self.class_to_idx = class_to_idx
        
        # Load split file
        self.samples = self._load_split_file()
        
        print(f"Loaded {len(self.samples)} samples from {split_file}")
        print(f"Modality: {modality}")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _load_split_file(self) -> List[Tuple[Path, int, str]]:
        """
        Load patient IDs and labels from split CSV file.
        
        Expected CSV format:
            patient_id,class
            Brats18_TCIA10_103_1,LGG
            ...
        
        Returns:
            List of (volume_path, label, patient_id) tuples
        """
        import pandas as pd
        
        df = pd.read_csv(self.split_file)
        samples = []
        
        for _, row in df.iterrows():
            patient_id = row['patient_id']
            class_name = row['class']
            
            # Construct volume path
            # Expected structure: data_root/<class>/<patient_id>/<patient_id>_<modality>.nii.gz
            volume_path = self.data_root / class_name / patient_id / f"{patient_id}_{self.modality}.nii.gz"
            
            # Fallback to .nii if .nii.gz doesn't exist
            if not volume_path.exists():
                volume_path = self.data_root / class_name / patient_id / f"{patient_id}_{self.modality}.nii"
            
            if not volume_path.exists():
                print(f"Warning: Volume not found: {volume_path}")
                continue
            
            label = self.class_to_idx[class_name]
            samples.append((volume_path, label, patient_id))
        
        return samples
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution statistics."""
        distribution = {}
        for _, label, _ in self.samples:
            class_name = [k for k, v in self.class_to_idx.items() if v == label][0]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Load and return a 3D volume.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (volume_tensor, label, patient_id)
            - volume_tensor: Shape (1, D, H, W) = (1, 128, 128, 128)
            - label: Class label (0 for LGG, 1 for HGG)
            - patient_id: Patient identifier
        """
        volume_path, label, patient_id = self.samples[idx]
        
        # Load NIfTI volume
        volume = self._load_nifti(volume_path)
        
        # Apply transforms if provided
        if self.transform is not None:
            # MONAI transforms expect dict with 'image' key
            volume_dict = {'image': volume}
            volume_dict = self.transform(volume_dict)
            volume = volume_dict['image']
            
            # Convert MetaTensor to regular tensor if needed
            if hasattr(volume, 'as_tensor'):
                volume = volume.as_tensor()
            elif hasattr(volume, 'numpy'):
                # If it's still a numpy array somehow, convert to tensor
                volume = torch.from_numpy(volume).float()
            
            # Ensure channel dimension is present (should be added by EnsureChannelFirstD)
            if volume.ndim == 3:
                volume = volume.unsqueeze(0)  # (1, D, H, W)
        else:
            # Convert to tensor if no transforms
            if isinstance(volume, np.ndarray):
                volume = torch.from_numpy(volume).float()
                # Add channel dimension if missing
                if volume.ndim == 3:
                    volume = volume.unsqueeze(0)  # (1, D, H, W)
        
        return volume, label, patient_id
    
    def _load_nifti(self, path: Path) -> np.ndarray:
        """
        Load NIfTI volume using SimpleITK.
        
        Args:
            path: Path to NIfTI file
        
        Returns:
            Volume as numpy array (D, H, W)
        """
        try:
            image = sitk.ReadImage(str(path))
            volume = sitk.GetArrayFromImage(image)
            return volume.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Error loading NIfTI file {path}: {e}")
    
    def get_all_labels(self) -> List[int]:
        """Get all labels for class balancing."""
        return [label for _, label, _ in self.samples]


if __name__ == "__main__":
    """
    Test Volume3DDataset.
    """
    print("Testing Volume3DDataset")
    print("=" * 60)
    
    # Example usage (adjust paths as needed)
    data_root = Path("data/processed/stage_4_resize/train")
    split_file = Path("splits/fold_0_train.csv")
    
    if not split_file.exists():
        print(f"Warning: Split file not found: {split_file}")
        print("Skipping dataset test.")
        exit(0)
    
    # Create dataset
    dataset = Volume3DDataset(
        data_root=data_root,
        split_file=split_file,
        modality='flair',
        transform=None
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test loading a sample
    if len(dataset) > 0:
        volume, label, patient_id = dataset[0]
        print(f"Sample 0:")
        print(f"  Patient ID: {patient_id}")
        print(f"  Label: {label}")
        print(f"  Volume shape: {volume.shape}")
        print(f"  Volume dtype: {volume.dtype}")
        print(f"  Volume range: [{volume.min():.2f}, {volume.max():.2f}]")
    
    print("\n" + "=" * 60)
    print("Volume3DDataset test passed!")

