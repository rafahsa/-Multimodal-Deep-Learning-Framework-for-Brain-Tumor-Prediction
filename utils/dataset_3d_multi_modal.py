"""
Multi-Modality 3D Volume Dataset for ResNet50-3D

This module provides a PyTorch Dataset class for loading multi-modality 3D MRI volumes
(T1, T1ce, T2, FLAIR) with early fusion (stacked as channels).

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


class MultiModalVolume3DDataset(Dataset):
    """
    Dataset for loading multi-modality 3D MRI volumes with early fusion.
    
    Loads all 4 modalities (T1, T1ce, T2, FLAIR) for each patient and stacks them
    as channels, resulting in input shape (4, D, H, W).
    
    All modalities are loaded from the same patient directory and undergo the same
    spatial augmentations to maintain spatial correspondence.
    
    Args:
        data_root: Root directory containing preprocessed data (stage_4_resize/train/)
        split_file: Path to CSV file with patient IDs and labels (fold_X_train.csv or fold_X_val.csv)
        modalities: List of modalities to load (default: ['t1', 't1ce', 't2', 'flair'])
        transform: Optional transform pipeline (from augmentations_3d.py)
                  Applied to all modalities consistently
        class_to_idx: Dictionary mapping class names to indices (default: {'LGG': 0, 'HGG': 1})
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        split_file: Union[str, Path],
        modalities: List[str] = ['t1', 't1ce', 't2', 'flair'],
        transform: Optional[Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None
    ):
        self.data_root = Path(data_root)
        self.split_file = Path(split_file)
        self.modalities = [m.lower() for m in modalities]
        self.transform = transform
        
        if class_to_idx is None:
            self.class_to_idx = {'LGG': 0, 'HGG': 1}
        else:
            self.class_to_idx = class_to_idx
        
        # Validate modalities
        valid_modalities = ['t1', 't1ce', 't2', 'flair']
        for mod in self.modalities:
            if mod not in valid_modalities:
                raise ValueError(f"Invalid modality: {mod}. Must be one of {valid_modalities}")
        
        if len(self.modalities) != 4:
            raise ValueError(f"Expected 4 modalities, got {len(self.modalities)}")
        
        # Load split file
        self.samples = self._load_split_file()
        
        print(f"Loaded {len(self.samples)} samples from {split_file}")
        print(f"Modalities: {', '.join(self.modalities)}")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _load_split_file(self) -> List[Tuple[List[Path], int, str]]:
        """
        Load patient IDs and labels from split CSV file.
        
        Expected CSV format:
            patient_id,class
            Brats18_TCIA10_103_1,LGG
            ...
        
        Returns:
            List of (modality_paths, label, patient_id) tuples
            modality_paths: List of paths in order [t1, t1ce, t2, flair]
        """
        import pandas as pd
        
        df = pd.read_csv(self.split_file)
        samples = []
        
        for _, row in df.iterrows():
            patient_id = row['patient_id']
            class_name = row['class']
            
            # Construct paths for all modalities
            modality_paths = []
            missing_modalities = []
            
            for modality in self.modalities:
                # Expected structure: data_root/<class>/<patient_id>/<patient_id>_<modality>.nii.gz
                volume_path = self.data_root / class_name / patient_id / f"{patient_id}_{modality}.nii.gz"
                
                # Fallback to .nii if .nii.gz doesn't exist
                if not volume_path.exists():
                    volume_path = self.data_root / class_name / patient_id / f"{patient_id}_{modality}.nii"
                
                if not volume_path.exists():
                    missing_modalities.append(modality)
                    continue
                
                modality_paths.append(volume_path)
            
            # Skip if any modality is missing
            if missing_modalities:
                print(f"Warning: Missing modalities {missing_modalities} for patient {patient_id}. Skipping.")
                continue
            
            if len(modality_paths) != len(self.modalities):
                print(f"Warning: Expected {len(self.modalities)} modalities for {patient_id}, got {len(modality_paths)}. Skipping.")
                continue
            
            label = self.class_to_idx[class_name]
            samples.append((modality_paths, label, patient_id))
        
        return samples
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution in the dataset."""
        distribution = {}
        for _, label, _ in self.samples:
            class_name = [k for k, v in self.class_to_idx.items() if v == label][0]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution
    
    def get_all_labels(self) -> List[int]:
        """Get all labels for class balancing."""
        return [label for _, label, _ in self.samples]
    
    def _load_nifti(self, volume_path: Path) -> np.ndarray:
        """
        Load NIfTI volume and return as numpy array.
        
        Args:
            volume_path: Path to NIfTI file
            
        Returns:
            3D numpy array of shape (D, H, W)
        """
        try:
            sitk_image = sitk.ReadImage(str(volume_path))
            volume = sitk.GetArrayFromImage(sitk_image)
            return volume.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Error loading {volume_path}: {e}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get multi-modality volume for a single patient.
        
        Returns:
            volume: Stacked multi-modality volume of shape (4, D, H, W)
            label: Class label (0 for LGG, 1 for HGG)
            patient_id: Patient identifier
        """
        modality_paths, label, patient_id = self.samples[idx]
        
        # Load all modalities
        modalities_data = []
        for modality_path in modality_paths:
            volume = self._load_nifti(modality_path)  # Shape: (D, H, W)
            # Verify each modality is 3D
            assert volume.ndim == 3, f"Expected 3D volume (D, H, W), got {volume.ndim}D with shape {volume.shape} for {modality_path}"
            modalities_data.append(volume)
        
        # Stack modalities as channels: (4, D, H, W)
        # Each modality is (D, H, W), stack along channel dimension (axis=0)
        multi_modal_volume = np.stack(modalities_data, axis=0)  # Shape: (4, D, H, W)
        
        # AUTHORITATIVE SHAPE CHECK: Verify shape after stacking
        assert multi_modal_volume.shape[0] == 4, f"After stacking: Expected 4 channels at dim 0, got {multi_modal_volume.shape[0]}. Full shape: {multi_modal_volume.shape}"
        assert multi_modal_volume.ndim == 4, f"After stacking: Expected 4D array (4, D, H, W), got {multi_modal_volume.ndim}D with shape {multi_modal_volume.shape}"
        D, H, W = multi_modal_volume.shape[1], multi_modal_volume.shape[2], multi_modal_volume.shape[3]
        
        # Apply transforms if provided
        if self.transform is not None:
            # MONAI transforms expect dictionary with 'image' key
            # Input shape: (4, D, H, W) - channels already first
            volume_dict = {'image': multi_modal_volume}
            volume_dict = self.transform(volume_dict)
            multi_modal_volume = volume_dict['image']
            
            # Convert MetaTensor to regular tensor if needed
            if hasattr(multi_modal_volume, 'as_tensor'):
                multi_modal_volume = multi_modal_volume.as_tensor()
            elif hasattr(multi_modal_volume, 'numpy'):
                multi_modal_volume = torch.from_numpy(multi_modal_volume.numpy()).float()
            
            # Ensure it's a torch tensor
            if not isinstance(multi_modal_volume, torch.Tensor):
                multi_modal_volume = torch.from_numpy(multi_modal_volume).float()
        else:
            # Convert to tensor if no transforms
            if isinstance(multi_modal_volume, np.ndarray):
                multi_modal_volume = torch.from_numpy(multi_modal_volume).float()
        
        # FINAL AUTHORITATIVE SHAPE CHECK: Must be (4, D, H, W)
        assert multi_modal_volume.ndim == 4, f"Final volume must be 4D (4, D, H, W), got {multi_modal_volume.ndim}D with shape {multi_modal_volume.shape}"
        assert multi_modal_volume.shape[0] == 4, f"Final volume must have 4 channels at dim 0, got {multi_modal_volume.shape[0]}. Full shape: {multi_modal_volume.shape}. This indicates a transform incorrectly changed channel dimension."
        
        return multi_modal_volume, label, patient_id

