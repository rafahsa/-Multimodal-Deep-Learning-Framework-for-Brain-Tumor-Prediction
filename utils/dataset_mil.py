"""
Multiple Instance Learning (MIL) Dataset for Brain Tumor Classification

This module provides a PyTorch Dataset class for MIL-based classification,
where each patient (bag) contains multiple 2D slices (instances).

Each bag contains N slices extracted from the 3D multi-modal volume.
The dataset supports fixed bag sizes with sampling or padding strategies.

Author: Medical Imaging Pipeline
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random

import torch
from torch.utils.data import Dataset
import SimpleITK as sitk

try:
    from monai.transforms import Compose
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("Warning: MONAI not installed. Transforms will be limited.")


class MILSliceDataset(Dataset):
    """
    Dataset for MIL-based classification using 2D slices as instances.
    
    Each patient (bag) contains multiple 2D slices (instances) extracted from
    the 3D multi-modal volume. The dataset supports fixed bag sizes with
    sampling or padding strategies.
    
    Args:
        data_root: Root directory containing preprocessed data (stage_4_resize/train/)
        split_file: Path to CSV file with patient IDs and labels
        modalities: List of modalities to load (default: ['t1', 't1ce', 't2', 'flair'])
        bag_size: Fixed number of slices per bag (default: 64)
        sampling_strategy: 'random', 'sequential', or 'entropy' (default: 'random')
        transform: Optional transform pipeline (applied per slice)
        class_to_idx: Dictionary mapping class names to indices
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        split_file: Union[str, Path],
        modalities: List[str] = ['t1', 't1ce', 't2', 'flair'],
        bag_size: int = 64,
        sampling_strategy: str = 'random',
        transform: Optional[Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        seed: int = 42
    ):
        self.data_root = Path(data_root)
        self.split_file = Path(split_file)
        self.modalities = [m.lower() for m in modalities]
        self.bag_size = bag_size
        self.sampling_strategy = sampling_strategy
        self.transform = transform
        self.seed = seed
        
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
        
        print(f"Loaded {len(self.samples)} patients (bags) from {split_file}")
        print(f"Modalities: {', '.join(self.modalities)}")
        print(f"Bag size: {bag_size}, Sampling: {sampling_strategy}")
        print(f"Class distribution: {self._get_class_distribution()}")
    
    def _load_split_file(self) -> List[Tuple[str, int, str]]:
        """
        Load patient IDs and labels from split CSV file.
        
        Returns:
            List of (patient_id, label, class_name) tuples
            class_name is needed to construct correct file paths
        """
        import pandas as pd
        
        df = pd.read_csv(self.split_file)
        samples = []
        
        for _, row in df.iterrows():
            patient_id = row['patient_id']
            class_name = row['class']
            
            if class_name not in self.class_to_idx:
                continue
            
            label = self.class_to_idx[class_name]
            samples.append((patient_id, label, class_name))
        
        return samples
    
    def _get_class_distribution(self) -> Dict[str, int]:
        """Get class distribution in the dataset."""
        distribution = {}
        for _, label, class_name in self.samples:
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution
    
    def _load_volume(self, patient_id: str, class_name: str) -> np.ndarray:
        """
        Load 3D multi-modal volume for a patient.
        
        Args:
            patient_id: Patient identifier (e.g., 'Brats18_2013_11_1')
            class_name: Class name ('HGG' or 'LGG') - needed for correct path construction
        
        Returns:
            volume: (4, D, H, W) numpy array where channels are [T1, T1ce, T2, FLAIR]
        """
        # Construct path following the same pattern as MultiModalVolume3DDataset:
        # data_root/<class_name>/<patient_id>/<patient_id>_<modality>.nii.gz
        patient_dir = self.data_root / class_name / patient_id
        
        volume_channels = []
        missing_modalities = []
        
        for mod in self.modalities:
            # Expected filename pattern: <patient_id>_<modality>.nii.gz
            # Example: Brats18_2013_11_1_t1.nii.gz
            volume_path = patient_dir / f"{patient_id}_{mod}.nii.gz"
            
            # Fallback to .nii if .nii.gz doesn't exist
            if not volume_path.exists():
                volume_path = patient_dir / f"{patient_id}_{mod}.nii"
            
            if not volume_path.exists():
                missing_modalities.append(mod)
                continue
            
            # Load volume
            try:
                volume = sitk.ReadImage(str(volume_path))
                volume_array = sitk.GetArrayFromImage(volume)  # (D, H, W)
                volume_channels.append(volume_array)
            except Exception as e:
                raise RuntimeError(f"Error loading {volume_path}: {e}")
        
        # Check if all modalities were found
        if missing_modalities:
            raise FileNotFoundError(
                f"Modalities {missing_modalities} not found for patient {patient_id} "
                f"in directory {patient_dir}. Expected pattern: {patient_id}_<modality>.nii.gz"
            )
        
        if len(volume_channels) != len(self.modalities):
            raise RuntimeError(
                f"Expected {len(self.modalities)} modalities for {patient_id}, "
                f"but only loaded {len(volume_channels)}"
            )
        
        # Stack modalities: (4, D, H, W)
        multi_modal_volume = np.stack(volume_channels, axis=0)
        
        return multi_modal_volume
    
    def _extract_slices(self, volume: np.ndarray) -> np.ndarray:
        """
        Extract 2D slices from 3D volume.
        
        Args:
            volume: (4, D, H, W) multi-modal volume
        
        Returns:
            slices: (N, 4, H, W) where N is number of slices (D dimension)
        """
        # volume shape: (4, D, H, W)
        # Extract slices along depth dimension (axis=1)
        # Transpose to (D, 4, H, W) then reshape
        D = volume.shape[1]
        slices = np.transpose(volume, (1, 0, 2, 3))  # (D, 4, H, W)
        return slices
    
    def _sample_slices(self, slices: np.ndarray) -> np.ndarray:
        """
        Sample or pad slices to fixed bag size.
        
        Args:
            slices: (N, 4, H, W) where N is variable
        
        Returns:
            bag: (bag_size, 4, H, W) fixed-size bag
        """
        N = slices.shape[0]
        
        if N >= self.bag_size:
            # Sample slices
            if self.sampling_strategy == 'random':
                # Random sampling
                np.random.seed(self.seed)
                indices = np.random.choice(N, size=self.bag_size, replace=False)
                indices = sorted(indices)  # Keep spatial order
                bag = slices[indices]
            elif self.sampling_strategy == 'sequential':
                # Sequential sampling (evenly spaced)
                step = N / self.bag_size
                indices = [int(i * step) for i in range(self.bag_size)]
                bag = slices[indices]
            elif self.sampling_strategy == 'entropy':
                # Entropy-based sampling (select most informative slices)
                # Compute entropy for each slice
                entropies = []
                for i in range(N):
                    slice_2d = slices[i]  # (4, H, W)
                    # Compute entropy across all modalities
                    entropy = -np.sum(slice_2d * np.log(slice_2d + 1e-10))
                    entropies.append(entropy)
                entropies = np.array(entropies)
                # Select top bag_size slices by entropy
                top_indices = np.argsort(entropies)[-self.bag_size:]
                top_indices = sorted(top_indices)
                bag = slices[top_indices]
            else:
                raise ValueError(f"Unsupported sampling_strategy: {self.sampling_strategy}")
        else:
            # Pad with zeros
            pad_size = self.bag_size - N
            pad_shape = (pad_size,) + slices.shape[1:]
            pad = np.zeros(pad_shape, dtype=slices.dtype)
            bag = np.concatenate([slices, pad], axis=0)
        
        return bag
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a bag (patient) with multiple slices (instances).
        
        Returns:
            bag_of_slices: (bag_size, 4, H, W) tensor
            label: Patient-level label (0 or 1)
            patient_id: Patient identifier
        """
        patient_id, label, class_name = self.samples[idx]
        
        # Load 3D volume (requires class_name for correct path construction)
        volume = self._load_volume(patient_id, class_name)  # (4, D, H, W)
        
        # Extract slices
        slices = self._extract_slices(volume)  # (D, 4, H, W)
        
        # Sample/pad to fixed bag size
        bag = self._sample_slices(slices)  # (bag_size, 4, H, W)
        
        # Convert to tensor first
        bag = torch.from_numpy(bag).float()
        
        # Apply transforms (per slice) if provided
        if self.transform is not None:
            # Apply transform to each slice
            transformed_bag = []
            for i in range(self.bag_size):
                slice_2d = bag[i]  # (4, H, W)
                # Apply transform (expects (C, H, W))
                transformed_slice = self.transform(slice_2d)
                transformed_bag.append(transformed_slice)
            bag = torch.stack(transformed_bag, dim=0)  # (bag_size, 4, H, W)
        
        return bag, label, patient_id


def get_all_labels(dataset: MILSliceDataset) -> List[int]:
    """
    Get all labels from dataset (for WeightedRandomSampler).
    
    Args:
        dataset: MILSliceDataset instance
    
    Returns:
        List of integer labels
    """
    return [label for _, label, _ in dataset]


if __name__ == "__main__":
    # Test dataset
    data_root = "data/processed/stage_4_resize/train"
    split_file = "splits/fold_0_train.csv"
    
    dataset = MILSliceDataset(
        data_root=data_root,
        split_file=split_file,
        bag_size=64,
        sampling_strategy='random'
    )
    
    # Get one sample
    bag, label, patient_id = dataset[0]
    print(f"Bag shape: {bag.shape}")
    print(f"Label: {label}, Patient ID: {patient_id}")

