"""
MIL Dataset with ROI-based Sampling Support

This is a minimal modification of MILSliceDataset to support ROI-based
instance sampling using segmentation masks.

Key changes:
- Loads segmentation mask if path_seg is available in split CSV
- Adds 'roi' sampling strategy: 70% from tumor region, 30% from context
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random

import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from scipy import ndimage

# Import base class
from utils.dataset_mil import MILSliceDataset

try:
    from monai.transforms import Compose
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False


class MILSliceDatasetROI(MILSliceDataset):
    """
    MIL Dataset with ROI-based sampling support.
    
    Extends MILSliceDataset to support segmentation-guided sampling:
    - 70% of instances from tumor region (seg > 0)
    - 30% from near-tumor context or whole brain
    
    Args:
        Same as MILSliceDataset, plus:
        seg_data_root: Root directory for segmentation masks (default: data/raw/BraTS2018)
        roi_tumor_ratio: Fraction of instances from tumor region (default: 0.7)
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
        seed: int = 42,
        seg_data_root: Optional[Union[str, Path]] = None,
        roi_tumor_ratio: float = 0.7
    ):
        # Initialize parent class
        super().__init__(
            data_root=data_root,
            split_file=split_file,
            modalities=modalities,
            bag_size=bag_size,
            sampling_strategy=sampling_strategy,
            transform=transform,
            class_to_idx=class_to_idx,
            seed=seed
        )
        
        # ROI-specific settings
        # Default seg_data_root: try to find data/raw/BraTS2018 relative to project root
        if seg_data_root:
            self.seg_data_root = Path(seg_data_root)
        else:
            # Try multiple possible locations
            data_root_path = Path(data_root)
            # Option 1: data/raw/BraTS2018 (most common structure)
            candidate1 = data_root_path.parent.parent.parent / 'raw' / 'BraTS2018'
            # Option 2: data_root/../raw/BraTS2018 (if data_root is data/processed/...)
            candidate2 = data_root_path.parent.parent / 'raw' / 'BraTS2018'
            # Option 3: data_root/../../raw/BraTS2018 (if data_root is deeper)
            candidate3 = data_root_path.parent.parent.parent.parent / 'raw' / 'BraTS2018'
            
            # Use first candidate that exists, or fall back to candidate1
            if candidate1.exists():
                self.seg_data_root = candidate1
            elif candidate2.exists():
                self.seg_data_root = candidate2
            elif candidate3.exists():
                self.seg_data_root = candidate3
            else:
                # Fallback to most likely location
                self.seg_data_root = candidate1
        
        self.roi_tumor_ratio = roi_tumor_ratio
        
        # Load segmentation paths from split file
        self.seg_paths = self._load_seg_paths()
    
    def _load_seg_paths(self) -> Dict[str, Optional[str]]:
        """Load segmentation paths from split CSV if available."""
        import pandas as pd
        
        seg_paths = {}
        try:
            df = pd.read_csv(self.split_file)
            if 'path_seg' in df.columns:
                for _, row in df.iterrows():
                    patient_id = row['patient_id']
                    seg_path = row.get('path_seg', None)
                    seg_paths[patient_id] = str(seg_path) if pd.notna(seg_path) else None
        except Exception as e:
            print(f"Warning: Could not load segmentation paths: {e}")
        
        return seg_paths
    
    def _load_segmentation_mask(self, patient_id: str, class_name: str) -> Optional[np.ndarray]:
        """
        Load segmentation mask for a patient.
        
        Uses path_seg from CSV as the single source of truth.
        Handles both absolute and relative paths correctly.
        
        Returns:
            seg_mask: (D, H, W) boolean array where True indicates tumor region
            Returns None if segmentation file not found
        """
        # Get path_seg from CSV (single source of truth)
        seg_path = self.seg_paths.get(patient_id)
        
        if seg_path is None:
            return None
        
        # Resolve path: use path_seg exactly as provided
        seg_path_obj = Path(seg_path)
        
        # If absolute path, use as-is; otherwise resolve relative to seg_data_root
        if seg_path_obj.is_absolute():
            full_path = seg_path_obj
        else:
            full_path = self.seg_data_root / seg_path
        
        # Check if the path exists as-is
        if full_path.exists():
            try:
                seg_image = sitk.ReadImage(str(full_path))
                seg_array = sitk.GetArrayFromImage(seg_image)  # (D, H, W)
                
                # Create tumor mask: seg > 0 (any tumor label)
                tumor_mask = seg_array > 0
                
                return tumor_mask
            except Exception as e:
                print(f"Warning: Could not load segmentation for {patient_id} from {full_path}: {e}")
                return None
        
        # If not found, try alternative extension (.nii <-> .nii.gz)
        if full_path.suffix == '.gz' and full_path.suffixes[-2:] == ['.nii', '.gz']:
            # Current is .nii.gz, try .nii
            alt_path = full_path.parent / full_path.stem  # Remove .gz
        elif full_path.suffix == '.nii':
            # Current is .nii, try .nii.gz
            alt_path = full_path.parent / (full_path.name + '.gz')
        else:
            # Unknown extension, don't try alternative
            alt_path = None
        
        if alt_path is not None and alt_path.exists():
            try:
                seg_image = sitk.ReadImage(str(alt_path))
                seg_array = sitk.GetArrayFromImage(seg_image)  # (D, H, W)
                
                # Create tumor mask: seg > 0 (any tumor label)
                tumor_mask = seg_array > 0
                
                return tumor_mask
            except Exception as e:
                print(f"Warning: Could not load segmentation for {patient_id} from {alt_path}: {e}")
                return None
        
        # Path truly does not exist
        return None
    
    def _get_roi_indices(self, tumor_mask: np.ndarray, num_slices: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get indices for ROI-based sampling.
        
        Args:
            tumor_mask: (D, H, W) boolean array
            num_slices: Total number of slices (D dimension)
        
        Returns:
            tumor_indices: Array of slice indices with tumor
            context_indices: Array of slice indices for context (near-tumor or whole brain)
        """
        # Find slices with tumor (any slice where tumor_mask has True values)
        tumor_slices = []
        context_slices = []
        
        for d in range(num_slices):
            slice_mask = tumor_mask[d, :, :]
            has_tumor = np.any(slice_mask)
            
            if has_tumor:
                tumor_slices.append(d)
            else:
                context_slices.append(d)
        
        tumor_indices = np.array(tumor_slices, dtype=np.int64)
        context_indices = np.array(context_slices, dtype=np.int64)
        
        # If we have very few tumor slices, try to create a context ring
        if len(tumor_indices) < num_slices * 0.1:  # Less than 10% tumor slices
            # Create dilated mask for context ring
            try:
                dilated_mask = ndimage.binary_dilation(tumor_mask, structure=np.ones((3, 5, 5)), iterations=2)
                ring_mask = dilated_mask & (~tumor_mask)
                
                # Find ring slices
                ring_slices = []
                for d in range(num_slices):
                    if np.any(ring_mask[d, :, :]):
                        ring_slices.append(d)
                
                if len(ring_slices) > 0:
                    context_indices = np.array(ring_slices, dtype=np.int64)
            except:
                # If dilation fails, just use all non-tumor slices
                pass
        
        return tumor_indices, context_indices
    
    def _sample_slices_roi(self, slices: np.ndarray, patient_id: str, class_name: str) -> np.ndarray:
        """
        Sample slices using ROI-based strategy.
        
        70% from tumor region, 30% from context.
        """
        N = slices.shape[0]
        
        # Load segmentation mask
        tumor_mask = self._load_segmentation_mask(patient_id, class_name)
        
        if tumor_mask is None:
            # Fallback to random sampling if segmentation not available
            print(f"Warning: Segmentation not available for {patient_id}, using random sampling")
            np.random.seed(self.seed)
            indices = np.random.choice(N, size=min(self.bag_size, N), replace=False)
            indices = sorted(indices)
            return slices[indices]
        
        # Get ROI indices
        tumor_indices, context_indices = self._get_roi_indices(tumor_mask, N)
        
        # Calculate number of slices from each region
        n_tumor = int(self.bag_size * self.roi_tumor_ratio)
        n_context = self.bag_size - n_tumor
        
        # Sample from tumor region
        np.random.seed(self.seed)
        if len(tumor_indices) > 0:
            if len(tumor_indices) >= n_tumor:
                tumor_selected = np.random.choice(tumor_indices, size=n_tumor, replace=False)
            else:
                # Not enough tumor slices, use all and pad with context
                tumor_selected = tumor_indices
                n_context = self.bag_size - len(tumor_selected)
        else:
            # No tumor slices found, use all context
            tumor_selected = np.array([], dtype=np.int64)
            n_context = self.bag_size
        
        # Sample from context region
        if len(context_indices) > 0 and n_context > 0:
            if len(context_indices) >= n_context:
                context_selected = np.random.choice(context_indices, size=n_context, replace=False)
            else:
                # Not enough context slices, use all
                context_selected = context_indices
        else:
            context_selected = np.array([], dtype=np.int64)
        
        # Combine and sort
        all_selected = np.concatenate([tumor_selected, context_selected])
        if len(all_selected) < self.bag_size:
            # Pad with random slices if needed
            remaining = self.bag_size - len(all_selected)
            all_indices = np.arange(N)
            available = np.setdiff1d(all_indices, all_selected)
            if len(available) > 0:
                pad_selected = np.random.choice(available, size=min(remaining, len(available)), replace=False)
                all_selected = np.concatenate([all_selected, pad_selected])
        
        # Ensure we have exactly bag_size slices
        if len(all_selected) > self.bag_size:
            all_selected = np.random.choice(all_selected, size=self.bag_size, replace=False)
        
        all_selected = sorted(all_selected)
        bag = slices[all_selected]
        
        return bag
    
    def _sample_slices(self, slices: np.ndarray) -> np.ndarray:
        """
        Override parent method to add ROI sampling support.
        """
        if self.sampling_strategy == 'roi':
            # Get patient info from current sample
            # We need to pass patient_id and class_name, but they're not available here
            # So we'll need to modify __getitem__ to call this differently
            # For now, use a workaround: store in instance variable during __getitem__
            if hasattr(self, '_current_patient_id') and hasattr(self, '_current_class_name'):
                return self._sample_slices_roi(
                    slices, 
                    self._current_patient_id, 
                    self._current_class_name
                )
            else:
                # Fallback to random
                np.random.seed(self.seed)
                N = slices.shape[0]
                indices = np.random.choice(N, size=min(self.bag_size, N), replace=False)
                indices = sorted(indices)
                return slices[indices]
        else:
            # Use parent implementation
            return super()._sample_slices(slices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Override to set patient info for ROI sampling.
        """
        patient_id, label, class_name = self.samples[idx]
        
        # Load volume
        volume = self._load_volume(patient_id, class_name)
        
        # Extract slices
        slices = self._extract_slices(volume)
        
        # Sample slices (with ROI support if strategy is 'roi')
        if self.sampling_strategy == 'roi':
            bag = self._sample_slices_roi(slices, patient_id, class_name)
        else:
            bag = self._sample_slices(slices)
        
        # Pad if needed
        N = bag.shape[0]
        if N < self.bag_size:
            # Pad with last slice
            padding = np.repeat(bag[-1:], self.bag_size - N, axis=0)
            bag = np.concatenate([bag, padding], axis=0)
        
        # Convert to tensor
        bag_tensor = torch.from_numpy(bag).float()
        
        # Apply transforms if provided
        if self.transform is not None:
            # Apply transform to each slice
            transformed_slices = []
            for i in range(bag_tensor.shape[0]):
                slice_tensor = bag_tensor[i]  # (4, H, W)
                transformed = self.transform(slice_tensor)
                transformed_slices.append(transformed)
            bag_tensor = torch.stack(transformed_slices, dim=0)
        
        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return bag_tensor, label_tensor, patient_id


