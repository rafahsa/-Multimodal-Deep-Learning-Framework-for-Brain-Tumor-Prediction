#!/usr/bin/env python3
"""
Test Ensemble Meta-Learner on New Patient Images

This script loads new patient MRI images from test/DATA_FOR_TEST/ and
evaluates them using the trained ensemble meta-learner model.

Supports two directory structures:
1. Patient subdirectories: test/DATA_FOR_TEST/{PATIENT_ID}/{PATIENT_ID}_T1.nii
2. Flat structure: test/DATA_FOR_TEST/{PATIENT_ID}_T1.nii

The ensemble combines predictions from three base models:
- ResNet50-3D
- SwinUNETR-3D
- DualStreamMIL-3D

Usage:
    python scripts/ensemble/test_ensemble_on_new_patients.py [OPTIONS]
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import SimpleITK as sitk
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.resnet50_3d_fast.model import create_resnet50_3d
from models.swin_unetr_encoder import create_swin_unetr_classifier
from models.dual_stream_mil import create_dual_stream_mil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

RESULTS_DIR = project_root / 'results'


def find_latest_checkpoint(model_name: str, fold: int = 0, use_ema: bool = True) -> Optional[Path]:
    """
    Find the latest checkpoint for a model and fold.
    
    Args:
        model_name: Model name ('ResNet50-3D', 'SwinUNETR-3D', 'DualStreamMIL-3D')
        fold: Fold number (default: 0)
        use_ema: If True, prefer EMA checkpoint (default: True)
    
    Returns:
        Path to checkpoint file, or None if not found
    """
    model_dir = RESULTS_DIR / model_name / 'runs' / f'fold_{fold}'
    if not model_dir.exists():
        return None
    
    # Find all run directories
    run_dirs = sorted([d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('run_')],
                     key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not run_dirs:
        return None
    
    # Check latest run directory
    latest_run = run_dirs[0]
    checkpoint_dir = latest_run / 'checkpoints'
    
    if not checkpoint_dir.exists():
        return None
    
    # Prefer EMA checkpoint if requested
    if use_ema:
        ema_checkpoint = checkpoint_dir / 'best_ema.pt'
        if ema_checkpoint.exists():
            return ema_checkpoint
    
    # Fall back to regular checkpoint
    regular_checkpoint = checkpoint_dir / 'best.pt'
    if regular_checkpoint.exists():
        return regular_checkpoint
    
    return None


def load_model_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict:
    """Load PyTorch checkpoint from file."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        return checkpoint
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        raise


def load_resnet50_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load ResNet50-3D model from checkpoint."""
    checkpoint = load_model_checkpoint(checkpoint_path, device)
    model = create_resnet50_3d(num_classes=2, in_channels=4, dropout=0.4)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_swin_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load SwinUNETR-3D model from checkpoint."""
    checkpoint = load_model_checkpoint(checkpoint_path, device)
    model = create_swin_unetr_classifier(
        num_classes=2, in_channels=4, img_size=(128, 128, 128),
        feature_size=48, use_checkpoint=False, dropout=0.3
    )
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_mil_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load DualStreamMIL-3D model from checkpoint."""
    checkpoint = load_model_checkpoint(checkpoint_path, device)
    # Note: DualStreamMIL doesn't accept input_channels - it's hardcoded to 4 channels in InstanceEncoder
    # Also, bag_size is not a model parameter - it's only used during inference
    model = create_dual_stream_mil(
        num_classes=2,
        instance_encoder_backbone='resnet18',
        instance_encoder_input_size=224,
        attention_type='gated',
        fusion_method='concat',
        dropout=0.5,
        use_hidden_layer=True
    )
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess_volume_resize(volume_array: np.ndarray, target_size: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
    """
    Resize a 3D volume to target size using SimpleITK resampling.
    
    This matches the preprocessing done in Stage 4 (resize to 128x128x128).
    Uses linear interpolation as in training preprocessing.
    
    Args:
        volume_array: 3D numpy array of shape (D, H, W)
        target_size: Target size as (x, y, z) tuple for SimpleITK (which uses xyz order)
                     Note: numpy arrays are (z, y, x), so target_size should be (x, y, z)
    
    Returns:
        Resized 3D numpy array of shape (target_size[2], target_size[1], target_size[0])
    """
    if volume_array.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume_array.ndim}D with shape {volume_array.shape}")
    
    # Convert numpy array to SimpleITK image
    # Note: numpy arrays are (z, y, x), SimpleITK expects (x, y, z)
    sitk_image = sitk.GetImageFromArray(volume_array)
    
    # Get current image properties
    old_size = np.array(sitk_image.GetSize())  # SimpleITK: (x, y, z)
    old_spacing = np.array(sitk_image.GetSpacing())  # (x, y, z)
    old_origin = sitk_image.GetOrigin()
    old_direction = sitk_image.GetDirection()
    
    # Convert target_size to tuple (x, y, z)
    target_size_xyz = tuple(int(s) for s in target_size)
    
    # Compute new spacing
    # new_spacing[i] = old_spacing[i] * (old_size[i] / new_size[i])
    new_spacing = old_spacing * (old_size / np.array(target_size_xyz))
    
    # Set up resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size_xyz)
    resampler.SetOutputSpacing(new_spacing.tolist())
    resampler.SetOutputOrigin(old_origin)
    resampler.SetOutputDirection(old_direction)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)  # Linear interpolation as in training
    
    # Execute resampling
    resized_image = resampler.Execute(sitk_image)
    
    # Convert back to numpy array (z, y, x)
    resized_array = sitk.GetArrayFromImage(resized_image).astype(np.float32)
    
    return resized_array


def preprocess_volume_zscore(volume_array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Apply Z-score normalization to a 3D volume.
    
    This matches the preprocessing done in Stage 2 (z-score normalization).
    Normalization is computed ONLY on brain voxels (values > 0).
    Background voxels (zeros) remain zero.
    
    Args:
        volume_array: 3D numpy array of shape (D, H, W)
        eps: Small epsilon to avoid division by zero (default: 1e-8)
    
    Returns:
        Normalized 3D numpy array with same shape
    """
    if volume_array.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume_array.ndim}D with shape {volume_array.shape}")
    
    # Create copy to avoid modifying original
    normalized_array = volume_array.copy().astype(np.float32)
    
    # Find brain voxels (values > 0)
    brain_mask = volume_array > 0
    brain_voxels = volume_array[brain_mask]
    
    if len(brain_voxels) == 0:
        # All zeros - return as is
        logger.warning("  Warning: Volume contains only zeros, skipping normalization")
        return normalized_array
    
    # Compute mean and std on brain voxels only
    mean = float(np.mean(brain_voxels))
    std = float(np.std(brain_voxels))
    
    if std < eps:
        # Very low variance - avoid division by zero
        logger.warning(f"  Warning: Very low std ({std:.2e}) in volume, skipping normalization")
        return normalized_array
    
    # Apply Z-score normalization
    normalized_array = (normalized_array - mean) / (std + eps)
    
    # Preserve background: set voxels that were 0 to 0
    normalized_array[~brain_mask] = 0.0
    
    return normalized_array


def load_nifti_volume(volume_path: Path, preprocess: bool = True) -> np.ndarray:
    """
    Load NIfTI volume and return as numpy array.
    
    Optionally applies preprocessing (resize to 128x128x128 and z-score normalization)
    to match training data preprocessing pipeline.
    
    Args:
        volume_path: Path to NIfTI file (.nii or .nii.gz)
        preprocess: If True, apply resize and z-score normalization (default: True)
    
    Returns:
        3D numpy array of shape (D, H, W)
        If preprocess=True, shape will be (128, 128, 128)
    """
    if not volume_path.exists():
        raise FileNotFoundError(f"Volume not found: {volume_path}")
    
    try:
        sitk_image = sitk.ReadImage(str(volume_path))
        volume = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
        
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got {volume.ndim}D with shape {volume.shape}")
        
        # Apply preprocessing to match training data pipeline
        if preprocess:
            original_shape = volume.shape
            logger.info(f"    Original shape: {original_shape}")
            
            # Step 1: Resize to (128, 128, 128) - matches Stage 4 preprocessing
            if volume.shape != (128, 128, 128):
                logger.info(f"    Resizing from {volume.shape} to (128, 128, 128)...")
                volume = preprocess_volume_resize(volume, target_size=(128, 128, 128))
                logger.info(f"    ✓ Resized to: {volume.shape}")
            
            # Step 2: Z-score normalization - matches Stage 2 preprocessing
            logger.info(f"    Applying z-score normalization...")
            volume = preprocess_volume_zscore(volume, eps=1e-8)
            logger.info(f"    ✓ Normalization applied")
        
        return volume
    except Exception as e:
        raise RuntimeError(f"Error loading {volume_path}: {e}")


def load_patient_images(test_dir: Path, patient_id: str) -> torch.Tensor:
    """
    Load all 4 modalities (T1, T1ce/T1c, T2, FLAIR) for a patient.
    
    Args:
        test_dir: Root test directory containing patient subdirectories
        patient_id: Patient identifier (e.g., 'UCSF-PDGM-0004')
    
    Returns:
        Multi-modal volume tensor of shape (4, D, H, W)
    """
    # Check if patient has a subdirectory or files are directly in test_dir
    patient_subdir = test_dir / patient_id
    if patient_subdir.exists() and patient_subdir.is_dir():
        # Files are in patient subdirectory
        patient_dir = patient_subdir
        logger.info(f"  Loading from patient subdirectory: {patient_dir}")
    else:
        # Files are directly in test_dir
        patient_dir = test_dir
        logger.info(f"  Loading from test directory: {patient_dir}")
    
    # Verify directory exists
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")
    
    # Define modality mappings - handle both T1ce and T1c naming
    # Order matters: T1, T1ce/T1c, T2, FLAIR (must match training data order)
    modality_configs = [
        ('T1', 'T1'),
        ('T1ce', ['T1ce', 'T1c']),  # Try T1ce first, then T1c
        ('T2', 'T2'),
        ('FLAIR', 'FLAIR')
    ]
    
    modalities_data = []
    loaded_modality_names = []
    loaded_shapes = []
    
    for expected_name, search_names in modality_configs:
        if isinstance(search_names, str):
            search_names = [search_names]
        
        volume_path = None
        found_modality = None
        
        for modality_name in search_names:
            # Try both .nii and .nii.gz
            for ext in ['.nii', '.nii.gz']:
                candidate_path = patient_dir / f"{patient_id}_{modality_name}{ext}"
                if candidate_path.exists():
                    volume_path = candidate_path
                    found_modality = modality_name
                    break
            if volume_path:
                break
        
        if not volume_path or not volume_path.exists():
            raise FileNotFoundError(
                f"Modality {expected_name} not found for patient {patient_id}. "
                f"Searched for: {[f'{patient_id}_{name}.nii(.gz)' for name in search_names]} "
                f"in {patient_dir}. Available files: {list(patient_dir.glob('*.nii*'))}"
            )
        
        # Load and preprocess volume (resize to 128x128x128 and z-score normalization)
        logger.info(f"  Loading {found_modality} ({expected_name}) from {volume_path.name}...")
        volume = load_nifti_volume(volume_path, preprocess=True)
        
        # Validate loaded and preprocessed volume
        if volume is None:
            raise ValueError(f"Failed to load volume from {volume_path}")
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got {volume.ndim}D with shape {volume.shape} for {volume_path}")
        if volume.shape != (128, 128, 128):
            raise ValueError(f"Expected preprocessed volume shape (128, 128, 128), got {volume.shape} for {volume_path}")
        if np.isnan(volume).any():
            raise ValueError(f"NaN detected in {modality_name} volume from {volume_path}")
        if np.isinf(volume).any():
            raise ValueError(f"Inf detected in {modality_name} volume from {volume_path}")
        if volume.size == 0:
            raise ValueError(f"Empty volume loaded from {volume_path}")
        
        modalities_data.append(volume)
        loaded_modality_names.append(found_modality)
        loaded_shapes.append(volume.shape)
        logger.info(f"  ✓ Loaded and preprocessed {found_modality} ({expected_name}): shape {volume.shape}")
    
    # Verify all modalities have been preprocessed to the same spatial dimensions (128, 128, 128)
    if len(set(loaded_shapes)) > 1:
        raise ValueError(
            f"Modalities have different spatial dimensions after preprocessing: "
            f"{dict(zip(loaded_modality_names, loaded_shapes))}. "
            f"All should be (128, 128, 128)."
        )
    if loaded_shapes[0] != (128, 128, 128):
        raise ValueError(
            f"Expected all preprocessed volumes to have shape (128, 128, 128), "
            f"but got {loaded_shapes[0]}"
        )
    
    # Stack modalities as channels: (4, 128, 128, 128)
    # Order: T1, T1ce/T1c, T2, FLAIR (must match training data)
    try:
        multi_modal_volume = np.stack(modalities_data, axis=0)
    except ValueError as e:
        raise ValueError(
            f"Failed to stack modalities. Shapes: {loaded_shapes}, "
            f"Modalities: {loaded_modality_names}. Error: {e}"
        )
    
    # Validate stacked volume
    if multi_modal_volume.shape[0] != 4:
        raise ValueError(f"Expected 4 channels after stacking, got {multi_modal_volume.shape[0]}")
    if np.isnan(multi_modal_volume).any():
        raise ValueError(f"NaN detected in stacked multi-modal volume")
    if np.isinf(multi_modal_volume).any():
        raise ValueError(f"Inf detected in stacked multi-modal volume")
    
    # Convert to tensor
    volume_tensor = torch.from_numpy(multi_modal_volume).float()
    
    # Final validation
    if volume_tensor.shape[0] != 4:
        raise ValueError(f"Final volume tensor has {volume_tensor.shape[0]} channels, expected 4")
    if torch.isnan(volume_tensor).any():
        raise ValueError(f"NaN detected in final volume tensor")
    if torch.isinf(volume_tensor).any():
        raise ValueError(f"Inf detected in final volume tensor")
    
    logger.info(f"  ✓ Multi-modal volume stacked successfully: shape {volume_tensor.shape}")
    logger.info(f"    Channel order: {', '.join(loaded_modality_names)}")
    logger.info(f"    Preprocessed: Resized to (128, 128, 128) and z-score normalized")
    
    return volume_tensor


def find_patients(test_dir: Path) -> List[str]:
    """
    Find all patient IDs in the test directory.
    
    Supports two directory structures:
    1. Patient subdirectories: test_dir/{PATIENT_ID}/{PATIENT_ID}_T1.nii
    2. Flat structure: test_dir/{PATIENT_ID}_T1.nii
    
    Returns:
        List of patient IDs
    """
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    patient_ids = set()
    
    # First, check for patient subdirectories
    subdirs = [d for d in test_dir.iterdir() if d.is_dir()]
    
    if subdirs:
        # Structure: test_dir/{PATIENT_ID}/files
        logger.info(f"Found {len(subdirs)} patient subdirectories")
        for subdir in subdirs:
            # Check if subdirectory contains T1 files
            t1_files = list(subdir.glob("*_T1.nii")) + list(subdir.glob("*_T1.nii.gz"))
            if t1_files:
                # Extract patient ID from subdirectory name
                patient_ids.add(subdir.name)
            else:
                # Try extracting from filename if files are named differently
                all_nii = list(subdir.glob("*.nii")) + list(subdir.glob("*.nii.gz"))
                if all_nii:
                    # Extract patient ID from first file (assuming consistent naming)
                    first_file = all_nii[0]
                    # Try to extract patient ID (remove _T1, _T1c, _T2, _FLAIR suffixes)
                    patient_id_candidate = first_file.name.split('_')[0]
                    if patient_id_candidate:
                        patient_ids.add(patient_id_candidate)
    
    # If no subdirectories or no patients found, check flat structure
    if not patient_ids:
        logger.info("Checking for flat file structure")
        # Find all T1 files directly in test_dir
        t1_files = list(test_dir.glob("*_T1.nii")) + list(test_dir.glob("*_T1.nii.gz"))
        
        if not t1_files:
            raise FileNotFoundError(
                f"No T1 files found in {test_dir}. "
                f"Expected either: (1) Patient subdirectories with T1 files, or "
                f"(2) T1 files directly in test directory."
            )
        
        # Extract patient IDs from filenames
        for t1_file in t1_files:
            # Extract patient ID from filename (e.g., UCSF-PDGM-0004_T1.nii -> UCSF-PDGM-0004)
            patient_id = t1_file.name.replace('_T1.nii', '').replace('_T1.nii.gz', '')
            patient_ids.add(patient_id)
    
    if not patient_ids:
        raise FileNotFoundError(f"No patients found in {test_dir}")
    
    return sorted(list(patient_ids))


def predict_resnet50(model: nn.Module, volume: torch.Tensor, device: torch.device) -> float:
    """
    Get HGG probability from ResNet50-3D model.
    
    Args:
        model: Loaded ResNet50-3D model in eval mode
        volume: Multi-modal volume tensor of shape (4, D, H, W)
        device: Torch device (cuda or cpu)
    
    Returns:
        HGG probability as float (0.0 to 1.0)
    
    Note:
        - Extracts probability of class 1 (HGG) from softmax output
        - Validates predictions are in valid range [0, 1]
        - Checks for NaN/Inf values
    """
    model.eval()
    
    # Verify input shape
    if volume.ndim != 4:
        raise ValueError(f"Expected 4D volume (C, D, H, W), got {volume.ndim}D with shape {volume.shape}")
    if volume.shape[0] != 4:
        raise ValueError(f"Expected 4 channels, got {volume.shape[0]} channels")
    
    volume = volume.unsqueeze(0).to(device)  # Add batch dimension: (1, 4, D, H, W)
    
    with torch.no_grad():
        try:
            with autocast():
                logits = model(volume)
                
                # Validate logits
                if torch.isnan(logits).any():
                    raise ValueError(f"NaN detected in ResNet50-3D logits: {logits}")
                if logits.shape != (1, 2):
                    raise ValueError(f"Expected logits shape (1, 2), got {logits.shape}")
                
                probs = torch.softmax(logits, dim=1)
                
                # Validate probabilities
                if torch.isnan(probs).any():
                    raise ValueError(f"NaN detected in ResNet50-3D probabilities: {probs}")
                if torch.isinf(probs).any():
                    raise ValueError(f"Inf detected in ResNet50-3D probabilities: {probs}")
                if not torch.allclose(probs.sum(dim=1), torch.ones(1).to(device), atol=1e-5):
                    raise ValueError(f"Probabilities don't sum to 1: {probs.sum(dim=1)}")
                
                # Extract HGG probability (class 1) and ensure it's a Python float
                hgg_prob_tensor = probs[0, 1].detach().cpu()
                hgg_prob = float(hgg_prob_tensor.item())
                
                # Validate final probability
                if np.isnan(hgg_prob) or np.isinf(hgg_prob):
                    raise ValueError(f"Invalid HGG probability: {hgg_prob}")
                if not (0.0 <= hgg_prob <= 1.0):
                    raise ValueError(f"HGG probability out of range [0, 1]: {hgg_prob}")
                
                return hgg_prob
                
        except Exception as e:
            logger.error(f"Error in ResNet50-3D prediction: {e}")
            logger.error(f"  Input volume shape: {volume.shape}")
            raise


def predict_swin(model: nn.Module, volume: torch.Tensor, device: torch.device) -> float:
    """
    Get HGG probability from SwinUNETR-3D model.
    
    Args:
        model: Loaded SwinUNETR-3D model in eval mode
        volume: Multi-modal volume tensor of shape (4, D, H, W)
        device: Torch device (cuda or cpu)
    
    Returns:
        HGG probability as float (0.0 to 1.0)
    
    Note:
        - Extracts probability of class 1 (HGG) from softmax output
        - Validates predictions are in valid range [0, 1]
        - Checks for NaN/Inf values
    """
    model.eval()
    
    # Verify input shape
    if volume.ndim != 4:
        raise ValueError(f"Expected 4D volume (C, D, H, W), got {volume.ndim}D with shape {volume.shape}")
    if volume.shape[0] != 4:
        raise ValueError(f"Expected 4 channels, got {volume.shape[0]} channels")
    
    volume = volume.unsqueeze(0).to(device)  # Add batch dimension: (1, 4, D, H, W)
    
    with torch.no_grad():
        try:
            with autocast():
                logits = model(volume)
                
                # Validate logits
                if torch.isnan(logits).any():
                    raise ValueError(f"NaN detected in SwinUNETR-3D logits: {logits}")
                if logits.shape != (1, 2):
                    raise ValueError(f"Expected logits shape (1, 2), got {logits.shape}")
                
                probs = torch.softmax(logits, dim=1)
                
                # Validate probabilities
                if torch.isnan(probs).any():
                    raise ValueError(f"NaN detected in SwinUNETR-3D probabilities: {probs}")
                if torch.isinf(probs).any():
                    raise ValueError(f"Inf detected in SwinUNETR-3D probabilities: {probs}")
                if not torch.allclose(probs.sum(dim=1), torch.ones(1).to(device), atol=1e-5):
                    raise ValueError(f"Probabilities don't sum to 1: {probs.sum(dim=1)}")
                
                # Extract HGG probability (class 1) and ensure it's a Python float
                hgg_prob_tensor = probs[0, 1].detach().cpu()
                hgg_prob = float(hgg_prob_tensor.item())
                
                # Validate final probability
                if np.isnan(hgg_prob) or np.isinf(hgg_prob):
                    raise ValueError(f"Invalid HGG probability: {hgg_prob}")
                if not (0.0 <= hgg_prob <= 1.0):
                    raise ValueError(f"HGG probability out of range [0, 1]: {hgg_prob}")
                
                return hgg_prob
                
        except Exception as e:
            logger.error(f"Error in SwinUNETR-3D prediction: {e}")
            logger.error(f"  Input volume shape: {volume.shape}")
            raise


def predict_mil(model: nn.Module, volume: torch.Tensor, device: torch.device, bag_size: int = 32) -> float:
    """
    Get HGG probability from DualStreamMIL-3D model.
    
    Args:
        model: Loaded DualStreamMIL-3D model in eval mode
        volume: Multi-modal volume tensor of shape (4, D, H, W)
        device: Torch device (cuda or cpu)
        bag_size: Number of slices to sample for MIL bag (default: 32)
    
    Returns:
        HGG probability as float (0.0 to 1.0)
    
    Note:
        - Samples slices from depth dimension to create bag
        - Extracts probability of class 1 (HGG) from softmax output
        - Validates predictions are in valid range [0, 1]
        - Checks for NaN/Inf values
        - Model expects input shape: (B, N, 4, H, W) where N is number of slices
    """
    model.eval()
    
    # Verify input shape
    if volume.ndim != 4:
        raise ValueError(f"Expected 4D volume (C, D, H, W), got {volume.ndim}D with shape {volume.shape}")
    if volume.shape[0] != 4:
        raise ValueError(f"Expected 4 channels, got {volume.shape[0]} channels")
    
    # MIL models expect bags of slices
    # Extract axial slices from volume (modality, depth, height, width)
    volume_np = volume.cpu().numpy()  # Shape: (4, D, H, W)
    
    # Validate volume data
    if np.isnan(volume_np).any():
        raise ValueError(f"NaN detected in input volume")
    if np.isinf(volume_np).any():
        raise ValueError(f"Inf detected in input volume")
    
    # Sample slices from the depth dimension
    depth = volume_np.shape[1]
    if depth <= 0:
        raise ValueError(f"Invalid depth dimension: {depth}")
    
    if depth <= bag_size:
        slice_indices = list(range(depth))
    else:
        # Sample evenly distributed slices
        slice_indices = np.linspace(0, depth - 1, bag_size, dtype=int).tolist()
        slice_indices = sorted(list(set(slice_indices)))  # Ensure unique and sorted
    
    if not slice_indices:
        raise ValueError(f"No valid slice indices generated. Depth: {depth}, Bag size: {bag_size}")
    
    # Extract slices: each slice is (4, H, W)
    slices = []
    for idx in slice_indices:
        if idx < 0 or idx >= depth:
            continue
        slice_data = volume_np[:, idx, :, :]  # Shape: (4, H, W)
        if slice_data.shape != (4, volume_np.shape[2], volume_np.shape[3]):
            raise ValueError(f"Unexpected slice shape: {slice_data.shape}, expected (4, {volume_np.shape[2]}, {volume_np.shape[3]})")
        slices.append(slice_data)
    
    if not slices:
        raise ValueError(f"No valid slices extracted. Depth: {depth}, Indices: {slice_indices}")
    
    # Stack into bag: (N, 4, H, W) where N = number of slices
    bag = np.stack(slices, axis=0)
    
    # Validate bag
    if np.isnan(bag).any():
        raise ValueError(f"NaN detected in bag after stacking slices")
    
    logger.debug(f"  MIL bag shape: {bag.shape} (expected: (N, 4, H, W))")
    
    # Convert to tensor and add batch dimension: (1, N, 4, H, W)
    bag_tensor = torch.from_numpy(bag).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        try:
            with autocast():
                # MIL model forward expects temperature parameter
                # Note: model.forward() can return either logits or (logits, interpretability_info)
                # We explicitly pass return_interpretability=False to get only logits
                model_output = model(bag_tensor, temperature=1.0, return_interpretability=False)
                
                # Handle both return types (logits only, or tuple) - though with return_interpretability=False should be logits only
                if isinstance(model_output, tuple):
                    logits = model_output[0]
                    logger.warning(f"  MIL model returned tuple, extracting logits. Expected logits only with return_interpretability=False")
                else:
                    logits = model_output
                
                # Ensure logits is a tensor
                if not isinstance(logits, torch.Tensor):
                    raise ValueError(f"Expected logits to be a torch.Tensor, got {type(logits)}: {model_output}")
                
                # Ensure logits are detached from computation graph
                logits = logits.detach()
                
                # Validate logits
                if torch.isnan(logits).any():
                    raise ValueError(f"NaN detected in DualStreamMIL-3D logits: {logits}")
                if torch.isinf(logits).any():
                    raise ValueError(f"Inf detected in DualStreamMIL-3D logits: {logits}")
                if logits.shape != (1, 2):
                    raise ValueError(f"Expected logits shape (1, 2), got {logits.shape}")
                
                probs = torch.softmax(logits, dim=1)
                
                # Validate probabilities
                if torch.isnan(probs).any():
                    raise ValueError(f"NaN detected in DualStreamMIL-3D probabilities: {probs}")
                if torch.isinf(probs).any():
                    raise ValueError(f"Inf detected in DualStreamMIL-3D probabilities: {probs}")
                if not torch.allclose(probs.sum(dim=1), torch.ones(1).to(device), atol=1e-5):
                    raise ValueError(f"Probabilities don't sum to 1: {probs.sum(dim=1)}")
                
                # Extract HGG probability (class 1) and ensure it's a Python float
                hgg_prob_tensor = probs[0, 1].detach().cpu()
                hgg_prob = float(hgg_prob_tensor.item())
                
                # Validate final probability
                if np.isnan(hgg_prob) or np.isinf(hgg_prob):
                    raise ValueError(f"Invalid HGG probability: {hgg_prob}")
                if not (0.0 <= hgg_prob <= 1.0):
                    raise ValueError(f"HGG probability out of range [0, 1]: {hgg_prob}")
                
                return hgg_prob
                
        except Exception as e:
            logger.error(f"Error in DualStreamMIL-3D prediction: {e}")
            logger.error(f"  Input volume shape: {volume.shape}")
            logger.error(f"  Bag tensor shape: {bag_tensor.shape}")
            logger.error(f"  Number of slices: {len(slice_indices)}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description='Test ensemble meta-learner on new patient images'
    )
    
    parser.add_argument(
        '--test-dir',
        type=str,
        default='test/DATA_FOR_TEST',
        help='Directory containing test patient images (default: test/DATA_FOR_TEST)'
    )
    parser.add_argument(
        '--meta-learner',
        type=str,
        default='ensemble/models/meta_learner_logistic_regression.joblib',
        help='Path to Logistic Regression meta-learner (default: ensemble/models/meta_learner_logistic_regression.joblib)'
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help='Fold number for auto-detecting checkpoints (default: 0)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for inference (default: auto)'
    )
    parser.add_argument(
        '--bag-size',
        type=int,
        default=32,
        help='Bag size for MIL model (default: 32)'
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info("=" * 80)
    logger.info("Ensemble Meta-Learner Test on New Patients")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    
    # Resolve paths
    test_dir = Path(args.test_dir)
    if not test_dir.is_absolute():
        test_dir = project_root / test_dir
    
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        sys.exit(1)
    
    meta_learner_path = Path(args.meta_learner)
    if not meta_learner_path.is_absolute():
        meta_learner_path = project_root / meta_learner_path
    
    if not meta_learner_path.exists():
        logger.error(f"Meta-learner not found: {meta_learner_path}")
        sys.exit(1)
    
    # Find patients
    logger.info(f"\nScanning test directory: {test_dir}")
    patient_ids = find_patients(test_dir)
    logger.info(f"Found {len(patient_ids)} patients: {patient_ids}")
    
    if not patient_ids:
        logger.error("No patients found in test directory")
        sys.exit(1)
    
    # Auto-detect checkpoints
    logger.info("\n" + "=" * 80)
    logger.info("Loading Models")
    logger.info("=" * 80)
    
    resnet_checkpoint = find_latest_checkpoint('ResNet50-3D', fold=args.fold, use_ema=True)
    swin_checkpoint = find_latest_checkpoint('SwinUNETR-3D', fold=args.fold, use_ema=True)
    mil_checkpoint = find_latest_checkpoint('DualStreamMIL-3D', fold=args.fold, use_ema=True)
    
    if not all([resnet_checkpoint, swin_checkpoint, mil_checkpoint]):
        logger.error("Could not find all required checkpoints. Please ensure models are trained.")
        sys.exit(1)
    
    logger.info(f"ResNet50-3D checkpoint: {resnet_checkpoint}")
    logger.info(f"SwinUNETR-3D checkpoint: {swin_checkpoint}")
    logger.info(f"DualStreamMIL-3D checkpoint: {mil_checkpoint}")
    
    # Load models
    logger.info("\nLoading base models...")
    resnet_model = load_resnet50_model(resnet_checkpoint, device)
    swin_model = load_swin_model(swin_checkpoint, device)
    mil_model = load_mil_model(mil_checkpoint, device)
    logger.info("✓ All base models loaded")
    
    # Load meta-learner
    logger.info(f"\nLoading meta-learner: {meta_learner_path}")
    meta_learner = joblib.load(meta_learner_path)
    
    # Validate meta-learner
    if not hasattr(meta_learner, 'predict'):
        raise ValueError("Meta-learner does not have 'predict' method")
    if not hasattr(meta_learner, 'predict_proba'):
        raise ValueError("Meta-learner does not have 'predict_proba' method")
    
    # Check expected feature count (should be 3: ResNet, Swin, MIL)
    if hasattr(meta_learner, 'n_features_in_'):
        expected_features = meta_learner.n_features_in_
        if expected_features != 3:
            raise ValueError(f"Meta-learner expects {expected_features} features, but we provide 3 (ResNet, Swin, MIL)")
        logger.info(f"  Meta-learner expects {expected_features} features (matches our 3 base models)")
    elif hasattr(meta_learner, 'coef_'):
        # LogisticRegression has coef_ shape (n_classes, n_features)
        if meta_learner.coef_.shape[1] != 3:
            raise ValueError(f"Meta-learner coefficients shape {meta_learner.coef_.shape} suggests {meta_learner.coef_.shape[1]} features, but we provide 3")
        logger.info(f"  Meta-learner coefficients shape: {meta_learner.coef_.shape} (expects {meta_learner.coef_.shape[1]} features)")
    
    # Verify class order for predict_proba
    if hasattr(meta_learner, 'classes_'):
        # classes_ array indicates the order of classes in predict_proba output
        # For binary classification with labels [0, 1], classes_ should be [0, 1]
        # predict_proba returns probabilities in order [prob_class_0, prob_class_1]
        # So index 0 = LGG (class 0), index 1 = HGG (class 1)
        if len(meta_learner.classes_) == 2:
            logger.info(f"  Meta-learner classes: {meta_learner.classes_} (LGG={meta_learner.classes_[0]}, HGG={meta_learner.classes_[1]})")
            if meta_learner.classes_[1] != 1:
                logger.warning(f"  Warning: HGG class index is {meta_learner.classes_[1]}, not 1. Adjust probability extraction if needed.")
        else:
            logger.warning(f"  Warning: Unexpected number of classes: {len(meta_learner.classes_)}")
    
    logger.info("✓ Meta-learner loaded and validated")
    
    # Process each patient
    logger.info("\n" + "=" * 80)
    logger.info("Processing Patients")
    logger.info("=" * 80)
    
    results = []
    
    for patient_id in patient_ids:
        logger.info(f"\nProcessing patient: {patient_id}")
        logger.info("-" * 80)
        
        try:
            # Load patient images
            logger.info(f"Loading images for patient {patient_id}...")
            volume = load_patient_images(test_dir, patient_id)
            
            # Validate volume after loading
            if volume is None:
                raise ValueError(f"Volume is None for patient {patient_id}")
            if not isinstance(volume, torch.Tensor):
                raise ValueError(f"Volume is not a Tensor, got {type(volume)}")
            if volume.ndim != 4:
                raise ValueError(f"Expected 4D volume, got {volume.ndim}D with shape {volume.shape}")
            if volume.shape[0] != 4:
                raise ValueError(f"Expected 4 channels, got {volume.shape[0]} channels")
            if torch.isnan(volume).any():
                raise ValueError(f"NaN detected in loaded volume")
            if torch.isinf(volume).any():
                raise ValueError(f"Inf detected in loaded volume")
            
            logger.info(f"  ✓ Volume loaded successfully: shape {volume.shape}")
            
            # Get predictions from base models with validation
            logger.info("\nGenerating base model predictions...")
            
            hgg_prob_resnet = predict_resnet50(resnet_model, volume, device)
            logger.info(f"  ✓ ResNet50-3D HGG probability: {hgg_prob_resnet:.6f}")
            
            # Validate ResNet prediction
            if np.isnan(hgg_prob_resnet) or np.isinf(hgg_prob_resnet):
                raise ValueError(f"Invalid ResNet50-3D prediction: {hgg_prob_resnet}")
            if not (0.0 <= hgg_prob_resnet <= 1.0):
                raise ValueError(f"ResNet50-3D prediction out of range [0, 1]: {hgg_prob_resnet}")
            
            hgg_prob_swin = predict_swin(swin_model, volume, device)
            logger.info(f"  ✓ SwinUNETR-3D HGG probability: {hgg_prob_swin:.6f}")
            
            # Validate Swin prediction
            if np.isnan(hgg_prob_swin) or np.isinf(hgg_prob_swin):
                raise ValueError(f"Invalid SwinUNETR-3D prediction: {hgg_prob_swin}")
            if not (0.0 <= hgg_prob_swin <= 1.0):
                raise ValueError(f"SwinUNETR-3D prediction out of range [0, 1]: {hgg_prob_swin}")
            
            hgg_prob_mil = predict_mil(mil_model, volume, device, bag_size=args.bag_size)
            logger.info(f"  ✓ DualStreamMIL-3D HGG probability: {hgg_prob_mil:.6f}")
            
            # Validate MIL prediction
            if np.isnan(hgg_prob_mil) or np.isinf(hgg_prob_mil):
                raise ValueError(f"Invalid DualStreamMIL-3D prediction: {hgg_prob_mil}")
            if not (0.0 <= hgg_prob_mil <= 1.0):
                raise ValueError(f"DualStreamMIL-3D prediction out of range [0, 1]: {hgg_prob_mil}")
            
            # Validate all predictions are valid before combining
            predictions = [hgg_prob_resnet, hgg_prob_swin, hgg_prob_mil]
            if any(np.isnan(p) or np.isinf(p) for p in predictions):
                raise ValueError(f"NaN or Inf detected in base model predictions: "
                               f"ResNet={hgg_prob_resnet}, Swin={hgg_prob_swin}, MIL={hgg_prob_mil}")
            if any(not (0.0 <= p <= 1.0) for p in predictions):
                raise ValueError(f"Predictions out of valid range [0, 1]: "
                               f"ResNet={hgg_prob_resnet}, Swin={hgg_prob_swin}, MIL={hgg_prob_mil}")
            
            # Prepare features for meta-learner
            # CRITICAL: Feature order must match training order: [hgg_prob_resnet, hgg_prob_swin, hgg_prob_mil]
            # This matches FEATURE_COLUMNS in train_meta_learner.py: ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']
            logger.info("\nPreparing features for ensemble meta-learner...")
            logger.info("  Feature order: [ResNet50-3D, SwinUNETR-3D, DualStreamMIL-3D] (must match training)")
            
            features = np.array([[hgg_prob_resnet, hgg_prob_swin, hgg_prob_mil]], dtype=np.float32)
            
            # Validate features array
            if features.shape != (1, 3):
                raise ValueError(f"Expected features shape (1, 3), got {features.shape}")
            if np.isnan(features).any():
                raise ValueError(f"NaN detected in features array: {features}")
            if np.isinf(features).any():
                raise ValueError(f"Inf detected in features array: {features}")
            
            # Verify feature values are reasonable
            if np.any(features < 0) or np.any(features > 1):
                raise ValueError(f"Features out of valid range [0, 1]: {features}")
            
            logger.info(f"  ✓ Features array created: shape {features.shape}, dtype {features.dtype}")
            logger.info(f"    [0] ResNet50-3D:    {features[0, 0]:.6f}")
            logger.info(f"    [1] SwinUNETR-3D:   {features[0, 1]:.6f}")
            logger.info(f"    [2] DualStreamMIL-3D: {features[0, 2]:.6f}")
            
            # Verify feature order matches meta-learner expectations (if coefficients available)
            if hasattr(meta_learner, 'coef_') and meta_learner.coef_.shape[1] == 3:
                logger.info(f"  ✓ Feature order verified: meta-learner expects {meta_learner.coef_.shape[1]} features")
            
            # Get ensemble prediction
            logger.info("Running ensemble meta-learner...")
            
            # Get prediction
            ensemble_prediction_array = meta_learner.predict(features)
            if len(ensemble_prediction_array) != 1:
                raise ValueError(f"Expected single prediction, got array of length {len(ensemble_prediction_array)}")
            ensemble_prediction = int(ensemble_prediction_array[0])
            
            # Get probabilities (shape: (n_samples, n_classes))
            ensemble_proba_array = meta_learner.predict_proba(features)
            if ensemble_proba_array.shape != (1, 2):
                raise ValueError(f"Expected proba shape (1, 2), got {ensemble_proba_array.shape}")
            
            # Validate proba array
            if np.isnan(ensemble_proba_array).any():
                raise ValueError(f"NaN detected in ensemble probabilities: {ensemble_proba_array}")
            if np.isinf(ensemble_proba_array).any():
                raise ValueError(f"Inf detected in ensemble probabilities: {ensemble_proba_array}")
            if not np.allclose(ensemble_proba_array.sum(axis=1), 1.0, atol=1e-5):
                raise ValueError(f"Ensemble probabilities don't sum to 1: {ensemble_proba_array.sum(axis=1)}")
            
            # Extract HGG probability (class 1) - index [0, 1] means first sample, second class
            ensemble_probability = float(ensemble_proba_array[0, 1])
            lgg_probability = float(ensemble_proba_array[0, 0])
            
            logger.info(f"  Meta-learner output probabilities: LGG={lgg_probability:.6f}, HGG={ensemble_probability:.6f}")
            
            # Validate ensemble predictions
            if np.isnan(ensemble_probability) or np.isinf(ensemble_probability):
                raise ValueError(f"Invalid ensemble HGG probability: {ensemble_probability}")
            if not (0.0 <= ensemble_probability <= 1.0):
                raise ValueError(f"Ensemble HGG probability out of range [0, 1]: {ensemble_probability}")
            if ensemble_prediction not in [0, 1]:
                raise ValueError(f"Invalid ensemble prediction: {ensemble_prediction} (expected 0 or 1, got {type(ensemble_prediction)})")
            
            logger.info(f"  ✓ Ensemble HGG probability: {ensemble_probability:.6f}")
            logger.info(f"  ✓ Ensemble prediction: {'HGG' if ensemble_prediction == 1 else 'LGG'}")
            
            # Store results
            results.append({
                'patient_id': patient_id,
                'resnet_prob': float(hgg_prob_resnet),
                'swin_prob': float(hgg_prob_swin),
                'mil_prob': float(hgg_prob_mil),
                'ensemble_prob': float(ensemble_probability),
                'ensemble_pred': int(ensemble_prediction)
            })
            
            logger.info(f"✓ Successfully processed patient {patient_id}")
            
        except Exception as e:
            logger.error(f"✗ Error processing patient {patient_id}: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            continue
    
    # Print results in the requested format
    print("\n" + "=" * 80)
    print("ENSEMBLE MODEL TEST RESULTS")
    print("=" * 80)
    
    for result in results:
        patient_id = result['patient_id']
        # Extract patient number (e.g., "UCSF-PDGM-0004" -> "0004")
        # Handle various formats: UCSF-PDGM-0004 -> 0004, Patient001 -> 001, etc.
        if '-' in patient_id:
            patient_num = patient_id.split('-')[-1]
        elif '_' in patient_id:
            parts = patient_id.split('_')
            # Take last part that looks like a number
            for part in reversed(parts):
                if any(c.isdigit() for c in part):
                    patient_num = ''.join(filter(str.isdigit, part))
                    break
            else:
                patient_num = parts[-1]
        else:
            # Extract digits from the end
            patient_num = ''.join(filter(str.isdigit, patient_id)) or patient_id
        
        print(f"\nPatient: {patient_id}")
        print("-" * 80)
        print(f"  ResNet50-3D HGG probability:    {result['resnet_prob']:.6f}")
        print(f"  SwinUNETR-3D HGG probability:   {result['swin_prob']:.6f}")
        print(f"  DualStreamMIL-3D HGG probability: {result['mil_prob']:.6f}")
        print(f"  Ensemble HGG probability:       {result['ensemble_prob']:.6f}")
        print(f"  Ensemble prediction:            {'HGG' if result['ensemble_pred'] == 1 else 'LGG'}")
        
        # Format as requested: "HGG probability for patient 0004: 0.95"
        print(f"HGG probability for patient {patient_num}: {result['ensemble_prob']:.2f}")
    
    print("\n" + "=" * 80)
    print(f"Processed {len(results)} patients successfully")
    print("=" * 80)
    
    return results


if __name__ == '__main__':
    main()

