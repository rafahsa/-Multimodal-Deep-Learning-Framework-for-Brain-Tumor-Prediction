"""
Step 1: Test-Time Augmentation (TTA) for Swin and ResNet

Applies light, MRI-safe augmentations to Swin and ResNet inference.
Generates N=8-16 predictions per patient and averages probabilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import SimpleITK as sitk
import sys
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.resnet50_3d_fast.model import create_resnet50_3d
from models.swin_unetr_encoder import create_swin_unetr_classifier
from utils.augmentations_3d import get_train_transforms_3d

# Paths
DATA_ROOT = PROJECT_ROOT / 'data' / 'processed' / 'stage_4_resize' / 'train'
SPLITS_DIR = PROJECT_ROOT / 'splits'
RESULTS_DIR = PROJECT_ROOT / 'results'
NUM_TTA = 12  # Number of TTA samples per patient


def find_latest_checkpoint(model_name: str, fold: int, use_ema: bool = True) -> Optional[Path]:
    """Find latest checkpoint for a model and fold."""
    model_dirs = {
        'ResNet50-3D': RESULTS_DIR / 'ResNet50-3D' / 'runs',
        'SwinUNETR-3D': RESULTS_DIR / 'SwinUNETR-3D' / 'runs'
    }
    
    if model_name not in model_dirs:
        return None
    
    fold_dir = model_dirs[model_name] / f'fold_{fold}'
    if not fold_dir.exists():
        return None
    
    # Find latest run
    runs = sorted(fold_dir.glob('run_*'), key=lambda x: x.stat().st_mtime, reverse=True)
    if not runs:
        return None
    
    latest_run = runs[0]
    checkpoint_dir = latest_run / 'checkpoints'
    
    if use_ema:
        checkpoint_path = checkpoint_dir / 'best_ema.pt'
        if checkpoint_path.exists():
            return checkpoint_path
    
    checkpoint_path = checkpoint_dir / 'best.pt'
    if checkpoint_path.exists():
        return checkpoint_path
    
    return None


def load_volume(patient_id: str, class_name: str) -> Optional[torch.Tensor]:
    """Load multi-modal volume for a patient."""
    patient_dir = DATA_ROOT / class_name / patient_id
    if not patient_dir.exists():
        return None
    
    modalities = ['t1', 't1ce', 't2', 'flair']
    volume_channels = []
    
    for mod in modalities:
        volume_path = patient_dir / f"{patient_id}_{mod}.nii.gz"
        if not volume_path.exists():
            volume_path = patient_dir / f"{patient_id}_{mod}.nii"
        
        if not volume_path.exists():
            return None
        
        try:
            volume = sitk.ReadImage(str(volume_path))
            volume_array = sitk.GetArrayFromImage(volume).astype(np.float32)
            volume_channels.append(volume_array)
        except Exception as e:
            logger.warning(f"Error loading {volume_path}: {e}")
            return None
    
    # Stack modalities: (4, D, H, W)
    multi_modal_volume = np.stack(volume_channels, axis=0)
    return torch.from_numpy(multi_modal_volume).float()


def get_tta_transforms():
    """Get light, MRI-safe TTA transforms."""
    try:
        from monai.transforms import (
            RandAffine, RandFlip, RandGaussianNoise,
            Compose, EnsureChannelFirstD, ToTensorD
        )
        
        # Light augmentations: small rotations, flips, intensity shifts
        transforms = [
            EnsureChannelFirstD(keys=['image']),
            RandAffine(
                prob=0.8,
                rotate_range=(0.1, 0.1, 0.1),  # Small rotation (±5.7 degrees)
                translate_range=(5, 5, 5),  # Small translation
                scale_range=(0.95, 1.05),  # Small scale
                mode='bilinear',
                padding_mode='zeros'
            ),
            RandFlip(prob=0.5, spatial_axis=[0, 1, 2]),
            RandGaussianNoise(prob=0.3, std=0.01),  # Light noise
            ToTensorD(keys=['image'])
        ]
        
        return Compose(transforms)
    except ImportError:
        logger.warning("MONAI not available, using simple transforms")
        return None


def apply_simple_tta(volume: torch.Tensor) -> torch.Tensor:
    """Apply simple TTA augmentation (flip, small rotation)."""
    # Random flip
    if np.random.rand() > 0.5:
        volume = torch.flip(volume, dims=[1])  # Flip along depth
    if np.random.rand() > 0.5:
        volume = torch.flip(volume, dims=[2])  # Flip along height
    if np.random.rand() > 0.5:
        volume = torch.flip(volume, dims=[3])  # Flip along width
    
    # Small intensity shift
    volume = volume + torch.randn_like(volume) * 0.01
    
    return volume


def predict_with_tta(
    model: nn.Module,
    volume: torch.Tensor,
    device: torch.device,
    num_tta: int = NUM_TTA
) -> float:
    """Predict with TTA by averaging over multiple augmented versions."""
    model.eval()
    predictions = []
    
    # Get TTA transforms
    tta_transform = get_tta_transforms()
    
    with torch.no_grad():
        for _ in range(num_tta):
            if tta_transform is not None:
                # Use MONAI transforms
                volume_dict = {'image': volume.numpy()}
                volume_dict = tta_transform(volume_dict)
                volume_aug = volume_dict['image']
                
                if isinstance(volume_aug, torch.Tensor):
                    volume_aug = volume_aug.to(device)
                else:
                    volume_aug = torch.from_numpy(volume_aug).float().to(device)
            else:
                # Fallback: simple augmentation
                volume_aug = apply_simple_tta(volume).to(device)
            
            # Add batch dimension
            volume_aug = volume_aug.unsqueeze(0)  # (1, 4, D, H, W)
            
            # Predict
            with autocast():
                logits = model(volume_aug)
                probs = torch.softmax(logits, dim=1)
                hgg_prob = probs[0, 1].item()
                predictions.append(hgg_prob)
    
    # Average predictions
    return np.mean(predictions)


def apply_tta_to_oof(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Apply TTA to Swin and ResNet OOF predictions.
    
    For each patient:
    1. Load volume
    2. Load model for that fold
    3. Apply TTA (N=12 augmentations)
    4. Average probabilities
    5. Save as swin_prob_tta and resnet_prob_tta
    """
    logger.info("Applying Test-Time Augmentation to Swin and ResNet...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output DataFrame
    df_tta = df.copy()
    df_tta['swin_prob_tta'] = np.nan
    df_tta['resnet_prob_tta'] = np.nan
    
    # Get class mapping from patient_id
    # Infer class from data directory or use label
    def get_class_name(patient_id: str, label: int) -> str:
        hgg_dir = DATA_ROOT / 'HGG' / patient_id
        lgg_dir = DATA_ROOT / 'LGG' / patient_id
        if hgg_dir.exists():
            return 'HGG'
        elif lgg_dir.exists():
            return 'LGG'
        else:
            return 'HGG' if label == 1 else 'LGG'
    
    # Process by fold to load models efficiently
    for fold in range(5):
        logger.info(f"\nProcessing fold {fold}...")
        
        # Load models for this fold
        resnet_checkpoint = find_latest_checkpoint('ResNet50-3D', fold, use_ema=True)
        swin_checkpoint = find_latest_checkpoint('SwinUNETR-3D', fold, use_ema=True)
        
        if not resnet_checkpoint or not swin_checkpoint:
            logger.warning(f"Missing checkpoints for fold {fold}, skipping")
            continue
        
        # Load models
        logger.info(f"Loading ResNet checkpoint: {resnet_checkpoint}")
        resnet_checkpoint_data = torch.load(resnet_checkpoint, map_location=device, weights_only=False)
        resnet_model = create_resnet50_3d(num_classes=2, in_channels=4, dropout=0.4)
        state_dict = resnet_checkpoint_data['model_state_dict']
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        resnet_model.load_state_dict(state_dict)
        resnet_model.to(device)
        resnet_model.eval()
        
        logger.info(f"Loading Swin checkpoint: {swin_checkpoint}")
        swin_checkpoint_data = torch.load(swin_checkpoint, map_location=device, weights_only=False)
        swin_model = create_swin_unetr_classifier(
            num_classes=2, in_channels=4, img_size=(128, 128, 128),
            feature_size=48, use_checkpoint=False, dropout=0.3
        )
        state_dict = swin_checkpoint_data['model_state_dict']
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        swin_model.load_state_dict(state_dict)
        swin_model.to(device)
        swin_model.eval()
        
        # Get patients in this fold
        fold_mask = df_tta['fold'] == fold
        fold_patients = df_tta[fold_mask]
        
        logger.info(f"Processing {len(fold_patients)} patients in fold {fold}...")
        
        # Process each patient
        for idx, row in tqdm(fold_patients.iterrows(), total=len(fold_patients), desc=f"Fold {fold}"):
            patient_id = row['patient_id']
            label = row['label']
            class_name = get_class_name(patient_id, label)
            
            # Load volume
            volume = load_volume(patient_id, class_name)
            if volume is None:
                logger.warning(f"Could not load volume for {patient_id}, skipping")
                continue
            
            # Apply TTA for ResNet
            try:
                resnet_prob_tta = predict_with_tta(resnet_model, volume, device, num_tta=NUM_TTA)
                df_tta.loc[idx, 'resnet_prob_tta'] = resnet_prob_tta
            except Exception as e:
                logger.warning(f"Error in ResNet TTA for {patient_id}: {e}")
                # Fallback to original probability
                df_tta.loc[idx, 'resnet_prob_tta'] = row.get('hgg_prob_resnet', 0.5)
            
            # Apply TTA for Swin
            try:
                swin_prob_tta = predict_with_tta(swin_model, volume, device, num_tta=NUM_TTA)
                df_tta.loc[idx, 'swin_prob_tta'] = swin_prob_tta
            except Exception as e:
                logger.warning(f"Error in Swin TTA for {patient_id}: {e}")
                # Fallback to original probability
                df_tta.loc[idx, 'swin_prob_tta'] = row.get('hgg_prob_swin', 0.5)
        
        # Free GPU memory
        del resnet_model, swin_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Fill any remaining NaN with original probabilities
    df_tta['resnet_prob_tta'] = df_tta['resnet_prob_tta'].fillna(df_tta['hgg_prob_resnet'])
    df_tta['swin_prob_tta'] = df_tta['swin_prob_tta'].fillna(df_tta['hgg_prob_swin'])
    
    # Save results
    output_file = output_dir / 'oof_predictions_with_tta.csv'
    df_tta.to_csv(output_file, index=False)
    logger.info(f"\n✓ Saved TTA predictions to: {output_file}")
    
    # Compare with baseline
    logger.info("\nTTA vs Baseline Comparison:")
    logger.info(f"ResNet - Mean change: {(df_tta['resnet_prob_tta'] - df_tta['hgg_prob_resnet']).mean():.6f}")
    logger.info(f"Swin - Mean change: {(df_tta['swin_prob_tta'] - df_tta['hgg_prob_swin']).mean():.6f}")
    
    return df_tta

