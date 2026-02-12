#!/usr/bin/env python3
"""
Generate 3D Grad-CAM Heatmaps for ResNet50-3D

This script generates Grad-CAM heatmaps for selected patient volumes using
trained ResNet50-3D checkpoints, then visualizes them as 2D overlays in
the three standard planes (axial, coronal, sagittal).

Usage:
    python scripts/analysis/generate_cnn_gradcam_3d.py \
        --checkpoint "AUTO" \
        --patient_ids_file "data/selected_patients.txt" \
        --output_dir "ensemble/results/interpretability/cnn_gradcam" \
        --target_class "pred" \
        --num_slices 12 \
        --fold 0

Author: Medical Imaging Pipeline
"""

import sys
from pathlib import Path

# Set project root explicitly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import SimpleITK as sitk

# Project imports
from models.resnet50_3d_fast.model import create_resnet50_3d
from utils.interpretability.gradcam_3d import create_gradcam_for_resnet50_3d

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_ROOT = PROJECT_ROOT / 'data' / 'processed' / 'stage_4_resize' / 'train'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Variant configurations
VARIANT_CONFIGS = {
    'mri': {
        'model_name': 'ResNet50-3D',
        'checkpoint_base': 'results/ResNet50-3D',
        'input_type': 'Full 3D volumes (128³)',
        'notes': 'Baseline variant'
    },
    'roi_mri': {
        'model_name': 'ResNet50-3D',
        'checkpoint_base': 'results/ResNet50-3D',
        'input_type': 'Full 3D volumes (128³)',
        'notes': 'Same as mri (ROI only affects MIL, not ResNet50-3D)'
    }
}


def get_output_dir(variant: str) -> Path:
    """Get output directory for a variant."""
    return PROJECT_ROOT / 'ensemble' / 'results' / 'interpretability' / variant / 'cnn_gradcam'


def find_latest_checkpoint(variant: str, fold: int = 0, use_ema: bool = True) -> Optional[Path]:
    """
    Find the latest checkpoint for a variant and fold.
    
    Args:
        variant: Variant name ('mri' or 'roi_mri')
        fold: Fold number (default: 0)
        use_ema: If True, prefer EMA checkpoint (default: True)
    
    Returns:
        Path to checkpoint file, or None if not found
    """
    if variant not in VARIANT_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Must be one of {list(VARIANT_CONFIGS.keys())}")
    
    config = VARIANT_CONFIGS[variant]
    model_name = config['model_name']
    checkpoint_base = Path(config['checkpoint_base'])
    
    # Support both absolute and relative paths
    if checkpoint_base.is_absolute():
        model_dir = checkpoint_base / 'runs' / f'fold_{fold}'
    else:
        model_dir = PROJECT_ROOT / checkpoint_base / 'runs' / f'fold_{fold}'
    if not model_dir.exists():
        return None
    
    # Find all run directories (they may be named run_* or YYYYMMDD_HHMMSS)
    run_dirs = []
    for d in model_dir.iterdir():
        if d.is_dir():
            if d.name.startswith('run_') or (len(d.name) == 15 and d.name.replace('_', '').isdigit()):
                run_dirs.append(d)
    
    if not run_dirs:
        return None
    
    # Sort by modification time (newest first)
    run_dirs = sorted(run_dirs, key=lambda x: x.stat().st_mtime, reverse=True)
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


def load_resnet50_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load ResNet50-3D model from checkpoint."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model
    model = create_resnet50_3d(num_classes=2, in_channels=4, dropout=0.4)
    
    # Load state dict
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info("✓ Model loaded successfully")
    return model


def load_patient_volume(patient_id: str, class_name: Optional[str] = None) -> Tuple[torch.Tensor, str]:
    """
    Load multi-modal volume for a patient.
    
    Args:
        patient_id: Patient ID
        class_name: Class name (LGG or HGG). If None, tries both.
    
    Returns:
        Tuple of (volume_tensor, actual_class_name)
        volume_tensor: (4, D, H, W) tensor
    """
    modalities = ['t1', 't1ce', 't2', 'flair']
    volume_channels = []
    
    # Try to find patient in either class directory
    if class_name is None:
        class_names = ['LGG', 'HGG']
    else:
        class_names = [class_name]
    
    found_class = None
    for cn in class_names:
        patient_dir = DATA_ROOT / cn / patient_id
        if patient_dir.exists():
            found_class = cn
            break
    
    if found_class is None:
        raise FileNotFoundError(f"Patient {patient_id} not found in {DATA_ROOT}")
    
    patient_dir = DATA_ROOT / found_class / patient_id
    
    # Load all modalities
    for mod in modalities:
        volume_path = patient_dir / f"{patient_id}_{mod}.nii.gz"
        if not volume_path.exists():
            volume_path = patient_dir / f"{patient_id}_{mod}.nii"
        
        if not volume_path.exists():
            raise FileNotFoundError(f"Missing modality {mod} for patient {patient_id}")
        
        try:
            volume = sitk.ReadImage(str(volume_path))
            volume_array = sitk.GetArrayFromImage(volume).astype(np.float32)
            volume_channels.append(volume_array)
        except Exception as e:
            raise RuntimeError(f"Error loading {volume_path}: {e}")
    
    # Stack modalities: (4, D, H, W)
    multi_modal_volume = np.stack(volume_channels, axis=0)
    volume_tensor = torch.from_numpy(multi_modal_volume).float()
    
    # Verify shape
    assert volume_tensor.shape == (4, 128, 128, 128), \
        f"Expected shape (4, 128, 128, 128), got {volume_tensor.shape}"
    
    return volume_tensor, found_class


def get_background_slice(volume: np.ndarray, plane: str, slice_idx: int) -> np.ndarray:
    """
    Extract a 2D slice from 3D volume for background visualization.
    
    Args:
        volume: 3D volume of shape (D, H, W) or (C, D, H, W)
        plane: 'axial', 'coronal', or 'sagittal'
        slice_idx: Slice index
    
    Returns:
        2D slice as numpy array
    """
    if volume.ndim == 4:
        # Multi-channel: use mean across channels or T1ce (index 1)
        volume = volume[1, :, :, :]  # Use T1ce for background
    
    if plane == 'axial':
        # Axial: z-axis (first dimension)
        if slice_idx >= volume.shape[0]:
            slice_idx = volume.shape[0] - 1
        return volume[slice_idx, :, :]
    elif plane == 'coronal':
        # Coronal: y-axis (second dimension)
        if slice_idx >= volume.shape[1]:
            slice_idx = volume.shape[1] - 1
        return volume[:, slice_idx, :]
    elif plane == 'sagittal':
        # Sagittal: x-axis (third dimension)
        if slice_idx >= volume.shape[2]:
            slice_idx = volume.shape[2] - 1
        return volume[:, :, slice_idx]
    else:
        raise ValueError(f"Unknown plane: {plane}")


def visualize_gradcam_overlay(
    background: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on background image.
    
    Args:
        background: 2D background image (grayscale)
        heatmap: 2D heatmap (values in [0, 1])
        alpha: Transparency of heatmap overlay (default: 0.5)
        colormap: Colormap name (default: 'jet')
    
    Returns:
        RGB image with overlay
    """
    # Normalize background to [0, 1]
    bg_min = background.min()
    bg_max = background.max()
    if bg_max > bg_min:
        background_norm = (background - bg_min) / (bg_max - bg_min)
    else:
        background_norm = np.zeros_like(background)
    
    # Convert background to RGB (grayscale)
    bg_rgb = np.stack([background_norm] * 3, axis=-1)
    
    # Apply colormap to heatmap
    cmap = cm.get_cmap(colormap)
    heatmap_rgb = cmap(heatmap)[:, :, :3]  # Remove alpha channel
    
    # Overlay: blend background and heatmap
    overlay = (1 - alpha) * bg_rgb + alpha * heatmap_rgb
    
    return overlay


def create_montage(
    volume: np.ndarray,
    heatmap: np.ndarray,
    plane: str,
    output_path: Path,
    num_slices: int = 12
):
    """
    Create montage of slices with Grad-CAM overlay.
    
    Args:
        volume: 3D volume of shape (C, D, H, W) or (D, H, W)
        heatmap: 3D heatmap of shape (D, H, W)
        plane: 'axial', 'coronal', or 'sagittal'
        num_slices: Number of slices to show
        output_path: Path to save montage
    """
    # Get volume dimensions
    if volume.ndim == 4:
        D, H, W = volume.shape[1], volume.shape[2], volume.shape[3]
    else:
        D, H, W = volume.shape
    
    # Determine slice dimension
    if plane == 'axial':
        num_slices_available = D
    elif plane == 'coronal':
        num_slices_available = H
    elif plane == 'sagittal':
        num_slices_available = W
    else:
        raise ValueError(f"Unknown plane: {plane}")
    
    # Select evenly spaced slices
    slice_indices = np.linspace(0, num_slices_available - 1, num_slices, dtype=int)
    
    # Create grid
    cols = 4
    rows = (num_slices + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, slice_idx in enumerate(slice_indices):
        ax = axes[idx]
        
        # Get background slice
        bg_slice = get_background_slice(volume, plane, slice_idx)
        
        # Get heatmap slice
        heatmap_slice = get_background_slice(heatmap, plane, slice_idx)
        
        # Create overlay
        overlay = visualize_gradcam_overlay(bg_slice, heatmap_slice, alpha=0.5)
        
        # Display
        ax.imshow(overlay)
        ax.set_title(f'{plane.capitalize()} {slice_idx}', fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(slice_indices), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved {plane} montage: {output_path}")


def create_summary_visualization(
    volume: np.ndarray,
    heatmap: np.ndarray,
    patient_id: str,
    prediction_info: dict,
    output_path: Path,
    num_slices: int = 9
):
    """
    Create summary visualization with all three planes.
    
    Args:
        volume: 3D volume of shape (C, D, H, W) or (D, H, W)
        heatmap: 3D heatmap of shape (D, H, W)
        patient_id: Patient ID
        prediction_info: Prediction information dict
        output_path: Path to save summary
        num_slices: Number of slices per plane
    """
    fig, axes = plt.subplots(3, num_slices, figsize=(3 * num_slices, 9))
    
    planes = ['axial', 'coronal', 'sagittal']
    
    for plane_idx, plane in enumerate(planes):
        # Get volume dimensions
        if volume.ndim == 4:
            D, H, W = volume.shape[1], volume.shape[2], volume.shape[3]
        else:
            D, H, W = volume.shape
        
        # Determine slice dimension
        if plane == 'axial':
            num_slices_available = D
        elif plane == 'coronal':
            num_slices_available = H
        elif plane == 'sagittal':
            num_slices_available = W
        
        # Select evenly spaced slices
        slice_indices = np.linspace(0, num_slices_available - 1, num_slices, dtype=int)
        
        for slice_idx_idx, slice_idx in enumerate(slice_indices):
            ax = axes[plane_idx, slice_idx_idx]
            
            # Get background slice
            bg_slice = get_background_slice(volume, plane, slice_idx)
            
            # Get heatmap slice
            heatmap_slice = get_background_slice(heatmap, plane, slice_idx)
            
            # Create overlay
            overlay = visualize_gradcam_overlay(bg_slice, heatmap_slice, alpha=0.5)
            
            # Display
            ax.imshow(overlay)
            if slice_idx_idx == 0:
                ax.set_ylabel(plane.capitalize(), fontsize=12, fontweight='bold')
            ax.set_title(f'{slice_idx}', fontsize=8)
            ax.axis('off')
    
    # Add title with prediction info
    pred_class = prediction_info['predicted_class_name']
    hgg_prob = prediction_info['probabilities']['HGG']
    title = f"{patient_id} | Predicted: {pred_class} (HGG prob: {hgg_prob:.3f})"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved summary visualization: {output_path}")


def save_heatmap_nifti(heatmap: np.ndarray, output_path: Path):
    """Save heatmap as NIfTI file."""
    # Create SimpleITK image from numpy array
    sitk_image = sitk.GetImageFromArray(heatmap)
    
    # Save
    sitk.WriteImage(sitk_image, str(output_path))
    logger.info(f"  ✓ Saved heatmap NIfTI: {output_path}")


def process_patient(
    patient_id: str,
    model: nn.Module,
    gradcam,
    device: torch.device,
    target_class: Optional[int],
    output_dir: Path,
    num_slices: int = 12
) -> Optional[dict]:
    """
    Process a single patient: load volume, generate Grad-CAM, save visualizations.
    
    Returns:
        Metadata dict for this patient
    """
    logger.info(f"\nProcessing patient: {patient_id}")
    
    # Load volume
    try:
        volume_tensor, class_name = load_patient_volume(patient_id)
        logger.info(f"  ✓ Loaded volume from class: {class_name}")
    except Exception as e:
        logger.error(f"  ✗ Error loading volume: {e}")
        return None
    
    # Add batch dimension
    volume_batch = volume_tensor.unsqueeze(0).to(device)  # (1, 4, D, H, W)
    
    # Generate Grad-CAM
    try:
        cam_heatmap, prediction_info = gradcam.generate_cam_with_prediction(
            volume_batch,
            target_class=target_class
        )
        logger.info(f"  ✓ Generated Grad-CAM heatmap")
        logger.info(f"    Predicted: {prediction_info['predicted_class_name']} "
                   f"(HGG prob: {prediction_info['probabilities']['HGG']:.3f})")
        logger.info(f"    Target class for CAM: {prediction_info['target_class_name']}")
    except Exception as e:
        logger.error(f"  ✗ Error generating Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Create patient output directory
    patient_dir = output_dir / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    
    # Save heatmap as NIfTI
    heatmap_nifti_path = patient_dir / 'gradcam_3d.nii.gz'
    save_heatmap_nifti(cam_heatmap, heatmap_nifti_path)
    
    # Convert volume to numpy for visualization
    volume_np = volume_tensor.cpu().numpy()  # (4, D, H, W)
    
    # Create montages for each plane
    for plane in ['axial', 'coronal', 'sagittal']:
        montage_path = patient_dir / f'{plane}_overlay.png'
        create_montage(volume_np, cam_heatmap, plane, montage_path, num_slices)
    
    # Create summary visualization
    summary_path = patient_dir / 'summary.png'
    create_summary_visualization(
        volume_np, cam_heatmap, patient_id, prediction_info,
        summary_path, num_slices=9
    )
    
    # Save metadata
    metadata = {
        'patient_id': patient_id,
        'class': class_name,
        **prediction_info
    }
    metadata_path = patient_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  ✓ Saved metadata: {metadata_path}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Generate 3D Grad-CAM heatmaps for ResNet50-3D',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--model_variant',
        type=str,
        default='mri',
        choices=['mri', 'roi_mri'],
        help='Model variant: "mri" (baseline) or "roi_mri" (ROI ensemble context) (default: mri)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='AUTO',
        help='Checkpoint path or "AUTO" to find latest (default: AUTO)'
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        help='Fold number for AUTO checkpoint (default: 0)'
    )
    parser.add_argument(
        '--patient_ids',
        type=str,
        nargs='+',
        help='List of patient IDs (e.g., Brats18_TCIA10_103_1 Brats18_TCIA10_104_1)'
    )
    parser.add_argument(
        '--patient_ids_file',
        type=str,
        help='Path to text file with patient IDs (one per line)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: auto-generated based on variant)'
    )
    parser.add_argument(
        '--target_class',
        type=str,
        default='pred',
        choices=['pred', '0', '1'],
        help='Target class for Grad-CAM: "pred" (predicted), "0" (LGG), or "1" (HGG) (default: pred)'
    )
    parser.add_argument(
        '--num_slices',
        type=int,
        default=12,
        help='Number of slices per montage (default: 12)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available, else cpu)'
    )
    
    args = parser.parse_args()
    
    # Determine device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Get variant config
    variant = args.model_variant
    if variant not in VARIANT_CONFIGS:
        logger.error(f"Unknown variant: {variant}. Must be one of {list(VARIANT_CONFIGS.keys())}")
        return 1
    
    config = VARIANT_CONFIGS[variant]
    logger.info(f"\n{'='*80}")
    logger.info(f"MODEL VARIANT: {variant}")
    logger.info(f"{'='*80}")
    logger.info(f"Model name: {config['model_name']}")
    logger.info(f"Input type: {config['input_type']}")
    logger.info(f"Notes: {config['notes']}")
    
    # Load checkpoint
    if args.checkpoint.upper() == 'AUTO':
        checkpoint_path = find_latest_checkpoint(variant, fold=args.fold, use_ema=True)
        if checkpoint_path is None:
            logger.error(f"Could not find checkpoint for variant {variant}, fold {args.fold}")
            logger.error("Please specify --checkpoint manually or ensure model is trained.")
            return 1
        logger.info(f"Found checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return 1
    
    # Load model
    model = load_resnet50_model(checkpoint_path, device)
    
    # Create Grad-CAM
    gradcam = create_gradcam_for_resnet50_3d(model, device)
    
    # Parse target class
    if args.target_class == 'pred':
        target_class = None
    else:
        target_class = int(args.target_class)
    
    # Get patient IDs
    patient_ids = []
    if args.patient_ids:
        patient_ids = args.patient_ids
    elif args.patient_ids_file:
        patient_ids_file = Path(args.patient_ids_file)
        if not patient_ids_file.exists():
            logger.error(f"Patient IDs file not found: {patient_ids_file}")
            return 1
        with open(patient_ids_file, 'r') as f:
            patient_ids = [line.strip() for line in f if line.strip()]
    else:
        logger.error("Must specify either --patient_ids or --patient_ids_file")
        return 1
    
    if not patient_ids:
        logger.error("No patient IDs provided")
        return 1
    
    logger.info(f"Processing {len(patient_ids)} patients")
    
    # Create output directory
    if args.output_dir is None:
        output_dir = get_output_dir(variant)
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Save checkpoint info
    checkpoint_info = {
        'variant': variant,
        'checkpoint_path': str(checkpoint_path.resolve()),
        'fold': args.fold,
        'target_class': args.target_class,
        'num_patients': len(patient_ids),
        'model_name': config['model_name'],
        'input_type': config['input_type'],
        'input_shape': '(1, 4, 128, 128, 128)',
        'notes': config['notes']
    }
    with open(output_dir / 'checkpoint_info.json', 'w') as f:
        json.dump(checkpoint_info, f, indent=2)
    
    # Process each patient
    all_metadata = []
    for patient_id in patient_ids:
        try:
            metadata = process_patient(
                patient_id, model, gradcam, device,
                target_class, output_dir, args.num_slices
            )
            if metadata:
                all_metadata.append(metadata)
        except Exception as e:
            logger.error(f"Failed to process {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    summary = {
        'checkpoint_info': checkpoint_info,
        'patients': all_metadata
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing complete!")
    logger.info(f"Processed {len(all_metadata)}/{len(patient_ids)} patients")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*80}")
    
    # Print example output paths
    if all_metadata:
        example_patient = all_metadata[0]['patient_id']
        logger.info(f"\nExample output paths for {example_patient}:")
        logger.info(f"  - Heatmap NIfTI: {output_dir / example_patient / 'gradcam_3d.nii.gz'}")
        logger.info(f"  - Axial montage: {output_dir / example_patient / 'axial_overlay.png'}")
        logger.info(f"  - Coronal montage: {output_dir / example_patient / 'coronal_overlay.png'}")
        logger.info(f"  - Sagittal montage: {output_dir / example_patient / 'sagittal_overlay.png'}")
        logger.info(f"  - Summary: {output_dir / example_patient / 'summary.png'}")
        logger.info(f"  - Metadata: {output_dir / example_patient / 'metadata.json'}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

