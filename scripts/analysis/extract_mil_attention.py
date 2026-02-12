#!/usr/bin/env python3
"""
Extract MIL Attention Weights for Baseline and ROI Variants

This script extracts slice-level attention weights from DualStreamMIL models
for both baseline and ROI variants, then compares their spatial behavior.

Author: Medical Imaging Pipeline
"""

import sys
from pathlib import Path

# Set project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Project imports
from models.dual_stream_mil import create_dual_stream_mil
from utils.dataset_mil import MILSliceDataset
from utils.dataset_mil_roi import MILSliceDatasetROI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_ROOT = PROJECT_ROOT / 'data' / 'processed' / 'stage_4_resize' / 'train'
RESULTS_DIR = PROJECT_ROOT / 'results'
SEG_DATA_ROOT = PROJECT_ROOT / 'data' / 'raw' / 'BraTS2018'  # For segmentation masks
SPLITS_DIR = PROJECT_ROOT / 'splits'
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'results' / 'interpretability' / 'mil_attention'


def find_mil_checkpoint(variant: str, fold: int = 0, use_ema: bool = True) -> Tuple[Optional[Path], Dict]:
    """
    Find MIL checkpoint for a variant.
    
    Args:
        variant: 'baseline' or 'roi'
        fold: Fold number
        use_ema: Prefer EMA checkpoint
    
    Returns:
        Tuple of (checkpoint_path, run_info_dict) or (None, {}) if not found
    """
    # Search in multiple locations
    search_dirs = [
        RESULTS_DIR / 'DualStreamMIL-3D' / 'runs' / f'fold_{fold}',
        PROJECT_ROOT / 'runs' / 'mil_roi_sanity' / 'runs' / f'fold_{fold}',
    ]
    
    # Find all run directories
    all_run_dirs = []
    for model_dir in search_dirs:
        if model_dir.exists():
            for d in model_dir.iterdir():
                if d.is_dir() and (d.name.startswith('run_') or (len(d.name) == 15 and d.name.replace('_', '').isdigit())):
                    all_run_dirs.append(d)
    
    if not all_run_dirs:
        logger.error(f"No run directories found in search paths for fold {fold}")
        return None, {}
    
    # Scan all runs and build detection table
    run_info_list = []
    matching_runs = []
    roi_indicators = ['roi', 'use_roi', 'roi_sampling', 'dataset_roi', 'MILSliceDatasetROI']
    
    for run_dir in all_run_dirs:
        config_file = run_dir / 'config.json'
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                sampling = config.get('sampling_strategy', 'unknown')
                bag_size = config.get('bag_size', 'unknown')
                
                # Check for ROI indicators in config or path
                config_str = json.dumps(config).lower()
                path_str = str(run_dir).lower()
                is_roi = any(ind.lower() in config_str for ind in roi_indicators) or 'roi' in path_str
                
                run_info = {
                    'run_name': run_dir.name,
                    'fold': fold,
                    'sampling_strategy': sampling,
                    'bag_size': bag_size,
                    'path': str(run_dir),
                    'is_roi': is_roi,
                    'config': config  # Store full config for later use
                }
                run_info_list.append(run_info)
                
                # Check if this matches the requested variant
                if variant == 'baseline' and not is_roi and sampling in ['random', 'entropy', 'sequential', 'hybrid']:
                    matching_runs.append((run_dir, run_info))
                elif variant == 'roi' and is_roi:
                    matching_runs.append((run_dir, run_info))
            except Exception as e:
                logger.debug(f"Error reading config {config_file}: {e}")
                continue
    
    # Print detection table
    logger.info(f"\n{'='*80}")
    logger.info(f"MIL CHECKPOINT DETECTION TABLE (Fold {fold})")
    logger.info(f"{'='*80}")
    logger.info(f"{'Path':<60} {'Fold':<6} {'Sampling':<15} {'Bag Size':<10} {'ROI':<6}")
    logger.info(f"{'-'*80}")
    for info in run_info_list:
        roi_str = "Yes" if info['is_roi'] else "No"
        logger.info(f"{info['path']:<60} {info['fold']:<6} {info['sampling_strategy']:<15} {str(info['bag_size']):<10} {roi_str:<6}")
    logger.info(f"{'='*80}\n")
    
    if not matching_runs:
        logger.error(f"No runs found for variant '{variant}' with fold {fold}")
        logger.error(f"Available sampling strategies: {set(r['sampling_strategy'] for r in run_info_list)}")
        logger.error(f"ROI runs found: {sum(1 for r in run_info_list if r['is_roi'])}")
        return None, {'runs_scanned': run_info_list}
    
    # Sort by modification time (newest first)
    matching_runs = sorted(matching_runs, key=lambda x: x[0].stat().st_mtime, reverse=True)
    latest_run, run_info = matching_runs[0]
    
    logger.info(f"Selected run for {variant}: {latest_run.name}")
    logger.info(f"  Path: {run_info['path']}")
    logger.info(f"  Sampling strategy: {run_info['sampling_strategy']}")
    logger.info(f"  Bag size: {run_info['bag_size']}")
    logger.info(f"  Is ROI: {run_info['is_roi']}")
    
    checkpoint_dir = latest_run / 'checkpoints'
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return None, run_info
    
    # Prefer EMA checkpoint
    if use_ema:
        ema_checkpoint = checkpoint_dir / 'best_ema.pt'
        if ema_checkpoint.exists():
            logger.info(f"Using EMA checkpoint: {ema_checkpoint.name}")
            return ema_checkpoint, run_info
    
    # Fall back to regular checkpoint
    regular_checkpoint = checkpoint_dir / 'best.pt'
    if regular_checkpoint.exists():
        logger.info(f"Using regular checkpoint: {regular_checkpoint.name}")
        return regular_checkpoint, run_info
    
    logger.error(f"No checkpoint found in {checkpoint_dir}")
    return None, run_info


def load_mil_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load DualStreamMIL model from checkpoint."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    
    # Create model
    model = create_dual_stream_mil(
        num_classes=2,
        instance_encoder_backbone=config.get('instance_encoder_backbone', 'resnet18'),
        instance_encoder_input_size=config.get('instance_encoder_input_size', 224),
        attention_type=config.get('attention_type', 'gated'),
        fusion_method=config.get('fusion_method', 'concat'),
        dropout=config.get('dropout', 0.5),
        use_hidden_layer=config.get('use_hidden_layer', True)
    )
    
    # Load state dict
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    logger.info("✓ Model loaded successfully")
    return model


def load_patient_bag(patient_id: str, variant: str, bag_size: int = 64, sampling_strategy: str = 'sequential') -> Tuple[torch.Tensor, str, Optional[np.ndarray]]:
    """
    Load patient bag of slices.
    
    Args:
        patient_id: Patient ID
        variant: 'baseline' or 'roi'
        bag_size: Number of slices per bag (from config)
        sampling_strategy: Sampling strategy from config (e.g., 'entropy', 'random', 'sequential')
    
    Returns:
        Tuple of (bag_tensor, class_name, slice_indices)
        bag_tensor: (N, 4, H, W) where N=bag_size
        slice_indices: Original z-coordinates of slices (if available)
    """
    # Find patient class
    found_class = None
    for class_name in ['LGG', 'HGG']:
        patient_dir = DATA_ROOT / class_name / patient_id
        if patient_dir.exists():
            found_class = class_name
            break
    
    if found_class is None:
        raise FileNotFoundError(f"Patient {patient_id} not found")
    
    # Create a temporary split file with just this patient
    import tempfile
    import pandas as pd
    
    temp_split = pd.DataFrame({
        'patient_id': [patient_id],
        'class': [found_class],
        'label': [1 if found_class == 'HGG' else 0]
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_split.to_csv(f, index=False)
        temp_split_file = Path(f.name)
    
    try:
        # Load bag based on variant - use sampling_strategy from config
        # For inference, we want deterministic results, but we should match training config
        # If training used 'entropy', we can still use it for inference (deterministic if entropy files exist)
        inference_sampling = sampling_strategy if sampling_strategy in ['sequential', 'entropy'] else 'sequential'
        
        if variant == 'roi':
            # Use ROI dataset
            dataset = MILSliceDatasetROI(
                data_root=DATA_ROOT,
                split_file=temp_split_file,
                bag_size=bag_size,
                sampling_strategy=inference_sampling,  # Use config strategy
                seg_data_root=SEG_DATA_ROOT,
                roi_tumor_ratio=0.7
            )
        else:
            # Use baseline dataset
            dataset = MILSliceDataset(
                data_root=DATA_ROOT,
                split_file=temp_split_file,
                bag_size=bag_size,
                sampling_strategy=inference_sampling  # Use config strategy
            )
        
        # Load bag (patient is at index 0)
        bag, label, _ = dataset[0]
        
        # Get slice indices based on sampling strategy
        volume_path = DATA_ROOT / found_class / patient_id / f"{patient_id}_t1.nii.gz"
        if not volume_path.exists():
            volume_path = DATA_ROOT / found_class / patient_id / f"{patient_id}_t1.nii"
        
        slice_indices = None
        if volume_path.exists():
            import SimpleITK as sitk
            volume = sitk.ReadImage(str(volume_path))
            volume_array = sitk.GetArrayFromImage(volume)
            depth = volume_array.shape[0]
            
            if inference_sampling == 'sequential':
                # Sequential sampling: evenly spaced indices
                step = depth / bag_size
                slice_indices = np.array([int(i * step) for i in range(bag_size)])
            elif inference_sampling == 'entropy':
                # For entropy, try to get actual slice indices from dataset if available
                # Otherwise fall back to sequential
                try:
                    # Check if dataset has slice_indices attribute
                    if hasattr(dataset, 'slice_indices') and len(dataset.slice_indices) > 0:
                        slice_indices = np.array(dataset.slice_indices[0])  # First patient
                    else:
                        # Fall back to sequential
                        step = depth / bag_size
                        slice_indices = np.array([int(i * step) for i in range(bag_size)])
                except:
                    step = depth / bag_size
                    slice_indices = np.array([int(i * step) for i in range(bag_size)])
            else:
                # For random or other strategies, use sequential for reproducibility
                step = depth / bag_size
                slice_indices = np.array([int(i * step) for i in range(bag_size)])
        
    finally:
        # Clean up temp file
        if temp_split_file.exists():
            temp_split_file.unlink()
    
    return bag, found_class, slice_indices


def extract_attention(
    model: nn.Module,
    bag: torch.Tensor,
    device: torch.device
) -> Dict:
    """
    Extract attention weights for a patient bag.
    
    Returns:
        Dict with attention_weights, selection_weights, critical_idx, etc.
    """
    model.eval()
    bag = bag.unsqueeze(0).to(device)  # Add batch dimension: (1, N, 4, H, W)
    
    with torch.no_grad():
        logits, interpretability = model(bag, return_interpretability=True)
        probs = torch.softmax(logits, dim=1)
    
    # Convert to numpy
    attention_weights = interpretability['attention_weights'].cpu().numpy()[0]  # (N,)
    selection_weights = interpretability['selection_weights'].cpu().numpy()[0]  # (N,)
    critical_idx = int(interpretability['critical_idx'][0].item())
    predicted_class = int(torch.argmax(logits, dim=1)[0].item())
    hgg_prob = float(probs[0, 1].item())
    
    return {
        'attention_weights': attention_weights,
        'selection_weights': selection_weights,
        'critical_idx': critical_idx,
        'predicted_class': predicted_class,
        'hgg_prob': hgg_prob
    }


def convert_to_serializable(obj: Any) -> Any:
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
    
    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (bool, str, type(None))):
        return obj
    elif isinstance(obj, (int, float)):
        return obj
    else:
        # Try to convert unknown types
        try:
            return float(obj) if isinstance(obj, (np.number, np.bool_)) else str(obj)
        except (TypeError, ValueError):
            return str(obj)


def load_tumor_mask(patient_id: str, class_name: str) -> Optional[np.ndarray]:
    """
    Load tumor segmentation mask.
    
    Returns:
        3D mask array (D, H, W) or None if not found
    """
    # Try multiple possible locations
    possible_paths = [
        SEG_DATA_ROOT / class_name / patient_id / f'{patient_id}_seg.nii.gz',
        SEG_DATA_ROOT / class_name / patient_id / f'{patient_id}_seg.nii',
        DATA_ROOT.parent.parent / 'raw' / 'BraTS2018' / class_name / patient_id / f'{patient_id}_seg.nii.gz',
    ]
    
    for seg_path in possible_paths:
        if seg_path.exists():
            try:
                seg_image = sitk.ReadImage(str(seg_path))
                seg_array = sitk.GetArrayFromImage(seg_image).astype(np.uint8)
                # Tumor mask: values > 0 (combines all tumor labels)
                tumor_mask = (seg_array > 0).astype(np.uint8)
                return tumor_mask
            except Exception as e:
                logger.debug(f"Error loading mask from {seg_path}: {e}")
                continue
    
    return None


def compute_tumor_overlap(
    attention_weights: np.ndarray,
    slice_indices: Optional[np.ndarray],
    tumor_mask: Optional[np.ndarray],
    top_k: int = 10
) -> Dict:
    """
    Compute overlap between high-attention slices and tumor mask.
    
    Returns:
        Dict with overlap statistics
    """
    if tumor_mask is None or slice_indices is None:
        return {
            'tumor_overlap_available': False,
            'top_k': top_k,
            'top_k_slice_indices': None,
            'tumor_slices': None,
            'top_k_tumor_slices': None,
            'top_k_overlap_ratio': None
        }
    
    # Get top-k attention slices
    top_k_indices = np.argsort(attention_weights)[-top_k:][::-1]
    top_k_slice_indices = slice_indices[top_k_indices]
    
    # Check which slices contain tumor
    tumor_slices = []
    for z_idx in top_k_slice_indices:
        if 0 <= z_idx < tumor_mask.shape[0]:
            slice_mask = tumor_mask[z_idx, :, :]
            if np.any(slice_mask > 0):
                tumor_slices.append(int(z_idx))
    
    overlap_ratio = len(tumor_slices) / top_k if top_k > 0 else 0.0
    
    return {
        'tumor_overlap_available': True,
        'top_k': top_k,
        'top_k_indices': top_k_indices.tolist(),
        'top_k_slice_indices': [int(x) for x in top_k_slice_indices.tolist()],
        'tumor_slices': tumor_slices,
        'top_k_tumor_slices': len(tumor_slices),
        'top_k_overlap_ratio': float(overlap_ratio)
    }


def create_attention_visualizations(
    attention_df: pd.DataFrame,
    patient_id: str,
    variant: str,
    output_dir: Path,
    top_k: int = 5
) -> None:
    """
    Create attention visualization plots.
    
    Args:
        attention_df: DataFrame with columns: slice_index, attention_weight, rank, has_tumor
        patient_id: Patient ID
        variant: Variant name
        output_dir: Output directory
        top_k: Number of top slices to highlight
    """
    patient_dir = output_dir / variant / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Attention bar plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    slice_indices = attention_df['slice_index'].values
    attention_weights = attention_df['attention_weight'].values
    has_tumor = attention_df['has_tumor'].values if 'has_tumor' in attention_df.columns else None
    
    # Get top-k indices
    top_k_indices = np.argsort(attention_weights)[-top_k:][::-1]
    top_k_slice_idx = slice_indices[top_k_indices]
    
    # Plot all bars
    colors = ['red' if idx in top_k_slice_idx else 'steelblue' for idx in slice_indices]
    ax.bar(slice_indices, attention_weights, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Mark tumor slices if available
    if has_tumor is not None and np.any(has_tumor):
        tumor_indices = slice_indices[has_tumor]
        for idx in tumor_indices:
            ax.axvline(x=idx, color='green', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Slice Index', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title(f'MIL Attention Distribution - {patient_id} ({variant})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    red_patch = mpatches.Patch(color='red', label=f'Top-{top_k} slices')
    blue_patch = mpatches.Patch(color='steelblue', label='Other slices')
    legend_items = [red_patch, blue_patch]
    if has_tumor is not None and np.any(has_tumor):
        green_line = mpatches.Patch(color='green', linestyle='--', label='Tumor slices')
        legend_items.append(green_line)
    ax.legend(handles=legend_items, loc='upper right')
    
    plt.tight_layout()
    plot_path = patient_dir / 'attention_plot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved attention plot: {plot_path}")
    
    # Plot 2: Attention vs Tumor overlay
    if has_tumor is not None and np.any(has_tumor):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Top: Attention weights
        ax1.bar(slice_indices, attention_weights, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_ylabel('Attention Weight', fontsize=12)
        ax1.set_title(f'Attention vs Tumor Overlay - {patient_id} ({variant})', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Highlight top-k
        for idx in top_k_slice_idx:
            ax1.axvline(x=idx, color='red', linestyle='-', alpha=0.7, linewidth=2)
        
        # Bottom: Tumor presence
        tumor_binary = has_tumor.astype(int)
        ax2.bar(slice_indices, tumor_binary, color='green', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Slice Index', fontsize=12)
        ax2.set_ylabel('Tumor Present', fontsize=12)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['No', 'Yes'])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Highlight top-k
        for idx in top_k_slice_idx:
            ax2.axvline(x=idx, color='red', linestyle='-', alpha=0.7, linewidth=2)
        
        plt.tight_layout()
        overlay_path = patient_dir / 'attention_vs_tumor.png'
        plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"  ✓ Saved attention vs tumor overlay: {overlay_path}")


def process_patient(
    patient_id: str,
    variant: str,
    model: nn.Module,
    device: torch.device,
    output_dir: Path,
    bag_size: int = 64,
    run_info: Optional[Dict] = None
) -> Optional[Dict]:
    """Process a single patient and extract attention."""
    logger.info(f"\nProcessing {patient_id} ({variant})...")
    
    try:
        # Get config from run_info
        if run_info and run_info.get('config'):
            checkpoint_config = run_info['config']
            actual_bag_size = checkpoint_config.get('bag_size', bag_size)
            actual_sampling = checkpoint_config.get('sampling_strategy', 'sequential')
            logger.info(f"  Using config from checkpoint: bag_size={actual_bag_size}, sampling={actual_sampling}")
        else:
            actual_bag_size = bag_size
            actual_sampling = 'sequential'
            logger.info(f"  Using default: bag_size={actual_bag_size}, sampling={actual_sampling}")
        
        # Load bag with config values
        bag, class_name, slice_indices = load_patient_bag(patient_id, variant, actual_bag_size, actual_sampling)
        logger.info(f"  ✓ Loaded bag: shape {bag.shape}, class: {class_name}")
        
        # Extract attention
        attention_info = extract_attention(model, bag, device)
        logger.info(f"  ✓ Extracted attention: predicted={attention_info['predicted_class']}, "
                   f"hgg_prob={attention_info['hgg_prob']:.3f}")
        
        # Load tumor mask if available
        tumor_mask = load_tumor_mask(patient_id, class_name)
        if tumor_mask is not None:
            logger.info(f"  ✓ Loaded tumor mask: shape {tumor_mask.shape}")
        else:
            logger.info(f"  ⚠ Tumor mask not found")
        
        # Compute overlap
        overlap_info = compute_tumor_overlap(
            attention_info['attention_weights'],
            slice_indices,
            tumor_mask,
            top_k=10
        )
        
        # Save per-slice attention CSV
        patient_dir = output_dir / variant / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        # Create attention dataframe
        N = len(attention_info['attention_weights'])
        attention_df = pd.DataFrame({
            'slice_index': range(N),
            'attention_weight': attention_info['attention_weights'],
            'selection_weight': attention_info['selection_weights'],
            'rank': np.argsort(attention_info['attention_weights'])[::-1] + 1  # Rank 1 = highest
        })
        
        # Add slice z-coordinates if available
        if slice_indices is not None:
            attention_df['z_coordinate'] = slice_indices
            attention_df['has_tumor'] = [
                np.any(tumor_mask[z, :, :] > 0) if tumor_mask is not None and 0 <= z < tumor_mask.shape[0] else False
                for z in slice_indices
            ]
        else:
            attention_df['z_coordinate'] = None
            attention_df['has_tumor'] = None
        
        # Save CSV
        csv_path = patient_dir / 'attention_weights.csv'
        attention_df.to_csv(csv_path, index=False)
        logger.info(f"  ✓ Saved attention CSV: {csv_path}")
        
        # Create visualizations
        create_attention_visualizations(attention_df, patient_id, variant, output_dir, top_k=5)
        
        # Save metadata (convert all numpy types to native Python types)
        metadata = {
            'patient_id': patient_id,
            'variant': variant,
            'class': class_name,
            'predicted_class': int(attention_info['predicted_class']),
            'predicted_class_name': 'HGG' if attention_info['predicted_class'] == 1 else 'LGG',
            'hgg_prob': float(attention_info['hgg_prob']),
            'critical_slice_idx': int(attention_info['critical_idx']),
            'bag_size': int(N),
            'overlap_info': convert_to_serializable(overlap_info),
            'top_k_indices': overlap_info.get('top_k_indices', []),
            'top_k_overlap_ratio': overlap_info.get('top_k_overlap_ratio'),
            'attention_stats': {
                'mean': float(np.mean(attention_info['attention_weights'])),
                'std': float(np.std(attention_info['attention_weights'])),
                'max': float(np.max(attention_info['attention_weights'])),
                'min': float(np.min(attention_info['attention_weights'])),
                'entropy': float(-np.sum(attention_info['attention_weights'] * np.log(attention_info['attention_weights'] + 1e-10)))
            }
        }
        
        # Convert to fully serializable format
        metadata = convert_to_serializable(metadata)
        
        metadata_path = patient_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"  ✓ Saved metadata: {metadata_path}")
        
        return metadata
        
    except Exception as e:
        logger.error(f"  ✗ Error processing {patient_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Extract MIL attention weights for baseline and ROI variants',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--variant',
        type=str,
        choices=['baseline', 'roi', 'both'],
        default='both',
        help='Variant to process: baseline, roi, or both (default: both)'
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=0,
        help='Fold number for checkpoint selection (default: 0)'
    )
    parser.add_argument(
        '--patient_ids_file',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'selected_patients.txt'),
        help='Path to file with patient IDs (default: data/selected_patients.txt)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(OUTPUT_DIR),
        help=f'Output directory (default: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--bag_size',
        type=int,
        default=64,
        help='Bag size (number of slices, default: 64)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: cuda if available, else cpu)'
    )
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load patient IDs
    patient_ids_file = Path(args.patient_ids_file)
    if not patient_ids_file.exists():
        logger.error(f"Patient IDs file not found: {patient_ids_file}")
        return 1
    
    with open(patient_ids_file, 'r') as f:
        patient_ids = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(patient_ids)} patient IDs")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process variants
    variants_to_process = ['baseline', 'roi'] if args.variant == 'both' else [args.variant]
    
    all_results = {}
    
    for variant in variants_to_process:
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING VARIANT: {variant.upper()}")
        logger.info(f"{'='*80}")
        
        # Find checkpoint
        checkpoint_path, run_info = find_mil_checkpoint(variant, fold=args.fold, use_ema=True)
        if checkpoint_path is None:
            logger.error(f"Could not find checkpoint for {variant} variant, fold {args.fold}")
            if run_info:
                logger.error(f"Scanned {len(run_info.get('runs_scanned', []))} runs but none matched")
            continue
        
        logger.info(f"Found checkpoint: {checkpoint_path}")
        
        # Load model
        model = load_mil_model(checkpoint_path, device)
        
        # Process patients
        variant_results = []
        for patient_id in patient_ids:
            metadata = process_patient(
                patient_id, variant, model, device,
                output_dir, args.bag_size, run_info
            )
            if metadata:
                variant_results.append(metadata)
        
        all_results[variant] = variant_results
        logger.info(f"\n✓ Processed {len(variant_results)}/{len(patient_ids)} patients for {variant}")
    
    # Save summary
    summary = {
        'variants_processed': variants_to_process,
        'total_patients': len(patient_ids),
        'results': all_results
    }
    
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n✓ Saved summary: {summary_path}")
    
    logger.info(f"\n{'='*80}")
    logger.info("ATTENTION EXTRACTION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Output directory: {output_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

