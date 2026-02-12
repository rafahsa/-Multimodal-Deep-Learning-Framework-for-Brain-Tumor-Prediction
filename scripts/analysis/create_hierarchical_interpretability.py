#!/usr/bin/env python3
"""
Create Hierarchical Interpretability Visualization

Combines CNN Grad-CAM, MIL attention, and ROI mask for multi-level interpretability.

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Optional, Tuple
import SimpleITK as sitk
from scipy import ndimage

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_ROOT = PROJECT_ROOT / 'data' / 'processed' / 'stage_4_resize' / 'train'
SEG_DATA_ROOT = PROJECT_ROOT / 'data' / 'raw' / 'BraTS2018'
CNN_GRADCAM_DIR = PROJECT_ROOT / 'ensemble' / 'results' / 'interpretability' / 'cnn_gradcam'
MIL_ATTENTION_DIR = PROJECT_ROOT / 'ensemble' / 'results' / 'interpretability' / 'mil_attention'
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'results' / 'interpretability' / 'hierarchical'


def load_tumor_mask(patient_id: str, class_name: str) -> Optional[np.ndarray]:
    """Load tumor segmentation mask."""
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
                tumor_mask = (seg_array > 0).astype(np.uint8)
                return tumor_mask
            except Exception as e:
                logger.debug(f"Error loading mask: {e}")
                continue
    
    return None


def get_tumor_slice_range(tumor_mask: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    """Get min and max slice indices that contain tumor."""
    if tumor_mask is None:
        return None, None
    
    tumor_slices = []
    for z in range(tumor_mask.shape[0]):
        if np.any(tumor_mask[z, :, :] > 0):
            tumor_slices.append(z)
    
    if not tumor_slices:
        return None, None
    
    return min(tumor_slices), max(tumor_slices)


def load_cnn_gradcam(patient_id: str, variant: Optional[str] = None, gradcam_dir: Optional[Path] = None) -> Optional[Dict]:
    """
    Load CNN Grad-CAM results.
    
    Search order:
    1. CNN_GRADCAM_DIR / patient_id (direct, no variant subfolder)
    2. CNN_GRADCAM_DIR / variant / patient_id (if variant provided)
    3. Custom gradcam_dir / patient_id (if provided)
    """
    # Try multiple paths in order
    search_paths = []
    
    if gradcam_dir is not None:
        # Custom directory provided
        search_paths.append(Path(gradcam_dir) / patient_id)
    else:
        # First: try direct path (no variant subfolder)
        search_paths.append(CNN_GRADCAM_DIR / patient_id)
        
        # Second: try with variant subfolder (if variant provided)
        if variant:
            search_paths.append(CNN_GRADCAM_DIR / variant / patient_id)
    
    # Try each path
    found_dir = None
    for search_path in search_paths:
        if search_path.exists():
            found_dir = search_path
            logger.info(f"  ✓ Found CNN Grad-CAM directory: {found_dir}")
            break
    
    if found_dir is None:
        logger.warning(f"  ✗ CNN Grad-CAM directory not found for {patient_id}")
        logger.warning(f"    Searched paths:")
        for sp in search_paths:
            logger.warning(f"      - {sp}")
        return None
    
    # Load metadata if available
    metadata_file = found_dir / 'metadata.json'
    metadata = {}
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.debug(f"Could not load metadata: {e}")
    
    # Load heatmap
    heatmap_file = found_dir / 'gradcam_3d.nii.gz'
    if not heatmap_file.exists():
        logger.warning(f"  ✗ CNN Grad-CAM heatmap not found: {heatmap_file}")
        return None
    
    try:
        heatmap_image = sitk.ReadImage(str(heatmap_file))
        heatmap_array = sitk.GetArrayFromImage(heatmap_image)
        metadata['heatmap'] = heatmap_array
        logger.info(f"  ✓ Loaded CNN Grad-CAM heatmap: shape {heatmap_array.shape}")
        return metadata
    except Exception as e:
        logger.error(f"  ✗ Error loading CNN Grad-CAM heatmap for {patient_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_cnn_peak_slice(gradcam_data: Dict) -> Optional[int]:
    """Get slice index with maximum CAM activation."""
    if gradcam_data is None or gradcam_data.get('heatmap') is None:
        return None
    
    heatmap = gradcam_data['heatmap']
    # Sum over H and W dimensions to get per-slice activation
    slice_activations = np.sum(heatmap, axis=(1, 2))
    peak_slice = int(np.argmax(slice_activations))
    return peak_slice


def compute_cnn_tumor_overlap(
    gradcam_data: Dict,
    tumor_mask: Optional[np.ndarray],
    threshold_percentile: float = 90.0
) -> Dict:
    """
    Compute overlap between high CAM values and tumor mask.
    
    Args:
        gradcam_data: Dict with 'heatmap' key containing 3D CAM array
        tumor_mask: 3D tumor mask array (D, H, W)
        threshold_percentile: Percentile to use for thresholding CAM (default: 90)
    
    Returns:
        Dict with overlap statistics
    """
    if gradcam_data is None or gradcam_data.get('heatmap') is None:
        return {
            'cnn_overlap_available': False,
            'cnn_roi_overlap_ratio': None,
            'high_cam_voxels': None,
            'overlap_voxels': None
        }
    
    if tumor_mask is None:
        return {
            'cnn_overlap_available': False,
            'cnn_roi_overlap_ratio': None,
            'high_cam_voxels': None,
            'overlap_voxels': None
        }
    
    heatmap = gradcam_data['heatmap']
    
    # Ensure shapes match (resize if needed)
    if heatmap.shape != tumor_mask.shape:
        logger.warning(f"Shape mismatch: heatmap {heatmap.shape} vs mask {tumor_mask.shape}, resizing...")
        # Resize heatmap to match tumor_mask
        zoom_factors = [tumor_mask.shape[i] / heatmap.shape[i] for i in range(3)]
        heatmap = ndimage.zoom(heatmap, zoom_factors, order=1)
    
    # Threshold CAM: use top percentile
    threshold = np.percentile(heatmap, threshold_percentile)
    high_cam_mask = (heatmap >= threshold).astype(np.uint8)
    
    # Compute overlap
    overlap_mask = (high_cam_mask & tumor_mask).astype(np.uint8)
    
    high_cam_voxels = int(np.sum(high_cam_mask))
    overlap_voxels = int(np.sum(overlap_mask))
    
    if high_cam_voxels > 0:
        overlap_ratio = overlap_voxels / high_cam_voxels
    else:
        overlap_ratio = 0.0
    
    return {
        'cnn_overlap_available': True,
        'cnn_roi_overlap_ratio': float(overlap_ratio),
        'high_cam_voxels': high_cam_voxels,
        'overlap_voxels': overlap_voxels,
        'threshold': float(threshold),
        'threshold_percentile': threshold_percentile
    }


def load_mil_attention(patient_id: str, variant: str) -> Optional[Dict]:
    """Load MIL attention results."""
    attention_dir = MIL_ATTENTION_DIR / variant / patient_id
    if not attention_dir.exists():
        return None
    
    metadata_file = attention_dir / 'metadata.json'
    if not metadata_file.exists():
        return None
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Load attention CSV
        csv_file = attention_dir / 'attention_weights.csv'
        if csv_file.exists():
            attention_df = pd.read_csv(csv_file)
            metadata['attention_df'] = attention_df
        else:
            metadata['attention_df'] = None
        
        return metadata
    except Exception as e:
        logger.error(f"Error loading MIL attention for {patient_id}: {e}")
        return None


def compute_overlap_ratios(
    mil_top_k_slices: list,
    cnn_overlap_info: Dict,
    roi_slice_range: Tuple[Optional[int], Optional[int]]
) -> Dict:
    """Compute overlap ratios between MIL, CNN, and ROI."""
    # MIL overlap: % of top-k slices within ROI range
    mil_overlap = None
    if roi_slice_range[0] is not None and roi_slice_range[1] is not None:
        roi_min, roi_max = roi_slice_range
        if mil_top_k_slices:
            mil_in_roi = sum(1 for s in mil_top_k_slices if roi_min <= s <= roi_max)
            mil_overlap = mil_in_roi / len(mil_top_k_slices) if len(mil_top_k_slices) > 0 else 0.0
    
    # CNN overlap: from computed overlap info
    cnn_overlap = cnn_overlap_info.get('cnn_roi_overlap_ratio') if cnn_overlap_info else None
    
    return {
        'mil_roi_overlap_ratio': mil_overlap,
        'cnn_roi_overlap_ratio': cnn_overlap
    }


def create_hierarchical_visualization(
    patient_id: str,
    class_name: str,
    cnn_gradcam: Optional[Dict],
    mil_attention: Optional[Dict],
    tumor_mask: Optional[np.ndarray],
    output_dir: Path
) -> Dict:
    """Create hierarchical interpretability visualization."""
    patient_dir = output_dir / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Row 1: CNN Grad-CAM (axial slices)
    if cnn_gradcam and cnn_gradcam.get('heatmap') is not None:
        heatmap = cnn_gradcam['heatmap']
        # Show middle slices
        num_slices = min(9, heatmap.shape[0])
        slice_indices = np.linspace(0, heatmap.shape[0] - 1, num_slices, dtype=int)
        
        for i, z_idx in enumerate(slice_indices):
            ax = plt.subplot(3, num_slices, i + 1)
            ax.imshow(heatmap[z_idx], cmap='hot', interpolation='nearest')
            ax.set_title(f'Slice {z_idx}', fontsize=8)
            ax.axis('off')
    else:
        ax = plt.subplot(3, 1, 1)
        ax.text(0.5, 0.5, 'CNN Grad-CAM\nNot Available', ha='center', va='center', fontsize=12)
        ax.axis('off')
    
    # Row 2: MIL attention bar plot
    ax2 = plt.subplot(3, 1, 2)
    if mil_attention and mil_attention.get('attention_df') is not None:
        attention_df = mil_attention['attention_df']
        slice_indices = attention_df['slice_index'].values
        attention_weights = attention_df['attention_weight'].values
        
        # Get top-5
        top_5_indices = np.argsort(attention_weights)[-5:][::-1]
        top_5_slice_idx = slice_indices[top_5_indices]
        
        colors = ['red' if idx in top_5_slice_idx else 'steelblue' for idx in slice_indices]
        ax2.bar(slice_indices, attention_weights, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Slice Index', fontsize=10)
        ax2.set_ylabel('Attention Weight', fontsize=10)
        ax2.set_title('MIL Attention Distribution', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'MIL Attention\nNot Available', ha='center', va='center', fontsize=12)
        ax2.axis('off')
    
    # Row 3: Tumor slice range (ROI mask)
    ax3 = plt.subplot(3, 1, 3)
    if tumor_mask is not None:
        roi_min, roi_max = get_tumor_slice_range(tumor_mask)
        if roi_min is not None and roi_max is not None:
            # Create bar showing tumor range
            depth = tumor_mask.shape[0]
            tumor_binary = np.zeros(depth)
            for z in range(roi_min, roi_max + 1):
                if np.any(tumor_mask[z, :, :] > 0):
                    tumor_binary[z] = 1
            
            ax3.bar(range(depth), tumor_binary, color='green', alpha=0.7, edgecolor='black', linewidth=0.5)
            ax3.set_xlabel('Slice Index', fontsize=10)
            ax3.set_ylabel('Tumor Present', fontsize=10)
            ax3.set_title(f'Tumor Slice Range: {roi_min} - {roi_max}', fontsize=12, fontweight='bold')
            ax3.set_ylim(-0.1, 1.1)
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(['No', 'Yes'])
            ax3.grid(True, alpha=0.3, axis='y')
        else:
            ax3.text(0.5, 0.5, 'No Tumor Detected', ha='center', va='center', fontsize=12)
            ax3.axis('off')
    else:
        ax3.text(0.5, 0.5, 'ROI Mask\nNot Available', ha='center', va='center', fontsize=12)
        ax3.axis('off')
    
    plt.suptitle(f'Hierarchical Interpretability - {patient_id} ({class_name})', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = patient_dir / 'hierarchical_interpretability.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Saved hierarchical visualization: {output_path}")
    
    return {
        'visualization_path': str(output_path)
    }


def process_patient(patient_id: str, cnn_variant: Optional[str] = None, mil_variant: str = 'baseline', cnn_gradcam_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> Optional[Dict]:
    """Process a single patient for hierarchical interpretability."""
    logger.info(f"\nProcessing {patient_id}...")
    
    # Find patient class
    found_class = None
    for class_name in ['LGG', 'HGG']:
        patient_dir = DATA_ROOT / class_name / patient_id
        if patient_dir.exists():
            found_class = class_name
            break
    
    if found_class is None:
        logger.error(f"Patient {patient_id} not found")
        return None
    
    # Load data
    cnn_gradcam = load_cnn_gradcam(patient_id, cnn_variant, cnn_gradcam_dir)
    mil_attention = load_mil_attention(patient_id, mil_variant)
    tumor_mask = load_tumor_mask(patient_id, found_class)
    
    # Extract metrics
    cnn_peak_slice = get_cnn_peak_slice(cnn_gradcam) if cnn_gradcam else None
    
    # Compute CNN overlap with tumor mask
    cnn_overlap_info = compute_cnn_tumor_overlap(cnn_gradcam, tumor_mask, threshold_percentile=90.0)
    
    mil_top_k_slices = []
    if mil_attention and mil_attention.get('overlap_info', {}).get('top_k_slice_indices'):
        mil_top_k_slices = mil_attention['overlap_info']['top_k_slice_indices']
    elif mil_attention and mil_attention.get('attention_df') is not None:
        # Fallback: get top-5 from attention_df
        attention_df = mil_attention['attention_df']
        top_5 = attention_df.nlargest(5, 'attention_weight')
        if 'z_coordinate' in top_5.columns:
            mil_top_k_slices = top_5['z_coordinate'].dropna().astype(int).tolist()
        else:
            mil_top_k_slices = top_5['slice_index'].astype(int).tolist()
    
    roi_slice_range = get_tumor_slice_range(tumor_mask)
    
    # Compute overlaps
    overlap_ratios = compute_overlap_ratios(mil_top_k_slices, cnn_overlap_info, roi_slice_range)
    
    # Use provided output_dir or default
    actual_output_dir = output_dir if output_dir is not None else OUTPUT_DIR
    
    # Create visualization
    vis_info = create_hierarchical_visualization(
        patient_id, found_class, cnn_gradcam, mil_attention, tumor_mask, actual_output_dir
    )
    
    # Create summary
    summary = {
        'patient_id': patient_id,
        'true_label': found_class,
        'predicted_label': mil_attention.get('predicted_class_name') if mil_attention else None,
        'cnn_peak_slice': cnn_peak_slice,
        'cnn_overlap_info': cnn_overlap_info,
        'mil_top_k_slices': mil_top_k_slices,
        'roi_slice_range': {
            'min': roi_slice_range[0],
            'max': roi_slice_range[1]
        },
        'mil_roi_overlap_ratio': overlap_ratios['mil_roi_overlap_ratio'],
        'cnn_roi_overlap_ratio': overlap_ratios['cnn_roi_overlap_ratio'],
        'aligned': (
            overlap_ratios['mil_roi_overlap_ratio'] is not None and overlap_ratios['mil_roi_overlap_ratio'] > 0.5 and
            overlap_ratios['cnn_roi_overlap_ratio'] is not None and overlap_ratios['cnn_roi_overlap_ratio'] > 0.5
        )
    }
    
    # Save summary
    summary_path = actual_output_dir / patient_id / 'hierarchical_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  ✓ Saved hierarchical summary: {summary_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Create hierarchical interpretability visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--patient_ids_file',
        type=str,
        default=str(PROJECT_ROOT / 'data' / 'selected_patients.txt'),
        help='Path to file with patient IDs (default: data/selected_patients.txt)'
    )
    parser.add_argument(
        '--cnn_variant',
        type=str,
        default=None,
        help='CNN variant to use as fallback (optional, searches direct path first)'
    )
    parser.add_argument(
        '--mil_variant',
        type=str,
        default='baseline',
        help='MIL variant to use (default: baseline)'
    )
    parser.add_argument(
        '--cnn_gradcam_dir',
        type=str,
        default=None,
        help='Custom path to CNN Grad-CAM directory (default: ensemble/results/interpretability/cnn_gradcam/{variant})'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory for hierarchical results (default: ensemble/results/interpretability/hierarchical)'
    )
    
    args = parser.parse_args()
    
    # Load patient IDs
    patient_ids_file = Path(args.patient_ids_file)
    if not patient_ids_file.exists():
        logger.error(f"Patient IDs file not found: {patient_ids_file}")
        return 1
    
    with open(patient_ids_file, 'r') as f:
        patient_ids = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(patient_ids)} patient IDs")
    logger.info(f"CNN variant: {args.cnn_variant}")
    logger.info(f"MIL variant: {args.mil_variant}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_DIR
    
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*80}")
    logger.info(f"PROCESSING VARIANT: {args.mil_variant.upper()}")
    logger.info(f"{'='*80}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process patients
    all_summaries = []
    cnn_gradcam_dir = Path(args.cnn_gradcam_dir) if args.cnn_gradcam_dir else None
    for patient_id in patient_ids:
        summary = process_patient(patient_id, args.cnn_variant, args.mil_variant, cnn_gradcam_dir, output_dir)
        if summary:
            all_summaries.append(summary)
    
    # Create validation summary table
    logger.info(f"\n{'='*80}")
    logger.info("VALIDATION SUMMARY TABLE")
    logger.info(f"{'='*80}")
    logger.info(f"{'Patient ID':<25} {'MIL Overlap':<15} {'CNN Overlap':<15} {'Aligned':<10}")
    logger.info(f"{'-'*80}")
    
    for summary in all_summaries:
        mil_overlap = f"{summary['mil_roi_overlap_ratio']:.2f}" if summary['mil_roi_overlap_ratio'] is not None else "N/A"
        cnn_overlap = f"{summary['cnn_roi_overlap_ratio']:.2f}" if summary['cnn_roi_overlap_ratio'] is not None else "N/A"
        aligned = "✓" if summary['aligned'] else "✗"
        logger.info(f"{summary['patient_id']:<25} {mil_overlap:<15} {cnn_overlap:<15} {aligned:<10}")
    
    logger.info(f"{'='*80}\n")
    
    # Save combined summary
    combined_summary = {
        'total_patients': len(patient_ids),
        'processed': len(all_summaries),
        'aligned_count': sum(1 for s in all_summaries if s['aligned']),
        'summaries': all_summaries
    }
    
    combined_path = output_dir / 'combined_summary.json'
    with open(combined_path, 'w') as f:
        json.dump(combined_summary, f, indent=2)
    logger.info(f"✓ Saved combined summary: {combined_path}")
    
    logger.info(f"\n{'='*80}")
    logger.info("HIERARCHICAL INTERPRETABILITY COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Variant: {args.mil_variant.upper()}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Aligned patients: {combined_summary['aligned_count']}/{combined_summary['processed']}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

