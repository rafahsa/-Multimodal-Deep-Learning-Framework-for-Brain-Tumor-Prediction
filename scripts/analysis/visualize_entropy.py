#!/usr/bin/env python3
"""
Entropy-based Slice Selection Visualization

This script creates publication-ready visualizations of entropy-based slice selection
for Multiple Instance Learning (MIL) models. It generates plots showing:
1. Entropy curve with top-k slices highlighted
2. Visual comparison of high-entropy vs low-entropy slices

This is a visualization-only tool for academic reporting and model interpretability.

Usage:
    python scripts/analysis/visualize_entropy.py --patient-id Brats18_2013_10_1 --modality flair
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("Error: matplotlib is required for visualization.")
    print("Install with: pip install matplotlib")
    sys.exit(1)

import numpy as np
import SimpleITK as sitk


def load_entropy_data(entropy_file: Path) -> dict:
    """
    Load entropy JSON file.
    
    Args:
        entropy_file: Path to entropy JSON file
        
    Returns:
        Dictionary with entropy data
    """
    if not entropy_file.exists():
        raise FileNotFoundError(f"Entropy file not found: {entropy_file}")
    
    with open(entropy_file, 'r') as f:
        return json.load(f)


def load_volume(volume_path: Path) -> np.ndarray:
    """
    Load NIfTI volume and return as numpy array.
    
    Args:
        volume_path: Path to NIfTI file
        
    Returns:
        Numpy array of shape (D, H, W)
    """
    if not volume_path.exists():
        raise FileNotFoundError(f"Volume file not found: {volume_path}")
    
    image = sitk.ReadImage(str(volume_path))
    array = sitk.GetArrayFromImage(image)  # Shape: (D, H, W)
    
    return array


def plot_entropy_curve(
    entropy_scores: List[float],
    top_k_indices: List[int],
    axis: str = "axial",
    top_k: int = 16,
    ax=None
) -> None:
    """
    Plot entropy curve with top-k slices highlighted.
    
    Args:
        entropy_scores: List of entropy values per slice
        top_k_indices: List of top-k slice indices
        axis: Slice axis name
        top_k: Number of top slices
        ax: Matplotlib axes (if None, uses current axes)
    """
    if ax is None:
        ax = plt.gca()
    
    slice_indices = np.arange(len(entropy_scores))
    
    # Plot all entropy values
    ax.plot(slice_indices, entropy_scores, 'b-', linewidth=1.5, alpha=0.7, label='All slices')
    
    # Highlight top-k slices
    top_k_entropies = [entropy_scores[i] for i in top_k_indices]
    ax.scatter(
        top_k_indices, top_k_entropies,
        c='red', marker='o', s=50, zorder=5,
        label=f'Top-{top_k} slices (selected)', edgecolors='darkred', linewidths=1
    )
    
    # Formatting
    ax.set_xlabel(f'Slice Index ({axis.capitalize()} axis)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Shannon Entropy', fontsize=12, fontweight='bold')
    ax.set_title(f'Entropy-based Slice Informativeness\n(Top-{top_k} slices selected for MIL)', 
                 fontsize=13, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Set y-axis to start from 0 or slightly below minimum
    y_min = min(entropy_scores)
    y_max = max(entropy_scores)
    y_range = y_max - y_min
    ax.set_ylim(max(0, y_min - 0.05 * y_range), y_max + 0.05 * y_range)
    
    ax.tick_params(labelsize=10)


def visualize_slices(
    volume: np.ndarray,
    slice_indices: List[int],
    entropy_scores: List[float],
    title_prefix: str,
    axis: str = "axial",
    num_slices: int = 3,
    axs=None
) -> None:
    """
    Visualize slices from volume.
    
    Args:
        volume: 3D volume array (D, H, W)
        slice_indices: List of slice indices to visualize
        entropy_scores: List of entropy values (for labeling)
        title_prefix: Prefix for subplot titles
        axis: Slice axis ("axial", "coronal", "sagittal")
        num_slices: Number of slices to show
        axs: Array of matplotlib axes (if None, creates new subplots)
    """
    num_slices = min(num_slices, len(slice_indices))
    
    if axs is None:
        fig, axs = plt.subplots(1, num_slices, figsize=(4*num_slices, 4))
        if num_slices == 1:
            axs = [axs]
    
    for idx, slice_idx in enumerate(slice_indices[:num_slices]):
        ax = axs[idx]
        
        # Extract slice
        if axis == "axial":
            slice_2d = volume[slice_idx, :, :]
        elif axis == "coronal":
            slice_2d = volume[:, slice_idx, :]
        else:  # sagittal
            slice_2d = volume[:, :, slice_idx]
        
        # Display slice
        entropy_val = entropy_scores[slice_idx]
        im = ax.imshow(slice_2d, cmap='gray', aspect='auto')
        
        # Formatting
        ax.set_title(f'{title_prefix}\nSlice {slice_idx} (H={entropy_val:.3f})',
                    fontsize=10, fontweight='bold')
        ax.set_xlabel('Width', fontsize=9)
        ax.set_ylabel('Height', fontsize=9)
        ax.tick_params(labelsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def create_entropy_visualization(
    patient_id: str,
    entropy_data: dict,
    volume: np.ndarray,
    output_path: Path,
    num_example_slices: int = 3
) -> None:
    """
    Create complete entropy visualization figure.
    
    Args:
        patient_id: Patient identifier
        entropy_data: Dictionary with entropy analysis results
        volume: 3D volume array
        output_path: Path to save figure
        num_example_slices: Number of example slices to show (high/low)
    """
    entropy_scores = entropy_data['entropy_per_slice']
    top_k_indices = sorted(entropy_data['top_k_slices'])  # Sort for display
    axis = entropy_data.get('axis', 'axial')
    top_k = entropy_data.get('top_k', len(top_k_indices))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.35, top=0.95, bottom=0.05)
    
    # 1. Entropy curve plot
    ax1 = fig.add_subplot(gs[0, 0])
    plot_entropy_curve(entropy_scores, top_k_indices, axis, top_k, ax=ax1)
    
    # 2. High-entropy slices
    # Get indices of highest entropy slices (not necessarily top-k, just highest values)
    sorted_indices_by_entropy = sorted(
        range(len(entropy_scores)),
        key=lambda i: entropy_scores[i],
        reverse=True
    )
    high_entropy_indices = sorted(sorted_indices_by_entropy[:num_example_slices])
    
    # Create nested gridspec for high-entropy slices
    gs2 = gs[1, 0].subgridspec(1, num_example_slices, wspace=0.3)
    axs_high = [fig.add_subplot(gs2[0, i]) for i in range(num_example_slices)]
    
    # Add title for high-entropy section
    fig.text(0.5, 0.64, 'High-Entropy Slices (Most Informative)', 
             fontsize=12, fontweight='bold', ha='center', va='bottom')
    
    visualize_slices(volume, high_entropy_indices, entropy_scores, 
                    'High Entropy', axis, num_example_slices, axs_high)
    
    # 3. Low-entropy slices
    low_entropy_indices = sorted(sorted_indices_by_entropy[-num_example_slices:])
    
    # Create nested gridspec for low-entropy slices
    gs3 = gs[2, 0].subgridspec(1, num_example_slices, wspace=0.3)
    axs_low = [fig.add_subplot(gs3[0, i]) for i in range(num_example_slices)]
    
    # Add title for low-entropy section
    fig.text(0.5, 0.315, 'Low-Entropy Slices (Least Informative)', 
             fontsize=12, fontweight='bold', ha='center', va='bottom')
    
    visualize_slices(volume, low_entropy_indices, entropy_scores,
                    'Low Entropy', axis, num_example_slices, axs_low)
    
    # Overall title
    fig.suptitle(f'Entropy-based Slice Selection: {patient_id}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize entropy-based slice selection for MIL"
    )
    parser.add_argument(
        '--patient-id',
        type=str,
        required=True,
        help='Patient ID (e.g., Brats18_2013_10_1)'
    )
    parser.add_argument(
        '--modality',
        type=str,
        default='flair',
        choices=['t1', 't1ce', 't2', 'flair'],
        help='Modality to visualize (default: flair)'
    )
    parser.add_argument(
        '--entropy-dir',
        type=str,
        default='data/entropy',
        help='Directory containing entropy JSON files'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/processed/stage_4_resize/train',
        help='Path to Stage 4 data root'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/entropy_visualization',
        help='Output directory for visualization figures'
    )
    parser.add_argument(
        '--num-slices',
        type=int,
        default=3,
        help='Number of example slices to show for high/low entropy (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent.parent
    entropy_dir = project_root / args.entropy_dir
    data_root = project_root / args.data_root
    output_dir = project_root / args.output_dir
    
    # Find patient class (HGG or LGG)
    patient_class = None
    for class_name in ['HGG', 'LGG']:
        patient_dir = data_root / class_name / args.patient_id
        if patient_dir.exists():
            patient_class = class_name
            break
    
    if patient_class is None:
        print(f"Error: Patient {args.patient_id} not found in {data_root}")
        sys.exit(1)
    
    # Load entropy data
    entropy_file = entropy_dir / f"{args.patient_id}_entropy.json"
    print(f"Loading entropy data from: {entropy_file}")
    entropy_data = load_entropy_data(entropy_file)
    
    # Load volume
    volume_path = data_root / patient_class / args.patient_id / f"{args.patient_id}_{args.modality}.nii.gz"
    if not volume_path.exists():
        volume_path = data_root / patient_class / args.patient_id / f"{args.patient_id}_{args.modality}.nii"
    
    print(f"Loading volume from: {volume_path}")
    volume = load_volume(volume_path)
    print(f"Volume shape: {volume.shape}")
    
    # Create visualization
    output_path = output_dir / f"{args.patient_id}_entropy.png"
    print(f"\nCreating visualization...")
    create_entropy_visualization(
        patient_id=args.patient_id,
        entropy_data=entropy_data,
        volume=volume,
        output_path=output_path,
        num_example_slices=args.num_slices
    )
    
    print(f"\nVisualization complete!")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()

