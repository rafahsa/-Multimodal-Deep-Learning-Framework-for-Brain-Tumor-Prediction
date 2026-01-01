#!/usr/bin/env python3
"""
Entropy-based Slice Informativeness Analysis Runner

This script computes entropy scores for slices in 3D MRI volumes from Stage 4.
Outputs are metadata-only JSON files used for MIL slice selection.

This is a metadata-only stage: no image files are modified or duplicated.

Usage:
    python scripts/analysis/run_entropy_analysis.py --modality flair --axis axial --top-k 16
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import SimpleITK as sitk
import torch

# Import entropy analysis utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.entropy_analysis import compute_slice_entropy, select_top_k_slices


def load_volume(volume_path: Path) -> torch.Tensor:
    """
    Load NIfTI volume and convert to torch tensor.
    
    Args:
        volume_path: Path to NIfTI file
        
    Returns:
        Torch tensor of shape (D, H, W)
    """
    image = sitk.ReadImage(str(volume_path))
    array = sitk.GetArrayFromImage(image)  # Shape: (D, H, W) in SimpleITK
    
    # Convert to torch tensor
    tensor = torch.from_numpy(array).float()
    
    return tensor


def process_patient(
    patient_id: str,
    volume_path: Path,
    axis: str = "axial",
    top_k: int = 16,
    num_bins: int = 256,
    normalize: bool = False,
    use_cuda: bool = False
) -> Dict:
    """
    Process a single patient volume and compute entropy.
    
    Args:
        patient_id: Patient identifier
        volume_path: Path to volume file
        axis: Slice axis ("axial", "coronal", "sagittal")
        top_k: Number of top slices to select
        num_bins: Number of bins for histogram
        normalize: Whether to normalize entropy
        use_cuda: Whether to use GPU (if available)
        
    Returns:
        Dictionary with entropy analysis results
    """
    try:
        # Load volume
        volume = load_volume(volume_path)
        
        # Move to GPU if requested and available
        if use_cuda and torch.cuda.is_available():
            volume = volume.cuda()
        
        # Compute entropy per slice
        entropy_scores = compute_slice_entropy(
            volume=volume,
            axis=axis,
            num_bins=num_bins,
            normalize=normalize
        )
        
        # Select top-k slices
        top_k_indices = select_top_k_slices(entropy_scores, k=top_k)
        
        # Build result dictionary
        result = {
            "patient_id": patient_id,
            "axis": axis,
            "entropy_per_slice": entropy_scores,
            "top_k": top_k,
            "top_k_slices": top_k_indices,
            "num_slices": len(entropy_scores)
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing {patient_id}: {e}")
        return None


def discover_patients(data_root: Path, modality: str = "flair") -> List[tuple]:
    """
    Discover patient volumes in Stage 4 outputs.
    
    Args:
        data_root: Path to data/processed/stage_4_resize/train/
        modality: Modality to process (t1, t1ce, t2, flair)
        
    Returns:
        List of (patient_id, class_name, volume_path) tuples
    """
    patients = []
    
    for class_name in ['HGG', 'LGG']:
        class_dir = data_root / class_name
        if not class_dir.exists():
            continue
        
        for patient_dir in sorted(class_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            
            patient_id = patient_dir.name
            
            # Look for modality file
            volume_file = patient_dir / f"{patient_id}_{modality}.nii.gz"
            if not volume_file.exists():
                volume_file = patient_dir / f"{patient_id}_{modality}.nii"
            
            if volume_file.exists():
                patients.append((patient_id, class_name, volume_file))
    
    return patients


def main():
    parser = argparse.ArgumentParser(
        description="Entropy-based slice informativeness analysis for MIL"
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/processed/stage_4_resize/train',
        help='Path to Stage 4 train directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/entropy',
        help='Output directory for entropy JSON files'
    )
    parser.add_argument(
        '--modality',
        type=str,
        default='flair',
        choices=['t1', 't1ce', 't2', 'flair'],
        help='Modality to analyze (default: flair)'
    )
    parser.add_argument(
        '--axis',
        type=str,
        default='axial',
        choices=['axial', 'coronal', 'sagittal'],
        help='Slice axis (default: axial)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=16,
        help='Number of top slices to select (default: 16)'
    )
    parser.add_argument(
        '--num-bins',
        type=int,
        default=256,
        help='Number of bins for histogram (default: 256)'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize entropy to [0, 1] range'
    )
    parser.add_argument(
        '--use-cuda',
        action='store_true',
        help='Use GPU if available'
    )
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        default=['HGG', 'LGG'],
        help='Classes to process (default: HGG LGG)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    project_root = Path(__file__).parent.parent.parent
    data_root = project_root / args.data_root
    output_dir = project_root / args.output_dir
    
    if not data_root.exists():
        print(f"Error: Data root not found: {data_root}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Entropy-based Slice Informativeness Analysis")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print(f"Output directory: {output_dir}")
    print(f"Modality: {args.modality}")
    print(f"Axis: {args.axis}")
    print(f"Top-k: {args.top_k}")
    print(f"Use CUDA: {args.use_cuda and torch.cuda.is_available()}")
    print("=" * 60)
    
    # Discover patients
    print(f"\nDiscovering patients...")
    all_patients = discover_patients(data_root, modality=args.modality)
    
    # Filter by class if specified
    patients = [
        (pid, cls, path) for pid, cls, path in all_patients
        if cls in args.classes
    ]
    
    print(f"Found {len(patients)} patients")
    
    if len(patients) == 0:
        print("Error: No patients found!")
        sys.exit(1)
    
    # Process patients
    print(f"\nProcessing patients...")
    processed = 0
    failed = 0
    
    for patient_id, class_name, volume_path in patients:
        result = process_patient(
            patient_id=patient_id,
            volume_path=volume_path,
            axis=args.axis,
            top_k=args.top_k,
            num_bins=args.num_bins,
            normalize=args.normalize,
            use_cuda=args.use_cuda
        )
        
        if result is None:
            failed += 1
            continue
        
        # Save JSON file
        output_file = output_dir / f"{patient_id}_entropy.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        processed += 1
        if processed % 50 == 0:
            print(f"  Processed {processed}/{len(patients)} patients...")
    
    print(f"\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)
    print(f"Total patients: {len(patients)}")
    print(f"Processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    print("\nNote: This is a metadata-only stage. No image files were modified.")


if __name__ == '__main__':
    main()

