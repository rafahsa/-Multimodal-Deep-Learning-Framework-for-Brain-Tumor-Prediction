#!/usr/bin/env python3
"""
Stage 1: N4 Bias Field Correction for BraTS2018 3D NIfTI volumes.

This script applies N4 bias field correction to imaging modalities (t1, t1ce, t2, flair)
while preserving segmentation masks. It supports resumability, parallel processing,
and comprehensive logging.

Usage:
    python scripts/preprocessing/run_stage1_n4.py --split train --classes HGG LGG --workers auto
    python scripts/preprocessing/run_stage1_n4.py --split train --classes HGG --dry-run
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import SimpleITK as sitk
import yaml


# Configure logging
def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Set up logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"stage1_n4_{timestamp}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Root logger
    logger = logging.getLogger('stage1_n4')
    logger.setLevel(getattr(logging, log_level))
    logger.handlers.clear()  # Avoid duplicate handlers
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


# N4 Correction Implementation
def create_brain_mask(image: sitk.Image) -> sitk.Image:
    """
    Create a brain mask using Otsu thresholding.
    
    Args:
        image: Input SimpleITK image
        
    Returns:
        Binary mask image (0 or 1)
    """
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(1)
    otsu_filter.SetOutsideValue(0)
    mask = otsu_filter.Execute(image)
    return mask


def apply_n4_correction(
    input_path: Path,
    output_path: Path,
    max_iterations: List[int] = [40, 40, 30, 20],
    num_control_points: int = 4,
    convergence_threshold: float = 0.001
) -> Tuple[bool, Optional[str]]:
    """
    Apply N4 bias field correction to a NIfTI volume.
    
    Args:
        input_path: Path to input NIfTI file
        output_path: Path to output NIfTI file
        max_iterations: Maximum iterations per resolution level
        num_control_points: Number of control points for B-spline
        convergence_threshold: Convergence threshold
        
    Returns:
        (success: bool, error_message: Optional[str])
    """
    try:
        # Read input image
        image = sitk.ReadImage(str(input_path))
        
        # Convert to float32 for processing
        image_float = sitk.Cast(image, sitk.sitkFloat32)
        
        # Create brain mask
        mask = create_brain_mask(image_float)
        
        # Initialize N4 filter
        n4_filter = sitk.N4BiasFieldCorrectionImageFilter()
        n4_filter.SetMaximumNumberOfIterations(max_iterations)
        n4_filter.SetNumberOfControlPoints(num_control_points)
        n4_filter.SetConvergenceThreshold(convergence_threshold)
        
        # Apply N4 correction
        corrected = n4_filter.Execute(image_float, mask)
        
        # Clamp negative values to 0 (but preserve image object)
        corrected_array = sitk.GetArrayFromImage(corrected)
        corrected_array[corrected_array < 0] = 0
        corrected = sitk.GetImageFromArray(corrected_array)
        
        # Preserve original spacing, origin, and direction
        corrected.SetSpacing(image.GetSpacing())
        corrected.SetOrigin(image.GetOrigin())
        corrected.SetDirection(image.GetDirection())
        
        # Cast back to original pixel type (if not float, convert appropriately)
        # For medical images, typically we want to preserve as float32 or convert to original
        original_type = image.GetPixelID()
        if original_type in [sitk.sitkUInt8, sitk.sitkInt16, sitk.sitkUInt16, sitk.sitkInt32, sitk.sitkUInt32]:
            # For integer types, we'll keep as float32 for precision
            corrected = sitk.Cast(corrected, sitk.sitkFloat32)
        else:
            corrected = sitk.Cast(corrected, original_type)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write output (use .nii.gz to save space)
        if not output_path.suffixes or output_path.suffix == '.nii':
            output_path = output_path.parent / f"{output_path.stem}.nii.gz"
        
        sitk.WriteImage(corrected, str(output_path))
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def process_single_file(
    input_path: Path,
    output_path: Path,
    modality: str,
    patient_id: str,
    n4_params: Dict,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Process a single NIfTI file with N4 correction.
    
    Returns a result dictionary for manifest tracking.
    """
    result = {
        'patient_id': patient_id,
        'modality': modality,
        'input_path': str(input_path),
        'output_path': str(output_path),
        'timestamp': datetime.now().isoformat(),
        'status': 'unknown',
        'error': None
    }
    
    try:
        # Check if already processed
        if output_path.exists() and output_path.stat().st_size > 0:
            # Verify it's readable
            try:
                test_image = sitk.ReadImage(str(output_path))
                if logger:
                    logger.info(f"SKIP (already exists): {output_path}")
                result['status'] = 'skipped'
                return result
            except:
                # File exists but corrupted, reprocess
                if logger:
                    logger.warning(f"EXISTS but corrupted, reprocessing: {output_path}")
        
        # Validate input file
        if not input_path.exists():
            error_msg = f"Input file not found: {input_path}"
            if logger:
                logger.error(error_msg)
            result['status'] = 'failed'
            result['error'] = error_msg
            return result
        
        # Apply N4 correction
        success, error_msg = apply_n4_correction(
            input_path,
            output_path,
            max_iterations=n4_params.get('max_iterations', [40, 40, 30, 20]),
            num_control_points=n4_params.get('num_control_points', 4),
            convergence_threshold=n4_params.get('convergence_threshold', 0.001)
        )
        
        if success:
            # Verify output
            try:
                verify_image = sitk.ReadImage(str(output_path))
                if logger:
                    logger.info(f"SUCCESS: {output_path}")
                result['status'] = 'success'
            except Exception as e:
                error_msg = f"Output verification failed: {str(e)}"
                if logger:
                    logger.error(f"VERIFY FAILED: {output_path} - {error_msg}")
                result['status'] = 'failed'
                result['error'] = error_msg
        else:
            if logger:
                logger.error(f"FAILED: {output_path} - {error_msg}")
            result['status'] = 'failed'
            result['error'] = error_msg
            
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        if logger:
            logger.error(f"ERROR processing {input_path}: {error_msg}")
        result['status'] = 'failed'
        result['error'] = error_msg
    
    return result


def worker_process_file(args_tuple: Tuple) -> Dict:
    """Wrapper for multiprocessing worker."""
    input_path, output_path, modality, patient_id, n4_params = args_tuple
    return process_single_file(
        input_path, output_path, modality, patient_id, n4_params, logger=None
    )


def load_manifest(manifest_path: Path) -> Dict[str, Dict]:
    """Load existing manifest file."""
    manifest = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        key = f"{entry['patient_id']}_{entry['modality']}"
                        manifest[key] = entry
        except Exception as e:
            print(f"Warning: Could not load manifest: {e}")
    return manifest


def save_manifest_entry(manifest_path: Path, entry: Dict):
    """Append a single entry to the manifest (JSONL format)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def discover_patients(
    data_root: Path,
    split: str,
    classes: List[str]
) -> List[Tuple[str, str, List[Tuple[str, Path]]]]:
    """
    Discover patient folders and their modality files.
    
    Returns:
        List of (class_name, patient_id, [(modality, input_path), ...])
    """
    patients = []
    modalities_to_process = ['t1', 't1ce', 't2', 'flair']  # NOT seg
    
    for class_name in classes:
        class_dir = data_root / f"MICCAI_BraTS_2018_Data_Training" / class_name
        if not class_dir.exists():
            continue
        
        for patient_dir in sorted(class_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            
            patient_id = patient_dir.name
            modality_files = []
            
            for modality in modalities_to_process:
                # Try both .nii and .nii.gz
                input_file = patient_dir / f"{patient_id}_{modality}.nii"
                if not input_file.exists():
                    input_file = patient_dir / f"{patient_id}_{modality}.nii.gz"
                
                if input_file.exists():
                    modality_files.append((modality, input_file))
            
            if modality_files:
                patients.append((class_name, patient_id, modality_files))
    
    return patients


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: N4 Bias Field Correction for BraTS2018"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/stage_1_n4.yaml',
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train'],
        help='Dataset split to process'
    )
    parser.add_argument(
        '--classes',
        type=str,
        nargs='+',
        default=['HGG', 'LGG'],
        help='Classes to process (HGG, LGG)'
    )
    parser.add_argument(
        '--workers',
        type=str,
        default='auto',
        help='Number of worker processes (int or "auto")'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry-run mode: list files without processing'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent.parent / config_path
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up paths
    project_root = Path(config['paths']['project_root'])
    data_root = project_root / config['paths']['data_root']
    output_root = project_root / config['paths']['output_root']
    log_dir = project_root / config['paths']['log_dir']
    manifest_path = output_root / config['paths']['manifest_file']
    
    # Set up logging
    logger = setup_logging(log_dir, config.get('log_level', 'INFO'))
    logger.info("=" * 80)
    logger.info("Stage 1: N4 Bias Field Correction")
    logger.info("=" * 80)
    logger.info(f"Config: {config_path}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Classes: {args.classes}")
    
    # Determine worker count
    if args.workers == 'auto':
        cpu_count = mp.cpu_count()
        num_workers = max(1, cpu_count - 2)
        logger.info(f"Auto-detected CPU cores: {cpu_count}, using workers: {num_workers}")
    else:
        num_workers = int(args.workers)
        logger.info(f"Using {num_workers} workers")
    
    # Discover patients
    logger.info(f"Discovering patients in {data_root}")
    patients = discover_patients(data_root, args.split, args.classes)
    logger.info(f"Found {len(patients)} patients")
    
    # Build task list
    tasks = []
    for class_name, patient_id, modality_files in patients:
        for modality, input_path in modality_files:
            output_path = output_root / args.split / class_name / patient_id / f"{patient_id}_{modality}.nii.gz"
            tasks.append((input_path, output_path, modality, patient_id))
    
    logger.info(f"Total tasks: {len(tasks)}")
    
    # Dry-run mode
    if args.dry_run:
        logger.info("DRY-RUN MODE: Listing files to process")
        print("\n" + "=" * 80)
        print("DRY-RUN: Files to be processed")
        print("=" * 80)
        for input_path, output_path, modality, patient_id in tasks[:20]:  # Show first 20
            print(f"  {patient_id}/{modality}: {input_path.name} -> {output_path}")
        if len(tasks) > 20:
            print(f"  ... and {len(tasks) - 20} more files")
        print(f"\nTotal: {len(tasks)} files")
        return
    
    # Load existing manifest
    manifest = load_manifest(manifest_path)
    logger.info(f"Loaded {len(manifest)} entries from manifest")
    
    # Filter out already processed files
    tasks_to_process = []
    for input_path, output_path, modality, patient_id in tasks:
        key = f"{patient_id}_{modality}"
        if key in manifest:
            entry = manifest[key]
            if entry['status'] == 'success' and Path(entry['output_path']).exists():
                continue  # Skip already processed
        tasks_to_process.append((input_path, output_path, modality, patient_id))
    
    logger.info(f"Tasks to process: {len(tasks_to_process)} (skipped: {len(tasks) - len(tasks_to_process)})")
    
    if not tasks_to_process:
        logger.info("All files already processed. Use --force to reprocess.")
        return
    
    # Prepare N4 parameters
    n4_params = config.get('n4_parameters', {})
    
    # Process files
    logger.info("Starting processing...")
    results = []
    
    if num_workers == 1:
        # Single-threaded for debugging
        for task in tasks_to_process:
            input_path, output_path, modality, patient_id = task
            result = process_single_file(
                input_path, output_path, modality, patient_id, n4_params, logger
            )
            results.append(result)
            save_manifest_entry(manifest_path, result)
    else:
        # Multi-threaded
        task_args = [
            (input_path, output_path, modality, patient_id, n4_params)
            for input_path, output_path, modality, patient_id in tasks_to_process
        ]
        
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(worker_process_file, task_args)
        
        # Save all results to manifest
        for result in results:
            save_manifest_entry(manifest_path, result)
        
        # Also log results
        for result in results:
            if result['status'] == 'success':
                logger.info(f"SUCCESS: {result['patient_id']}/{result['modality']}")
            elif result['status'] == 'skipped':
                logger.info(f"SKIPPED: {result['patient_id']}/{result['modality']}")
            else:
                logger.error(f"FAILED: {result['patient_id']}/{result['modality']} - {result['error']}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    
    status_counts = defaultdict(int)
    class_counts = defaultdict(int)
    
    for result in results:
        status_counts[result['status']] += 1
        class_counts[result['patient_id'].split('_')[0]] += 1
    
    print(f"\nStatus breakdown:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:12s}: {count:4d}")
    
    print(f"\nTotal processed: {len(results)}")
    print(f"Output directory: {output_root}")
    print(f"Manifest file: {manifest_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()

