#!/usr/bin/env python3
"""
Stage 4: Resize for BraTS2018 3D NIfTI volumes.

This script resizes MRI volumes from Stage 3 to a fixed target size (128, 128, 128)
using SimpleITK resampling with linear interpolation. It preserves direction and origin,
and correctly updates spacing. It supports resumability, parallel processing, and
comprehensive logging.

Usage:
    python scripts/preprocessing/run_stage4_resize.py --split train --classes HGG LGG --workers 8
    python scripts/preprocessing/run_stage4_resize.py --split train --classes HGG --dry-run
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

import numpy as np
import SimpleITK as sitk
import yaml


# Configure logging
def setup_logging(log_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Set up logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"stage4_resize_{timestamp}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Root logger
    logger = logging.getLogger('stage4_resize')
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


def apply_resize(
    input_path: Path,
    output_path: Path,
    target_size: Tuple[int, int, int],
    interpolation: str = "linear"
) -> Tuple[bool, Optional[str]]:
    """
    Apply resizing to a NIfTI volume using SimpleITK resampling.
    
    Args:
        input_path: Path to input NIfTI file (from Stage 3)
        output_path: Path to output NIfTI file
        target_size: Target size as (x, y, z) tuple
        interpolation: Interpolation method ("linear" or "nearest")
        
    Returns:
        (success: bool, error_message: Optional[str])
    """
    try:
        # Read input image
        image = sitk.ReadImage(str(input_path))
        
        # Get current image properties
        old_size = np.array(image.GetSize())  # SimpleITK: (x, y, z)
        old_spacing = np.array(image.GetSpacing())  # (x, y, z)
        old_origin = image.GetOrigin()
        old_direction = image.GetDirection()
        
        # Convert target_size to tuple (x, y, z) - already in xyz order
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
        
        # Set interpolation method
        if interpolation.lower() == "linear":
            resampler.SetInterpolator(sitk.sitkLinear)
        elif interpolation.lower() == "nearest":
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            # Default to linear
            resampler.SetInterpolator(sitk.sitkLinear)
        
        # Execute resampling
        resized = resampler.Execute(image)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write output (use .nii.gz to save space)
        if not output_path.suffixes or output_path.suffix == '.nii':
            output_path = output_path.parent / f"{output_path.stem}.nii.gz"
        
        # Write directly (SimpleITK handles file writing)
        sitk.WriteImage(resized, str(output_path))
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def process_single_file(
    input_path: Path,
    output_path: Path,
    modality: str,
    patient_id: str,
    resize_params: Dict,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Process a single NIfTI file with resizing.
    
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
            # Verify it's readable and has correct size
            try:
                test_image = sitk.ReadImage(str(output_path))
                test_array = sitk.GetArrayFromImage(test_image)
                target_size = tuple(resize_params.get('target_size', [128, 128, 128]))
                # Check if size matches (numpy arrays are z,y,x, SimpleITK sizes are x,y,z)
                if test_array.shape == (target_size[2], target_size[1], target_size[0]):
                    if logger:
                        logger.info(f"SKIP (already exists): {output_path}")
                    result['status'] = 'skipped'
                    return result
                else:
                    # Wrong size, reprocess
                    if logger:
                        logger.warning(f"EXISTS but wrong size, reprocessing: {output_path}")
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
        
        # Apply resizing
        success, error_msg = apply_resize(
            input_path,
            output_path,
            target_size=tuple(resize_params.get('target_size', [128, 128, 128])),
            interpolation=resize_params.get('interpolation', 'linear')
        )
        
        if success:
            # Verify output
            try:
                verify_image = sitk.ReadImage(str(output_path))
                verify_array = sitk.GetArrayFromImage(verify_image)
                target_size = tuple(resize_params.get('target_size', [128, 128, 128]))
                if verify_array.shape != (target_size[2], target_size[1], target_size[0]):
                    error_msg = f"Output size mismatch: expected {target_size}, got {verify_array.shape}"
                    if logger:
                        logger.error(f"VERIFY FAILED: {output_path} - {error_msg}")
                    result['status'] = 'failed'
                    result['error'] = error_msg
                else:
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
    input_path, output_path, modality, patient_id, resize_params = args_tuple
    return process_single_file(
        input_path, output_path, modality, patient_id, resize_params, logger=None
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
    input_root: Path,
    split: str,
    classes: List[str]
) -> List[Tuple[str, str, List[Tuple[str, Path]]]]:
    """
    Discover patient folders and their modality files from Stage 3 outputs.
    
    Returns:
        List of (class_name, patient_id, [(modality, input_path), ...])
    """
    patients = []
    modalities_to_process = ['t1', 't1ce', 't2', 'flair']  # NOT seg
    
    for class_name in classes:
        class_dir = input_root / split / class_name
        if not class_dir.exists():
            continue
        
        for patient_dir in sorted(class_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            
            patient_id = patient_dir.name
            modality_files = []
            
            for modality in modalities_to_process:
                # Try both .nii.gz and .nii (Stage 3 outputs are .nii.gz)
                input_file = patient_dir / f"{patient_id}_{modality}.nii.gz"
                if not input_file.exists():
                    input_file = patient_dir / f"{patient_id}_{modality}.nii"
                
                if input_file.exists():
                    modality_files.append((modality, input_file))
            
            if modality_files:
                patients.append((class_name, patient_id, modality_files))
    
    return patients


def safe_parse_int_list(value, param_name: str, default=None) -> Tuple[int, int, int]:
    """Safely parse a list of 3 integers from config."""
    if value is None:
        if default is None:
            raise ValueError(f"{param_name} is required in config")
        return tuple(default)
    
    try:
        if isinstance(value, list) and len(value) == 3:
            return tuple(int(v) for v in value)
        elif isinstance(value, str):
            # Try parsing string like "[128, 128, 128]"
            import ast
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and len(parsed) == 3:
                return tuple(int(v) for v in parsed)
        raise ValueError(f"{param_name} must be a list of 3 integers")
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid {param_name} in config: {value} (type={type(value).__name__}). "
            f"Expected list of 3 integers. Error: {e}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Resize for BraTS2018"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/stage_4_resize.yaml',
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
    input_root = project_root / config['paths']['input_root']  # Stage 3 outputs
    output_root = project_root / config['paths']['output_root']
    log_dir = project_root / config['paths']['log_dir']
    manifest_path = output_root / config['paths']['manifest_file']
    
    # Set up logging
    logger = setup_logging(log_dir, config.get('log_level', 'INFO'))
    logger.info("=" * 80)
    logger.info("Stage 4: Resize")
    logger.info("=" * 80)
    logger.info(f"Config: {config_path}")
    logger.info(f"Input root (Stage 3 outputs): {input_root}")
    logger.info(f"Output root: {output_root}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Classes: {args.classes}")
    
    # Parse and validate resize parameters
    resize_config = config.get('resize_parameters', {})
    try:
        target_size = safe_parse_int_list(
            resize_config.get('target_size', [128, 128, 128]),
            'target_size',
            default=[128, 128, 128]
        )
        interpolation = resize_config.get('interpolation', 'linear')
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    if interpolation not in ['linear', 'nearest']:
        logger.warning(f"Unknown interpolation method '{interpolation}', using 'linear'")
        interpolation = 'linear'
    
    logger.info(f"Target size: {target_size} (x, y, z)")
    logger.info(f"Interpolation: {interpolation}")
    
    # Determine worker count (capped formula)
    if args.workers == 'auto':
        cpu_count = mp.cpu_count()
        num_workers = min(max(1, cpu_count // 4), 16)
        logger.info(f"Auto-detected CPU cores: {cpu_count}, using capped workers: {num_workers}")
    else:
        num_workers = int(args.workers)
        logger.info(f"Using {num_workers} workers")
    
    # Discover patients
    logger.info(f"Discovering patients in {input_root}")
    patients = discover_patients(input_root, args.split, args.classes)
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
        print(f"Target size: {target_size}")
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
        logger.info("All files already processed.")
        return
    
    # Prepare resize parameters
    resize_params = {
        'target_size': list(target_size),
        'interpolation': interpolation
    }
    
    # Process files
    logger.info("Starting processing...")
    start_time = datetime.now()
    results = []
    
    if num_workers == 1:
        # Single-threaded for debugging
        for task in tasks_to_process:
            input_path, output_path, modality, patient_id = task
            result = process_single_file(
                input_path, output_path, modality, patient_id, resize_params, logger
            )
            results.append(result)
            save_manifest_entry(manifest_path, result)
    else:
        # Multi-threaded
        task_args = [
            (input_path, output_path, modality, patient_id, resize_params)
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
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    
    status_counts = defaultdict(int)
    class_counts = defaultdict(int)
    
    for result in results:
        status_counts[result['status']] += 1
        # Extract class from patient_id (first part before underscore)
        class_counts[result['patient_id'].split('_')[0]] += 1
    
    print(f"\nStatus breakdown:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:12s}: {count:4d}")
    
    print(f"\nClass breakdown:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name:12s}: {count:4d}")
    
    print(f"\nTotal processed: {len(results)}")
    print(f"Processing time: {duration}")
    print(f"Input directory: {input_root}")
    print(f"Output directory: {output_root}")
    print(f"Manifest file: {manifest_path}")
    print("=" * 80)
    
    logger.info(f"Processing completed in {duration}")
    logger.info(f"Summary: {dict(status_counts)}")


if __name__ == '__main__':
    main()

