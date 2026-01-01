#!/usr/bin/env python3
"""
Stage 3: ROI Cropping for BraTS2018 3D NIfTI volumes.

This script crops MRI volumes to a bounding box around the brain ROI, ensuring
all modalities for the same patient use the same bounding box. It supports
resumability, parallel processing, and comprehensive logging.

Usage:
    python scripts/preprocessing/run_stage3_crop.py --split train --classes HGG LGG --workers 8
    python scripts/preprocessing/run_stage3_crop.py --split train --classes HGG --dry-run
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import tempfile
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
    log_file = log_dir / f"stage3_crop_{timestamp}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Root logger
    logger = logging.getLogger('stage3_crop')
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


def compute_bounding_box_from_mask(
    mask: np.ndarray,
    padding: int = 10
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Compute bounding box from binary mask with padding.
    
    Args:
        mask: Binary mask array (3D)
        padding: Padding in voxels to add around bounding box
        
    Returns:
        ((z_min, z_max), (y_min, y_max), (x_min, x_max))
    """
    # Find indices where mask is True
    z_indices, y_indices, x_indices = np.where(mask)
    
    if len(z_indices) == 0:
        # No brain voxels found, return full volume
        return ((0, mask.shape[0]), (0, mask.shape[1]), (0, mask.shape[2]))
    
    # Compute bounding box
    z_min, z_max = int(z_indices.min()), int(z_indices.max()) + 1
    y_min, y_max = int(y_indices.min()), int(y_indices.max()) + 1
    x_min, x_max = int(x_indices.min()), int(x_indices.max()) + 1
    
    # Apply padding with bounds checking
    z_min = max(0, z_min - padding)
    z_max = min(mask.shape[0], z_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(mask.shape[1], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(mask.shape[2], x_max + padding)
    
    return ((z_min, z_max), (y_min, y_max), (x_min, x_max))


def compute_bbox_from_volume(
    image_array: np.ndarray,
    eps_mask: float = 1e-6,
    padding: int = 10
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Compute bounding box from image volume using mask threshold.
    
    Args:
        image_array: 3D image array
        eps_mask: Threshold for brain mask (abs(image) > eps_mask)
        padding: Padding in voxels to add around bounding box
        
    Returns:
        ((z_min, z_max), (y_min, y_max), (x_min, x_max))
    """
    # Create mask: brain voxels are those with abs(value) > eps_mask
    mask = np.abs(image_array) > eps_mask
    return compute_bounding_box_from_mask(mask, padding=padding)


def update_origin_after_crop(
    original_image: sitk.Image,
    crop_start: Tuple[int, int, int]
) -> Tuple[float, float, float]:
    """
    Update image origin after cropping based on crop indices.
    
    The new origin accounts for the physical position of the cropped region
    in the original image space.
    
    Args:
        original_image: Original SimpleITK image before cropping
        crop_start: Start indices of crop (z, y, x) in physical coordinates
        
    Returns:
        New origin (x, y, z) in physical coordinates
    """
    # Get original image properties
    spacing = np.array(original_image.GetSpacing())
    direction = np.array(original_image.GetDirection()).reshape(3, 3)
    origin = np.array(original_image.GetOrigin())
    
    # Crop start in (x, y, z) order (SimpleITK uses xyz, but arrays are zyx)
    # Convert from (z, y, x) to (x, y, z)
    crop_start_xyz = np.array([crop_start[2], crop_start[1], crop_start[0]])
    
    # Compute offset in physical space
    # offset = direction @ (spacing * crop_start)
    offset = direction @ (spacing * crop_start_xyz)
    
    # New origin = original origin + offset
    new_origin = origin + offset
    
    return tuple(new_origin.tolist())


def apply_roi_crop(
    input_path: Path,
    output_path: Path,
    bbox: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    eps_mask: float = 1e-6
) -> Tuple[bool, Optional[str]]:
    """
    Apply ROI cropping to a NIfTI volume using precomputed bounding box.
    
    Args:
        input_path: Path to input NIfTI file (from Stage 2)
        output_path: Path to output NIfTI file
        bbox: Bounding box ((z_min, z_max), (y_min, y_max), (x_min, x_max))
        eps_mask: Threshold for mask (not used here, bbox is precomputed)
        
    Returns:
        (success: bool, error_message: Optional[str])
    """
    try:
        # Read input image
        image = sitk.ReadImage(str(input_path))
        
        # Get bounding box coordinates
        (z_min, z_max), (y_min, y_max), (x_min, x_max) = bbox
        
        # Extract region using ExtractImageFilter
        # Note: SimpleITK uses xyz ordering, but arrays are zyx
        extract_filter = sitk.ExtractImageFilter()
        extract_filter.SetSize([int(x_max - x_min), int(y_max - y_min), int(z_max - z_min)])
        extract_filter.SetIndex([int(x_min), int(y_min), int(z_min)])  # xyz order
        
        cropped = extract_filter.Execute(image)
        
        # Update origin based on crop start position
        crop_start = (z_min, y_min, x_min)
        new_origin = update_origin_after_crop(image, crop_start)
        cropped.SetOrigin(new_origin)
        
        # Spacing and direction remain the same
        cropped.SetSpacing(image.GetSpacing())
        cropped.SetDirection(image.GetDirection())
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write output (use .nii.gz to save space)
        if not output_path.suffixes or output_path.suffix == '.nii':
            output_path = output_path.parent / f"{output_path.stem}.nii.gz"
        
        # Write directly (SimpleITK handles file writing)
        sitk.WriteImage(cropped, str(output_path))
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def compute_patient_bbox(
    modality_paths: Dict[str, Path],
    bbox_mode: str,
    reference_modality: str,
    eps_mask: float,
    padding: int = 10,
    logger: Optional[logging.Logger] = None
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """
    Compute bounding box for a patient's volumes.
    
    Args:
        modality_paths: Dictionary mapping modality -> input path
        bbox_mode: "reference_modality" or "union"
        reference_modality: Modality to use if bbox_mode="reference_modality"
        eps_mask: Threshold for brain mask
        logger: Optional logger
        
    Returns:
        Bounding box ((z_min, z_max), (y_min, y_max), (x_min, x_max)) or None if error
    """
    try:
        if bbox_mode == "reference_modality":
            # Use single reference modality
            if reference_modality not in modality_paths:
                error_msg = f"Reference modality '{reference_modality}' not found in available modalities"
                if logger:
                    logger.error(error_msg)
                return None
            
            ref_path = modality_paths[reference_modality]
            ref_image = sitk.ReadImage(str(ref_path))
            ref_array = sitk.GetArrayFromImage(ref_image)
            
            bbox = compute_bbox_from_volume(ref_array, eps_mask, padding)
            return bbox
            
        elif bbox_mode == "union":
            # Use union mask across all modalities
            all_bboxes = []
            
            for modality, path in modality_paths.items():
                try:
                    image = sitk.ReadImage(str(path))
                    image_array = sitk.GetArrayFromImage(image)
                    mask = np.abs(image_array) > eps_mask
                    all_bboxes.append(compute_bounding_box_from_mask(mask, padding=0))
                except Exception as e:
                    if logger:
                        logger.warning(f"Failed to load {modality} for bbox union: {e}")
                    continue
            
            # Apply padding to union bbox
            if all_bboxes:
                # Compute union: min of mins, max of maxs
                z_mins, z_maxs = zip(*[bbox[0] for bbox in all_bboxes])
                y_mins, y_maxs = zip(*[bbox[1] for bbox in all_bboxes])
                x_mins, x_maxs = zip(*[bbox[2] for bbox in all_bboxes])
                
                # Get volume shape to apply padding correctly
                first_image = sitk.ReadImage(str(list(modality_paths.values())[0]))
                shape = sitk.GetArrayFromImage(first_image).shape
                
                bbox = (
                    (max(0, min(z_mins) - padding), min(shape[0], max(z_maxs) + padding)),
                    (max(0, min(y_mins) - padding), min(shape[1], max(y_maxs) + padding)),
                    (max(0, min(x_mins) - padding), min(shape[2], max(x_maxs) + padding))
                )
            
                return bbox
            else:
                error_msg = "No modalities could be loaded for union bbox computation"
                if logger:
                    logger.error(error_msg)
                return None
        else:
            error_msg = f"Unknown bbox_mode: {bbox_mode}"
            if logger:
                logger.error(error_msg)
            return None
            
    except Exception as e:
        error_msg = f"Error computing patient bbox: {str(e)}"
        if logger:
            logger.error(error_msg)
        return None


def process_patient_modalities(
    patient_id: str,
    class_name: str,
    modality_paths: Dict[str, Path],
    output_root: Path,
    split: str,
    bbox: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    crop_params: Dict,
    logger: Optional[logging.Logger] = None
) -> List[Dict]:
    """
    Process all modalities for a patient using the same bounding box.
    
    Returns a list of result dictionaries for manifest tracking.
    """
    results = []
    # Bbox already includes padding from compute_patient_bbox
    
    for modality, input_path in modality_paths.items():
        output_path = output_root / split / class_name / patient_id / f"{patient_id}_{modality}.nii.gz"
        
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
                try:
                    test_image = sitk.ReadImage(str(output_path))
                    if logger:
                        logger.info(f"SKIP (already exists): {output_path}")
                    result['status'] = 'skipped'
                    results.append(result)
                    continue
                except:
                    if logger:
                        logger.warning(f"EXISTS but corrupted, reprocessing: {output_path}")
            
            # Validate input file
            if not input_path.exists():
                error_msg = f"Input file not found: {input_path}"
                if logger:
                    logger.error(error_msg)
                result['status'] = 'failed'
                result['error'] = error_msg
                results.append(result)
                continue
            
            # Apply cropping (bbox already includes padding)
            success, error_msg = apply_roi_crop(
                input_path,
                output_path,
                bbox,
                eps_mask=crop_params.get('eps_mask', 1e-6)
            )
            
            if success:
                # Verify output
                try:
                    verify_image = sitk.ReadImage(str(output_path))
                    if logger:
                        logger.info(f"SUCCESS: {patient_id}/{modality}")
                    result['status'] = 'success'
                except Exception as e:
                    error_msg = f"Output verification failed: {str(e)}"
                    if logger:
                        logger.error(f"VERIFY FAILED: {output_path} - {error_msg}")
                    result['status'] = 'failed'
                    result['error'] = error_msg
            else:
                if logger:
                    logger.error(f"FAILED: {patient_id}/{modality} - {error_msg}")
                result['status'] = 'failed'
                result['error'] = error_msg
                
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            if logger:
                logger.error(f"ERROR processing {patient_id}/{modality}: {error_msg}")
            result['status'] = 'failed'
            result['error'] = error_msg
        
        results.append(result)
    
    return results


def worker_process_patient(args_tuple: Tuple) -> List[Dict]:
    """Wrapper for multiprocessing worker."""
    return process_patient_modalities(*args_tuple)


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
) -> List[Tuple[str, str, Dict[str, Path]]]:
    """
    Discover patient folders and their modality files from Stage 2 outputs.
    
    Returns:
        List of (class_name, patient_id, {modality: input_path, ...})
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
            modality_paths = {}
            
            for modality in modalities_to_process:
                # Try both .nii.gz and .nii (Stage 2 outputs are .nii.gz)
                input_file = patient_dir / f"{patient_id}_{modality}.nii.gz"
                if not input_file.exists():
                    input_file = patient_dir / f"{patient_id}_{modality}.nii"
                
                if input_file.exists():
                    modality_paths[modality] = input_file
            
            if modality_paths:
                patients.append((class_name, patient_id, modality_paths))
    
    return patients


def safe_parse_int(value, param_name: str, default=None) -> int:
    """Safely parse an int value from config."""
    if value is None:
        if default is None:
            raise ValueError(f"{param_name} is required in config")
        return int(default)
    
    try:
        return int(value)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid {param_name} in config: {value} (type={type(value).__name__}). "
            f"Expected integer value. Error: {e}"
        )


def safe_parse_float(value, param_name: str, default=None) -> float:
    """Safely parse a float value from config."""
    if value is None:
        if default is None:
            raise ValueError(f"{param_name} is required in config")
        return float(default)
    
    try:
        return float(value)
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid {param_name} in config: {value} (type={type(value).__name__}). "
            f"Expected numeric value. Error: {e}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: ROI Cropping for BraTS2018"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/stage_3_crop.yaml',
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
    input_root = project_root / config['paths']['input_root']  # Stage 2 outputs
    output_root = project_root / config['paths']['output_root']
    log_dir = project_root / config['paths']['log_dir']
    manifest_path = output_root / config['paths']['manifest_file']
    
    # Set up logging
    logger = setup_logging(log_dir, config.get('log_level', 'INFO'))
    logger.info("=" * 80)
    logger.info("Stage 3: ROI Cropping")
    logger.info("=" * 80)
    logger.info(f"Config: {config_path}")
    logger.info(f"Input root (Stage 2 outputs): {input_root}")
    logger.info(f"Output root: {output_root}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Classes: {args.classes}")
    
    # Parse and validate crop parameters
    crop_config = config.get('crop_parameters', {})
    try:
        padding = safe_parse_int(crop_config.get('padding', 10), 'padding', default=10)
        eps_mask = safe_parse_float(crop_config.get('eps_mask', 1e-6), 'eps_mask', default=1e-6)
        bbox_mode = crop_config.get('bbox_mode', 'reference_modality')
        reference_modality = crop_config.get('reference_modality', 'flair')
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    
    if bbox_mode not in ['reference_modality', 'union']:
        logger.error(f"Invalid bbox_mode: {bbox_mode}. Must be 'reference_modality' or 'union'")
        sys.exit(1)
    
    if reference_modality not in ['t1', 't1ce', 't2', 'flair']:
        logger.error(f"Invalid reference_modality: {reference_modality}")
        sys.exit(1)
    
    logger.info(f"Crop padding: {padding} voxels")
    logger.info(f"Mask epsilon: {eps_mask}")
    logger.info(f"Bbox mode: {bbox_mode}")
    if bbox_mode == 'reference_modality':
        logger.info(f"Reference modality: {reference_modality}")
    
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
    
    # Build task list (per patient, not per file)
    tasks = []
    for class_name, patient_id, modality_paths in patients:
        output_root_patient = output_root / args.split / class_name / patient_id
        tasks.append((patient_id, class_name, modality_paths, output_root, args.split, crop_config))
    
    logger.info(f"Total patients to process: {len(tasks)}")
    
    # Dry-run mode
    if args.dry_run:
        logger.info("DRY-RUN MODE: Listing patients to process")
        print("\n" + "=" * 80)
        print("DRY-RUN: Patients to be processed")
        print("=" * 80)
        for class_name, patient_id, modality_paths in patients[:20]:  # Show first 20
            print(f"  {patient_id} ({class_name}): {len(modality_paths)} modalities")
        if len(patients) > 20:
            print(f"  ... and {len(patients) - 20} more patients")
        # Count total files
        total_files = sum(len(mp) for _, _, mp in patients)
        print(f"\nTotal patients: {len(patients)}")
        print(f"Total files (estimated): {total_files}")
        return
    
    # Load existing manifest
    manifest = load_manifest(manifest_path)
    logger.info(f"Loaded {len(manifest)} entries from manifest")
    
    # Filter out already processed patients (all modalities must be done)
    tasks_to_process = []
    for patient_id, class_name, modality_paths, output_root_task, split_task, crop_params_task in tasks:
        # Check if all modalities for this patient are already processed
        all_done = True
        for modality in modality_paths.keys():
            key = f"{patient_id}_{modality}"
            if key not in manifest:
                all_done = False
                break
            entry = manifest[key]
            output_path_check = Path(entry['output_path'])
            if entry['status'] != 'success' or not output_path_check.exists():
                all_done = False
                break
        
        if not all_done:
            tasks_to_process.append((patient_id, class_name, modality_paths, output_root_task, split_task, crop_params_task))
    
    logger.info(f"Patients to process: {len(tasks_to_process)} (skipped: {len(tasks) - len(tasks_to_process)})")
    
    if not tasks_to_process:
        logger.info("All patients already processed.")
        return
    
    # Prepare crop parameters
    crop_params = {
        'padding': padding,
        'eps_mask': eps_mask,
        'bbox_mode': bbox_mode,
        'reference_modality': reference_modality
    }
    
    # Process patients
    logger.info("Starting processing...")
    start_time = datetime.now()
    all_results = []
    
    if num_workers == 1:
        # Single-threaded for debugging
        for task in tasks_to_process:
            patient_id, class_name, modality_paths, output_root_task, split_task, _ = task
            
            # Compute bounding box for this patient
            bbox = compute_patient_bbox(
                modality_paths, bbox_mode, reference_modality, eps_mask, padding, logger
            )
            if bbox is None:
                # Failed to compute bbox, mark all modalities as failed
                for modality in modality_paths.keys():
                    result = {
                        'patient_id': patient_id,
                        'modality': modality,
                        'input_path': str(modality_paths[modality]),
                        'output_path': str(output_root_task / split_task / class_name / patient_id / f"{patient_id}_{modality}.nii.gz"),
                        'timestamp': datetime.now().isoformat(),
                        'status': 'failed',
                        'error': 'Failed to compute bounding box'
                    }
                    all_results.append(result)
                    save_manifest_entry(manifest_path, result)
                continue
            
            # Process all modalities with same bbox
            results = process_patient_modalities(
                patient_id, class_name, modality_paths, output_root_task, split_task,
                bbox, crop_params, logger
            )
            all_results.extend(results)
            for result in results:
                save_manifest_entry(manifest_path, result)
    else:
        # Multi-threaded
        # Precompute bounding boxes for all patients
        logger.info("Computing bounding boxes for all patients...")
        patient_bboxes = {}
        for task in tasks_to_process:
            patient_id, class_name, modality_paths, _, _, _ = task
            bbox = compute_patient_bbox(
                modality_paths, bbox_mode, reference_modality, eps_mask, logger=None
            )
            if bbox is None:
                logger.warning(f"Failed to compute bbox for {patient_id}, will skip")
            patient_bboxes[patient_id] = bbox
        
        # Prepare task arguments with precomputed bboxes
        task_args = []
        for task in tasks_to_process:
            patient_id, class_name, modality_paths, output_root_task, split_task, _ = task
            bbox = patient_bboxes.get(patient_id)
            if bbox is None:
                continue  # Skip patients with failed bbox computation
            task_args.append((
                patient_id, class_name, modality_paths, output_root_task, split_task,
                bbox, crop_params, None  # logger=None for workers
            ))
        
        with mp.Pool(processes=num_workers) as pool:
            results_list = pool.map(worker_process_patient, task_args)
        
        # Flatten results
        for results in results_list:
            all_results.extend(results)
        
        # Save all results to manifest
        for result in all_results:
            save_manifest_entry(manifest_path, result)
        
        # Also log results
        for result in all_results:
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
    
    for result in all_results:
        status_counts[result['status']] += 1
        # Extract class from patient_id (first part before underscore)
        class_counts[result['patient_id'].split('_')[0]] += 1
    
    print(f"\nStatus breakdown:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:12s}: {count:4d}")
    
    print(f"\nClass breakdown:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name:12s}: {count:4d}")
    
    print(f"\nTotal processed: {len(all_results)} files")
    print(f"Processing time: {duration}")
    print(f"Input directory: {input_root}")
    print(f"Output directory: {output_root}")
    print(f"Manifest file: {manifest_path}")
    print("=" * 80)
    
    logger.info(f"Processing completed in {duration}")
    logger.info(f"Summary: {dict(status_counts)}")


if __name__ == '__main__':
    main()

