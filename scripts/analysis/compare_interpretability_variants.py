#!/usr/bin/env python3
"""
Compare Interpretability Variants: Baseline vs ROI MIL

Compares hierarchical interpretability results between baseline and ROI MIL variants.

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
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default paths
BASELINE_SUMMARY = PROJECT_ROOT / 'ensemble' / 'results' / 'interpretability' / 'hierarchical_baseline' / 'combined_summary.json'
ROI_SUMMARY = PROJECT_ROOT / 'ensemble' / 'results' / 'interpretability' / 'hierarchical_roi' / 'combined_summary.json'


def load_summary(summary_path: Path) -> Optional[Dict]:
    """Load combined summary JSON file."""
    if not summary_path.exists():
        logger.error(f"Summary file not found: {summary_path}")
        return None
    
    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        logger.info(f"✓ Loaded summary: {summary_path}")
        return summary
    except Exception as e:
        logger.error(f"Error loading summary {summary_path}: {e}")
        return None


def extract_metrics(summary: Dict) -> Dict:
    """
    Extract interpretability metrics from summary.
    
    Returns:
        Dict with:
        - mil_overlaps: List of MIL overlap ratios (excluding None)
        - cnn_overlaps: List of CNN overlap ratios (excluding None)
        - aligned_count: Number of aligned patients
        - total_patients: Total number of patients
    """
    summaries = summary.get('summaries', [])
    
    mil_overlaps = []
    cnn_overlaps = []
    aligned_count = 0
    
    for patient_summary in summaries:
        mil_overlap = patient_summary.get('mil_roi_overlap_ratio')
        cnn_overlap = patient_summary.get('cnn_roi_overlap_ratio')
        aligned = patient_summary.get('aligned', False)
        
        if mil_overlap is not None:
            mil_overlaps.append(mil_overlap)
        if cnn_overlap is not None:
            cnn_overlaps.append(cnn_overlap)
        if aligned:
            aligned_count += 1
    
    total_patients = len(summaries)
    
    return {
        'mil_overlaps': mil_overlaps,
        'cnn_overlaps': cnn_overlaps,
        'aligned_count': aligned_count,
        'total_patients': total_patients
    }


def compute_statistics(metrics: Dict) -> Dict:
    """Compute average statistics from metrics."""
    mil_overlaps = metrics['mil_overlaps']
    cnn_overlaps = metrics['cnn_overlaps']
    aligned_count = metrics['aligned_count']
    total_patients = metrics['total_patients']
    
    avg_mil_overlap = sum(mil_overlaps) / len(mil_overlaps) if mil_overlaps else 0.0
    avg_cnn_overlap = sum(cnn_overlaps) / len(cnn_overlaps) if cnn_overlaps else 0.0
    aligned_percentage = (aligned_count / total_patients * 100) if total_patients > 0 else 0.0
    
    return {
        'avg_mil_overlap': avg_mil_overlap,
        'avg_cnn_overlap': avg_cnn_overlap,
        'aligned_count': aligned_count,
        'total_patients': total_patients,
        'aligned_percentage': aligned_percentage
    }


def print_comparison(baseline_stats: Dict, roi_stats: Dict):
    """Print comparison table between baseline and ROI."""
    logger.info(f"\n{'='*80}")
    logger.info("INTERPRETABILITY VARIANT COMPARISON")
    logger.info(f"{'='*80}\n")
    
    logger.info("BASELINE:")
    logger.info(f"  Avg MIL overlap:     {baseline_stats['avg_mil_overlap']:.3f}")
    logger.info(f"  Avg CNN overlap:     {baseline_stats['avg_cnn_overlap']:.3f}")
    logger.info(f"  Aligned patients:    {baseline_stats['aligned_count']} / {baseline_stats['total_patients']} ({baseline_stats['aligned_percentage']:.1f}%)")
    
    logger.info("\nROI:")
    logger.info(f"  Avg MIL overlap:     {roi_stats['avg_mil_overlap']:.3f}")
    logger.info(f"  Avg CNN overlap:     {roi_stats['avg_cnn_overlap']:.3f}")
    logger.info(f"  Aligned patients:    {roi_stats['aligned_count']} / {roi_stats['total_patients']} ({roi_stats['aligned_percentage']:.1f}%)")
    
    # Compute differences
    mil_diff = roi_stats['avg_mil_overlap'] - baseline_stats['avg_mil_overlap']
    cnn_diff = roi_stats['avg_cnn_overlap'] - baseline_stats['avg_cnn_overlap']
    aligned_diff = roi_stats['aligned_count'] - baseline_stats['aligned_count']
    
    logger.info(f"\n{'='*80}")
    logger.info("DIFFERENCES (ROI - BASELINE):")
    logger.info(f"{'='*80}")
    logger.info(f"  Avg MIL overlap:     {mil_diff:+.3f}")
    logger.info(f"  Avg CNN overlap:     {cnn_diff:+.3f}")
    logger.info(f"  Aligned count:       {aligned_diff:+d}")
    logger.info(f"{'='*80}\n")
    
    return {
        'mil_diff': mil_diff,
        'cnn_diff': cnn_diff,
        'aligned_diff': aligned_diff
    }


def interpret_results(baseline_stats: Dict, roi_stats: Dict, differences: Dict):
    """Provide scientific interpretation of comparison results."""
    logger.info(f"{'='*80}")
    logger.info("SCIENTIFIC INTERPRETATION")
    logger.info(f"{'='*80}\n")
    
    roi_mil_better = roi_stats['avg_mil_overlap'] > baseline_stats['avg_mil_overlap']
    roi_aligned_better = roi_stats['aligned_count'] > baseline_stats['aligned_count']
    
    if roi_mil_better or roi_aligned_better:
        logger.info("✓ ROI variant shows stronger interpretability alignment.")
        if roi_mil_better:
            logger.info(f"  - ROI has higher average MIL overlap ({roi_stats['avg_mil_overlap']:.3f} vs {baseline_stats['avg_mil_overlap']:.3f})")
        if roi_aligned_better:
            logger.info(f"  - ROI has more aligned patients ({roi_stats['aligned_count']} vs {baseline_stats['aligned_count']})")
    else:
        logger.info("✗ No interpretability gain observed from ROI variant.")
        if not roi_mil_better:
            logger.info(f"  - Baseline has higher average MIL overlap ({baseline_stats['avg_mil_overlap']:.3f} vs {roi_stats['avg_mil_overlap']:.3f})")
        if not roi_aligned_better:
            logger.info(f"  - Baseline has more or equal aligned patients ({baseline_stats['aligned_count']} vs {roi_stats['aligned_count']})")
    
    logger.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare interpretability between baseline and ROI MIL variants',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--baseline_summary',
        type=str,
        default=str(BASELINE_SUMMARY),
        help=f'Path to baseline combined_summary.json (default: {BASELINE_SUMMARY})'
    )
    parser.add_argument(
        '--roi_summary',
        type=str,
        default=str(ROI_SUMMARY),
        help=f'Path to ROI combined_summary.json (default: {ROI_SUMMARY})'
    )
    
    args = parser.parse_args()
    
    # Load summaries
    logger.info("Loading summary files...")
    baseline_summary = load_summary(Path(args.baseline_summary))
    roi_summary = load_summary(Path(args.roi_summary))
    
    if baseline_summary is None:
        logger.error("Failed to load baseline summary. Exiting.")
        return 1
    
    if roi_summary is None:
        logger.error("Failed to load ROI summary. Exiting.")
        return 1
    
    # Extract metrics
    logger.info("Extracting metrics...")
    baseline_metrics = extract_metrics(baseline_summary)
    roi_metrics = extract_metrics(roi_summary)
    
    # Compute statistics
    baseline_stats = compute_statistics(baseline_metrics)
    roi_stats = compute_statistics(roi_metrics)
    
    # Print comparison
    differences = print_comparison(baseline_stats, roi_stats)
    
    # Scientific interpretation
    interpret_results(baseline_stats, roi_stats, differences)
    
    # Save comparison results
    comparison_results = {
        'baseline': baseline_stats,
        'roi': roi_stats,
        'differences': differences,
        'interpretation': {
            'roi_mil_better': roi_stats['avg_mil_overlap'] > baseline_stats['avg_mil_overlap'],
            'roi_aligned_better': roi_stats['aligned_count'] > baseline_stats['aligned_count'],
            'conclusion': 'ROI variant shows stronger interpretability alignment' if 
                         (roi_stats['avg_mil_overlap'] > baseline_stats['avg_mil_overlap'] or 
                          roi_stats['aligned_count'] > baseline_stats['aligned_count']) 
                         else 'No interpretability gain observed from ROI variant'
        }
    }
    
    output_path = PROJECT_ROOT / 'ensemble' / 'results' / 'interpretability' / 'variant_comparison.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    logger.info(f"✓ Saved comparison results: {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

