#!/usr/bin/env python3
"""
Analyze Base Model Contributions to Ensemble Decision

This script analyzes the contribution of each base model to the final ensemble
decision by extracting and interpreting meta-learner coefficients.

It analyzes:
1. The saved meta-learner model (deployed version)
2. The best nested CV configuration (if available)
3. Provides interpretation suitable for papers/thesis
"""

import sys
from pathlib import Path

# Set project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
MODEL_FILE = PROJECT_ROOT / 'ensemble' / 'models' / 'meta_learner_logistic_regression.joblib'
METRICS_FILE = PROJECT_ROOT / 'ensemble' / 'results' / 'meta_learner_metrics.json'
ENHANCED_RESULTS_FILE = PROJECT_ROOT / 'ensemble' / 'results' / 'nested_cv_meta_features' / 'meta_features_results_20260209_005859.json'
OUTPUT_DIR = PROJECT_ROOT / 'reports'
OUTPUT_DIR.mkdir(exist_ok=True)

# Base model mapping
BASE_MODEL_MAPPING = {
    'hgg_prob_resnet': 'ResNet50-3D',
    'p_resnet': 'ResNet50-3D',
    'hgg_prob_swin': 'SwinUNETR-3D',
    'p_swin': 'SwinUNETR-3D',
    'hgg_prob_mil': 'DualStreamMIL-3D',
    'p_mil': 'DualStreamMIL-3D',
    'mil_prob': 'DualStreamMIL-3D',
}


def load_saved_model() -> Tuple[Optional[object], Optional[Dict]]:
    """Load the saved meta-learner model and extract coefficients."""
    if not MODEL_FILE.exists():
        logger.warning(f"Model file not found: {MODEL_FILE}")
        return None, None
    
    try:
        model = joblib.load(MODEL_FILE)
        logger.info(f"✓ Loaded model from: {MODEL_FILE}")
        
        # Extract coefficients
        coef = model.coef_[0] if hasattr(model, 'coef_') else None
        intercept = model.intercept_[0] if hasattr(model, 'intercept_') else None
        
        # Try to get feature names
        feature_names = None
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
        elif hasattr(model, 'coef_') and len(coef) == 3:
            # Assume standard order: ResNet, Swin, MIL
            feature_names = ['hgg_prob_resnet', 'hgg_prob_swin', 'hgg_prob_mil']
        
        model_info = {
            'coefficients': coef.tolist() if coef is not None and hasattr(coef, 'tolist') else (list(coef) if coef is not None else None),
            'intercept': float(intercept) if intercept is not None else None,
            'feature_names': feature_names.tolist() if feature_names is not None and hasattr(feature_names, 'tolist') else (list(feature_names) if feature_names is not None else None),
            'n_features': len(coef) if coef is not None else None,
        }
        
        return model, model_info
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None


def load_metrics_file() -> Optional[Dict]:
    """Load metrics file with coefficients."""
    if not METRICS_FILE.exists():
        logger.warning(f"Metrics file not found: {METRICS_FILE}")
        return None
    
    try:
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
        logger.info(f"✓ Loaded metrics from: {METRICS_FILE}")
        return metrics
    except Exception as e:
        logger.error(f"Failed to load metrics: {e}")
        return None


def load_enhanced_results() -> Optional[Dict]:
    """Load enhanced meta-features results."""
    if not ENHANCED_RESULTS_FILE.exists():
        logger.warning(f"Enhanced results file not found: {ENHANCED_RESULTS_FILE}")
        return None
    
    try:
        with open(ENHANCED_RESULTS_FILE, 'r') as f:
            results = json.load(f)
        logger.info(f"✓ Loaded enhanced results from: {ENHANCED_RESULTS_FILE}")
        return results
    except Exception as e:
        logger.error(f"Failed to load enhanced results: {e}")
        return None


def analyze_coefficients(coefficients: Dict, feature_names: List[str], 
                         intercept: float, source: str) -> Dict:
    """Analyze and interpret coefficients."""
    
    # Map feature names to base models
    base_model_coefs = {}
    meta_feature_coefs = {}
    
    for i, feat_name in enumerate(feature_names):
        coef_value = coefficients.get(feat_name, 0.0)
        
        # Check if it's a base model probability
        if feat_name in BASE_MODEL_MAPPING:
            model_name = BASE_MODEL_MAPPING[feat_name]
            base_model_coefs[model_name] = coef_value
        else:
            # It's a meta-feature
            meta_feature_coefs[feat_name] = coef_value
    
    # Compute absolute values for importance ranking
    base_model_importance = {k: abs(v) for k, v in base_model_coefs.items()}
    meta_feature_importance = {k: abs(v) for k, v in meta_feature_coefs.items()}
    
    # Rank by absolute value
    base_model_ranked = sorted(base_model_importance.items(), key=lambda x: x[1], reverse=True)
    meta_feature_ranked = sorted(meta_feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Interpretation
    analysis = {
        'source': source,
        'intercept': intercept,
        'base_model_coefficients': base_model_coefs,
        'base_model_importance_ranked': base_model_ranked,
        'meta_feature_coefficients': meta_feature_coefs if meta_feature_coefs else None,
        'meta_feature_importance_ranked': meta_feature_ranked if meta_feature_ranked else None,
        'strongest_model': base_model_ranked[0][0] if base_model_ranked else None,
        'weakest_model': base_model_ranked[-1][0] if base_model_ranked else None,
        'dominance_ratio': base_model_ranked[0][1] / base_model_ranked[-1][1] if len(base_model_ranked) > 1 and base_model_ranked[-1][1] > 0 else None,
    }
    
    return analysis


def generate_interpretation(analysis: Dict) -> str:
    """Generate human-readable interpretation."""
    interp = []
    
    interp.append("## Coefficient Interpretation\n")
    
    # Base model coefficients
    interp.append("### Base Model Contributions\n")
    
    base_coefs = analysis['base_model_coefficients']
    base_ranked = analysis['base_model_importance_ranked']
    
    for i, (model, importance) in enumerate(base_ranked, 1):
        coef_value = base_coefs[model]
        interp.append(f"{i}. **{model}**: Coefficient = {coef_value:.6f} (|coef| = {importance:.6f})")
        
        if coef_value > 0:
            interp.append(f"   - Positive influence: Higher {model} probability → Higher ensemble HGG probability")
        else:
            interp.append(f"   - Negative influence: Higher {model} probability → Lower ensemble HGG probability")
        
        if abs(coef_value) < 0.1:
            interp.append(f"   - ⚠️ **Very small coefficient**: Model has minimal influence")
        elif abs(coef_value) < 0.5:
            interp.append(f"   - **Moderate influence**: Model contributes but is not dominant")
        else:
            interp.append(f"   - **Strong influence**: Model is a key contributor to ensemble decision")
        
        interp.append("")
    
    # Dominance analysis
    if analysis['dominance_ratio']:
        ratio = analysis['dominance_ratio']
        strongest = analysis['strongest_model']
        weakest = analysis['weakest_model']
        
        interp.append(f"### Model Dominance Analysis\n")
        interp.append(f"- **Dominance Ratio**: {ratio:.2f}x (strongest / weakest)")
        interp.append(f"- **Strongest Model**: {strongest}")
        interp.append(f"- **Weakest Model**: {weakest}")
        
        if ratio > 5:
            interp.append(f"- **Interpretation**: {strongest} dominates the ensemble decision (high dominance)")
        elif ratio > 2:
            interp.append(f"- **Interpretation**: {strongest} has strong influence but other models still contribute (moderate dominance)")
        else:
            interp.append(f"- **Interpretation**: Models have complementary contributions (low dominance, high complementarity)")
        interp.append("")
    
    # Meta-features
    if analysis['meta_feature_coefficients']:
        interp.append("### Meta-Feature Contributions\n")
        meta_coefs = analysis['meta_feature_coefficients']
        meta_ranked = analysis['meta_feature_importance_ranked']
        
        for i, (feat, importance) in enumerate(meta_ranked[:5], 1):  # Top 5
            coef_value = meta_coefs[feat]
            interp.append(f"{i}. **{feat}**: Coefficient = {coef_value:.6f} (|coef| = {importance:.6f})")
        interp.append("")
    
    # Overall interpretation
    interp.append("### Overall Ensemble Behavior\n")
    
    all_positive = all(v > 0 for v in base_coefs.values())
    all_negative = all(v < 0 for v in base_coefs.values())
    
    if all_positive:
        interp.append("- All base models have **positive coefficients**: Ensemble combines models additively")
        interp.append("- Higher probabilities from any model increase ensemble HGG probability")
    elif all_negative:
        interp.append("- All base models have **negative coefficients**: Unusual pattern (check model configuration)")
    else:
        interp.append("- **Mixed coefficients**: Some models increase HGG probability, others decrease it")
        interp.append("- This suggests models may have complementary error patterns")
    
    interp.append("")
    
    return "\n".join(interp)


def generate_summary_table(analysis: Dict) -> str:
    """Generate summary table for paper/thesis."""
    
    base_coefs = analysis['base_model_coefficients']
    base_ranked = analysis['base_model_importance_ranked']
    
    table = "## Meta-Learner Coefficients Summary Table\n\n"
    table += "| Base Model | Coefficient | |Coefficient| | Rank | Interpretation |\n"
    table += "|------------|------------|--------------|------|----------------|\n"
    
    for i, (model, importance) in enumerate(base_ranked, 1):
        coef_value = base_coefs[model]
        
        # Interpretation
        if abs(coef_value) < 0.1:
            interp = "Minimal influence"
        elif abs(coef_value) < 0.5:
            interp = "Moderate influence"
        elif abs(coef_value) < 2.0:
            interp = "Strong influence"
        else:
            interp = "Dominant influence"
        
        sign = "+" if coef_value >= 0 else "-"
        table += f"| {model} | {coef_value:+.6f} | {importance:.6f} | {i} | {interp} |\n"
    
    table += f"\n**Intercept**: {analysis['intercept']:.6f}\n\n"
    
    return table


def main():
    logger.info("="*80)
    logger.info("ENSEMBLE CONTRIBUTION ANALYSIS")
    logger.info("="*80)
    
    # Load saved model
    logger.info("\n1. Loading saved meta-learner model...")
    model, model_info = load_saved_model()
    
    # Load metrics file
    logger.info("\n2. Loading metrics file...")
    metrics = load_metrics_file()
    
    # Load enhanced results
    logger.info("\n3. Loading enhanced meta-features results...")
    enhanced_results = load_enhanced_results()
    
    # Analyze saved model
    analyses = []
    
    if model_info and model_info['coefficients']:
        logger.info("\n4. Analyzing saved model coefficients...")
        
        # Create coefficients dict
        feature_names = model_info['feature_names']
        coef_values = model_info['coefficients']
        coefficients_dict = dict(zip(feature_names, coef_values))
        
        analysis = analyze_coefficients(
            coefficients_dict,
            feature_names,
            model_info['intercept'],
            "Saved Model (meta_learner_logistic_regression.joblib)"
        )
        analyses.append(analysis)
        
        logger.info(f"  ✓ Found {len(analysis['base_model_coefficients'])} base model coefficients")
        if analysis['meta_feature_coefficients']:
            logger.info(f"  ✓ Found {len(analysis['meta_feature_coefficients'])} meta-feature coefficients")
    
    # Analyze metrics file
    if metrics and 'model_coefficients' in metrics:
        logger.info("\n5. Analyzing metrics file coefficients...")
        
        coefficients_dict = metrics['model_coefficients']
        feature_names = list(coefficients_dict.keys())
        
        analysis = analyze_coefficients(
            coefficients_dict,
            feature_names,
            metrics.get('model_intercept', 0.0),
            "Metrics File (meta_learner_metrics.json)"
        )
        analyses.append(analysis)
    
    # Generate report
    logger.info("\n6. Generating analysis report...")
    
    report = "# Ensemble Base Model Contribution Analysis\n\n"
    report += f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n"
    report += "## Overview\n\n"
    report += "This report analyzes the contribution of each base model to the final ensemble decision "
    report += "by examining the meta-learner (Logistic Regression) coefficients.\n\n"
    report += "---\n\n"
    
    # Use the first (most recent) analysis
    if analyses:
        primary_analysis = analyses[0]
        
        report += generate_summary_table(primary_analysis)
        report += "\n---\n\n"
        report += generate_interpretation(primary_analysis)
        
        # Add note about enhanced meta-learner
        if enhanced_results:
            report += "\n---\n\n"
            report += "## Note on Enhanced Meta-Learner\n\n"
            report += "The best performing configuration uses an **Enhanced Meta-Learner** with meta-features "
            report += "(see `nested_cv_meta_features/meta_features_results_20260209_005859.json`). "
            report += "However, the deployed model file uses only the 3 base model probabilities.\n\n"
            report += "**Enhanced Meta-Learner Features:**\n"
            feature_names = enhanced_results.get('feature_names', [])
            for feat in feature_names:
                report += f"- {feat}\n"
            report += "\n"
            report += "The enhanced version includes probability statistics, margins, entropy, and argmax indicators "
            report += "in addition to base model probabilities, which improves performance but makes coefficient "
            report += "interpretation more complex.\n"
    
    # Save report
    output_file = OUTPUT_DIR / 'ensemble_contribution_analysis.md'
    with open(output_file, 'w') as f:
        f.write(report)
    
    logger.info(f"\n✓ Saved report to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("ENSEMBLE CONTRIBUTION SUMMARY")
    print("="*80)
    
    if analyses:
        analysis = analyses[0]
        print("\nBase Model Coefficients:")
        print("-" * 60)
        for model, coef in sorted(analysis['base_model_coefficients'].items(), 
                                  key=lambda x: abs(x[1]), reverse=True):
            print(f"  {model:20s}: {coef:+.6f} (|coef| = {abs(coef):.6f})")
        
        print(f"\nIntercept: {analysis['intercept']:.6f}")
        print(f"\nStrongest Model: {analysis['strongest_model']}")
        print(f"Weakest Model: {analysis['weakest_model']}")
        if analysis['dominance_ratio']:
            print(f"Dominance Ratio: {analysis['dominance_ratio']:.2f}x")
    
    print("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()

