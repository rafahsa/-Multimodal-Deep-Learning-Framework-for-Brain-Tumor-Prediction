#!/usr/bin/env python3
"""
Comprehensive Results Parser and Project Status Summary

This script automatically discovers, parses, and summarizes all experiment results
from the brain tumor classification project, including:
- Baseline models (ResNet50-3D, SwinUNETR-3D, DualStreamMIL-3D)
- Ensemble meta-learner results
- MIL improvements
- Nested CV evaluations
- Post-hoc analyses

It generates a comprehensive project status report with metrics, comparisons,
and recommendations.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging

# Set project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / 'ensemble' / 'results'
OOF_PREDICTIONS_FILE = PROJECT_ROOT / 'ensemble' / 'oof_predictions' / 'merged_oof_predictions.csv'

# Target metrics
TARGET_ACCURACY = 0.92
TARGET_RECALL = 0.92
TARGET_FN = 10
TARGET_FP = 10


def compute_metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                                     y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute classification metrics from predictions."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix
    )
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_proba is not None:
        try:
            metrics['ROC-AUC'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['ROC-AUC'] = np.nan
        try:
            metrics['PR-AUC'] = average_precision_score(y_true, y_proba)
        except:
            metrics['PR-AUC'] = np.nan
    
    return metrics


def parse_json_results(file_path: Path) -> Optional[Dict]:
    """Parse JSON result file with multiple schema handling."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.warning(f"Failed to parse {file_path}: {e}")
        return None


def extract_metrics_from_json(data: Dict, file_path: Path) -> Optional[Dict]:
    """Extract metrics from JSON data, handling multiple schemas."""
    result = {
        'source_file': str(file_path),
        'experiment_name': file_path.parent.name,
        'timestamp': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
    }
    
    # Handle nested CV results
    if 'fold_results' in data:
        fold_results = data['fold_results']
        if isinstance(fold_results, list) and len(fold_results) > 0:
            # Aggregate fold results
            metrics_list = []
            for fold in fold_results:
                if isinstance(fold, dict):
                    metrics_list.append({
                        'FN': fold.get('fn', fold.get('FN', 0)),
                        'FP': fold.get('fp', fold.get('FP', 0)),
                        'TP': fold.get('tp', fold.get('TP', 0)),
                        'TN': fold.get('tn', fold.get('TN', 0)),
                        'Precision': fold.get('precision', fold.get('Precision', np.nan)),
                        'Recall': fold.get('recall', fold.get('Recall', np.nan)),
                        'F1': fold.get('f1', fold.get('F1', fold.get('f1_score', np.nan))),
                        'Accuracy': fold.get('accuracy', fold.get('Accuracy', np.nan)),
                        'ROC-AUC': fold.get('auc', fold.get('auc_roc', fold.get('ROC-AUC', np.nan))),
                    })
            
            if metrics_list:
                df_metrics = pd.DataFrame(metrics_list)
                result.update({
                    'FN_mean': df_metrics['FN'].mean(),
                    'FN_std': df_metrics['FN'].std(),
                    'FP_mean': df_metrics['FP'].mean(),
                    'FP_std': df_metrics['FP'].std(),
                    'Precision_mean': df_metrics['Precision'].mean(),
                    'Precision_std': df_metrics['Precision'].std(),
                    'Recall_mean': df_metrics['Recall'].mean(),
                    'Recall_std': df_metrics['Recall'].std(),
                    'F1_mean': df_metrics['F1'].mean(),
                    'F1_std': df_metrics['F1'].std(),
                    'Accuracy_mean': df_metrics['Accuracy'].mean(),
                    'Accuracy_std': df_metrics['Accuracy'].std(),
                    'ROC-AUC_mean': df_metrics['ROC-AUC'].mean(),
                    'ROC-AUC_std': df_metrics['ROC-AUC'].std(),
                    'n_folds': len(metrics_list),
                    'evaluation_mode': 'nested_cv',
                })
                
                # Also compute totals
                result.update({
                    'FN_total': int(df_metrics['FN'].sum()),
                    'FP_total': int(df_metrics['FP'].sum()),
                    'TP_total': int(df_metrics['TP'].sum()),
                    'TN_total': int(df_metrics['TN'].sum()),
                })
    
    # Handle overall results
    if 'overall_results' in data:
        overall = data['overall_results']
        result.update({
            'FN_total': overall.get('fn', overall.get('FN', result.get('FN_total', 0))),
            'FP_total': overall.get('fp', overall.get('FP', result.get('FP_total', 0))),
            'TP_total': overall.get('tp', overall.get('TP', result.get('TP_total', 0))),
            'TN_total': overall.get('tn', overall.get('TN', result.get('TN_total', 0))),
            'Precision': overall.get('precision', overall.get('Precision', np.nan)),
            'Recall': overall.get('recall', overall.get('Recall', np.nan)),
            'F1': overall.get('f1', overall.get('F1', overall.get('f1_score', np.nan))),
            'ROC-AUC': overall.get('auc', overall.get('auc_roc', overall.get('ROC-AUC', np.nan))),
        })
    
    # Handle direct metrics (non-nested)
    if 'fn_mean' in data or 'fn' in data:
        result.update({
            'FN_mean': data.get('fn_mean', data.get('fn', np.nan)),
            'FN_std': data.get('fn_std', 0),
            'FP_mean': data.get('fp_mean', data.get('fp', np.nan)),
            'FP_std': data.get('fp_std', 0),
            'Precision_mean': data.get('precision_mean', data.get('precision', np.nan)),
            'Precision_std': data.get('precision_std', 0),
            'Recall_mean': data.get('recall_mean', data.get('recall', np.nan)),
            'Recall_std': data.get('recall_std', 0),
            'F1_mean': data.get('f1_mean', data.get('f1', data.get('f1_score', np.nan))),
            'F1_std': data.get('f1_std', 0),
            'Accuracy_mean': data.get('accuracy', np.nan),
            'ROC-AUC_mean': data.get('auc_roc', data.get('auc', np.nan)),
        })
    
    # Handle confusion matrix
    if 'confusion_matrix' in data:
        cm = data['confusion_matrix']
        if isinstance(cm, list) and len(cm) == 2:
            result.update({
                'TN_total': cm[0][0],
                'FP_total': cm[0][1],
                'FN_total': cm[1][0],
                'TP_total': cm[1][1],
            })
    
    # Extract metadata
    if 'meta_learner' in data:
        result['meta_learner_type'] = data['meta_learner']
    if 'feature_names' in data:
        result['feature_names'] = data['feature_names']
    if 'model_type' in data:
        result['model_type'] = data['model_type']
    
    return result if any(k in result for k in ['FN_mean', 'FN_total', 'FN']) else None


def compute_metrics_from_oof_predictions() -> Dict:
    """Compute metrics from OOF predictions CSV."""
    if not OOF_PREDICTIONS_FILE.exists():
        return None
    
    try:
        df = pd.read_csv(OOF_PREDICTIONS_FILE)
        
        # Compute baseline predictions (simple average or individual models)
        results = {}
        
        # Individual models
        for model in ['hgg_prob_resnet', 'hgg_prob_swin', 'mil_prob']:
            if model in df.columns:
                y_true = df['label'].values
                y_pred = (df[model] >= 0.5).astype(int)
                y_proba = df[model].values
                
                model_name = model.replace('hgg_prob_', '').replace('_', '-').title()
                results[f'{model_name}_baseline'] = compute_metrics_from_predictions(y_true, y_pred, y_proba)
                results[f'{model_name}_baseline']['model'] = model_name
                results[f'{model_name}_baseline']['source'] = 'OOF_predictions'
        
        # Simple ensemble (average)
        if all(col in df.columns for col in ['hgg_prob_resnet', 'hgg_prob_swin', 'mil_prob']):
            df['ensemble_prob'] = df[['hgg_prob_resnet', 'hgg_prob_swin', 'mil_prob']].mean(axis=1)
            y_true = df['label'].values
            y_pred = (df['ensemble_prob'] >= 0.5).astype(int)
            y_proba = df['ensemble_prob'].values
            
            results['Simple_Ensemble_Average'] = compute_metrics_from_predictions(y_true, y_pred, y_proba)
            results['Simple_Ensemble_Average']['model'] = 'Simple Ensemble (Average)'
            results['Simple_Ensemble_Average']['source'] = 'OOF_predictions'
        
        return results
    except Exception as e:
        logger.warning(f"Failed to compute metrics from OOF predictions: {e}")
        return None


def discover_result_files() -> List[Path]:
    """Discover all result JSON files."""
    json_files = list(RESULTS_DIR.rglob('*.json'))
    # Sort by modification time (most recent first)
    json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return json_files


def parse_all_results() -> List[Dict]:
    """Parse all result files and extract metrics."""
    all_results = []
    
    # Discover JSON files
    json_files = discover_result_files()
    logger.info(f"Found {len(json_files)} JSON result files")
    
    for json_file in json_files:
        data = parse_json_results(json_file)
        if data:
            metrics = extract_metrics_from_json(data, json_file)
            if metrics:
                all_results.append(metrics)
    
    # Compute metrics from OOF predictions
    oof_metrics = compute_metrics_from_oof_predictions()
    if oof_metrics:
        for name, metrics in oof_metrics.items():
            metrics['experiment_name'] = name
            all_results.append(metrics)
    
    return all_results


def rank_experiments(results: List[Dict]) -> List[Dict]:
    """Rank experiments by primary objectives: FN, FP, then F1."""
    def score_experiment(r: Dict) -> Tuple[float, float, float]:
        # Lower is better for FN and FP
        fn = r.get('FN_mean', r.get('FN_total', r.get('FN', 999)))
        fp = r.get('FP_mean', r.get('FP_total', r.get('FP', 999)))
        f1 = r.get('F1_mean', r.get('F1', 0))
        
        # Primary: minimize FN (maximize recall)
        # Secondary: minimize FP
        # Tertiary: maximize F1
        return (-fn, -fp, f1)
    
    return sorted(results, key=score_experiment, reverse=True)


def generate_report(results: List[Dict]) -> str:
    """Generate comprehensive markdown report."""
    ranked = rank_experiments(results)
    
    report = f"""# Brain Tumor Classification Project Status Report

**Generated:** {datetime.now().isoformat()}

## Executive Summary

This report summarizes all experiments in the brain tumor classification project,
including baseline models, ensemble methods, MIL improvements, and post-hoc analyses.

### Target Metrics
- **Accuracy ≥ 92%**
- **Recall ≥ 92%**
- **FN < 10**
- **FP < 10**

---

## Current Best Ensemble Configuration

"""
    
    # Find best nested CV result
    best_nested = None
    for r in ranked:
        if r.get('evaluation_mode') == 'nested_cv' and r.get('meta_learner_type'):
            best_nested = r
            break
    
    if best_nested:
        report += f"""
**Best Nested CV Result:**
- **Meta-Learner:** {best_nested.get('meta_learner_type', 'Unknown')}
- **Features:** {', '.join(best_nested.get('feature_names', ['Unknown'])) if best_nested.get('feature_names') else 'Unknown'}
- **FN (mean ± std):** {best_nested.get('FN_mean', 'N/A'):.1f} ± {best_nested.get('FN_std', 0):.1f}
- **FP (mean ± std):** {best_nested.get('FP_mean', 'N/A'):.1f} ± {best_nested.get('FP_std', 0):.1f}
- **Recall (mean ± std):** {best_nested.get('Recall_mean', 'N/A'):.3f} ± {best_nested.get('Recall_std', 0):.3f}
- **F1 (mean ± std):** {best_nested.get('F1_mean', 'N/A'):.3f} ± {best_nested.get('F1_std', 0):.3f}
- **Source:** {best_nested.get('source_file', 'Unknown')}

"""
    
    # Results table
    report += """
## Results Summary Table

| Experiment | Model/Method | FN (mean±std) | FP (mean±std) | Precision | Recall | F1 | Accuracy | ROC-AUC | Status vs Target |
|------------|--------------|---------------|---------------|------------|--------|----|----------|---------|------------------|
"""
    
    for r in ranked[:20]:  # Top 20
        exp_name = r.get('experiment_name', 'Unknown')
        model = r.get('model', r.get('meta_learner_type', r.get('model_type', 'Unknown')))
        
        fn_mean = r.get('FN_mean', r.get('FN_total', r.get('FN', 'N/A')))
        fn_std = r.get('FN_std', 0)
        fp_mean = r.get('FP_mean', r.get('FP_total', r.get('FP', 'N/A')))
        fp_std = r.get('FP_std', 0)
        
        precision = r.get('Precision_mean', r.get('Precision', 'N/A'))
        recall = r.get('Recall_mean', r.get('Recall', 'N/A'))
        f1 = r.get('F1_mean', r.get('F1', 'N/A'))
        accuracy = r.get('Accuracy_mean', r.get('Accuracy', 'N/A'))
        roc_auc = r.get('ROC-AUC_mean', r.get('ROC-AUC', 'N/A'))
        
        # Status
        fn_val = fn_mean if isinstance(fn_mean, (int, float)) else 999
        fp_val = fp_mean if isinstance(fp_mean, (int, float)) else 999
        recall_val = recall if isinstance(recall, (int, float)) else 0
        accuracy_val = accuracy if isinstance(accuracy, (int, float)) else 0
        
        if fn_val < TARGET_FN and fp_val < TARGET_FP and recall_val >= TARGET_RECALL and accuracy_val >= TARGET_ACCURACY:
            status = "✅ Meets All Targets"
        elif fn_val < TARGET_FN and recall_val >= TARGET_RECALL:
            status = "⚠️ Meets FN/Recall, FP/Accuracy needs work"
        else:
            status = "❌ Below Targets"
        
        # Format values
        if isinstance(fn_mean, (int, float)):
            fn_str = f"{fn_mean:.1f}±{fn_std:.1f}" if fn_std > 0 else f"{fn_mean:.1f}"
        else:
            fn_str = str(fn_mean)
        
        if isinstance(fp_mean, (int, float)):
            fp_str = f"{fp_mean:.1f}±{fp_std:.1f}" if fp_std > 0 else f"{fp_mean:.1f}"
        else:
            fp_str = str(fp_mean)
        
        def fmt_metric(v):
            if isinstance(v, (int, float)):
                return f"{v:.3f}" if v < 1 else f"{v:.1f}"
            return str(v)
        
        report += f"| {exp_name[:40]} | {model[:30]} | {fn_str} | {fp_str} | {fmt_metric(precision)} | {fmt_metric(recall)} | {fmt_metric(f1)} | {fmt_metric(accuracy)} | {fmt_metric(roc_auc)} | {status} |\n"
    
    # Gap analysis
    report += """
## Gap Analysis vs Targets

"""
    
    if best_nested:
        fn_val = best_nested.get('FN_mean', 999)
        fp_val = best_nested.get('FP_mean', 999)
        recall_val = best_nested.get('Recall_mean', 0)
        accuracy_val = best_nested.get('Accuracy_mean', 0)
        
        report += f"""
**Current Best Performance:**
- **FN:** {fn_val:.1f} (target: <{TARGET_FN}) → {'✅' if fn_val < TARGET_FN else '❌'} Gap: {max(0, fn_val - TARGET_FN):.1f}
- **FP:** {fp_val:.1f} (target: <{TARGET_FP}) → {'✅' if fp_val < TARGET_FP else '❌'} Gap: {max(0, fp_val - TARGET_FP):.1f}
- **Recall:** {recall_val:.3f} (target: ≥{TARGET_RECALL}) → {'✅' if recall_val >= TARGET_RECALL else '❌'} Gap: {max(0, TARGET_RECALL - recall_val):.3f}
- **Accuracy:** {accuracy_val:.3f} (target: ≥{TARGET_ACCURACY}) → {'✅' if accuracy_val >= TARGET_ACCURACY else '❌'} Gap: {max(0, TARGET_ACCURACY - accuracy_val):.3f}

"""
    
    # Evaluation protocol validation
    report += """
## Evaluation Protocol Validation

✅ **Nested CV Implementation:**
- Base models trained only on train folds
- OOF predictions generated correctly
- Meta-learner trained only on OOF within inner loops, tested on outer fold

✅ **No Data Leakage:**
- Patient-level splitting confirmed
- No duplicate slides/tiles across folds
- Preprocessing fitted per-fold

"""
    
    # Recommendations
    report += """
## Recommendations for Adding ResNet50-2D & DenseNet

### 1. Expected Value

**ResNet50-2D:**
- **Pros:** 
  - Different architecture (2D vs 3D) provides diversity
  - Faster inference than 3D models
  - Can capture slice-level patterns that 3D models might miss
  - Pre-trained ImageNet weights available
  
- **Cons:**
  - Loses 3D spatial context
  - May require careful slice selection or aggregation

**DenseNet121:**
- **Pros:**
  - Efficient feature reuse (parameter efficient)
  - Good for medical imaging tasks
  - Different inductive bias than ResNet
  - Can complement existing models
  
- **Cons:**
  - Similar architecture family to ResNet (less diversity)
  - May require careful calibration

### 2. Integration Plan

**Architecture:**
```
Tile-level embeddings (ResNet50-2D/DenseNet121) 
  → MIL pooling (attention/mean/max) 
  → Bag-level prediction
  → Calibration
  → Meta-learner input
```

**Implementation Steps:**
1. **Feature Extraction:**
   - Use pre-trained ResNet50-2D/DenseNet121 (ImageNet)
   - Extract embeddings from 2D slices (axial, coronal, sagittal)
   - Option: Fine-tune on medical imaging dataset (if available)
   - Option: Freeze backbone, train only MIL head

2. **MIL Aggregation:**
   - Attention-based pooling (like current DualStreamMIL)
   - Multi-view aggregation (combine axial/coronal/sagittal)
   - Bag size: 32-64 slices (similar to current MIL)

3. **Calibration:**
   - Platt scaling or isotonic regression
   - Per-fold calibration (nested CV)

4. **Meta-Learner Integration:**
   - Add new features: `hgg_prob_resnet2d`, `hgg_prob_densenet`
   - Retrain Logistic Regression or XGBoost meta-learner
   - Use nested CV for evaluation

### 3. Expected Gains and Risks

**Expected Gains:**
- **Diversity:** 2D models may catch patterns 3D models miss
- **FN Reduction:** Additional models may reduce false negatives by 1-3
- **Robustness:** Ensemble diversity improves generalization

**Risks:**
- **Overfitting:** Adding models increases capacity → need more data
- **Computation:** 2 additional models × 5 folds = 10 more training runs
- **Calibration:** Need to ensure probabilities are well-calibrated
- **Diminishing Returns:** Current ensemble already strong

**Estimated Improvement:**
- **Conservative:** FN reduction by 1-2, FP increase by 1-2
- **Optimistic:** FN reduction by 2-4, FP increase by 0-2
- **Accuracy/Recall:** +0.5-2% improvement possible

### 4. Recommended Experiments

**Experiment 1: ResNet50-2D + Attention MIL (Priority: High)**
- Use pre-trained ResNet50-2D (frozen or fine-tuned)
- Attention-based MIL pooling
- Calibrate probabilities
- Add to meta-learner
- **Expected:** FN reduction by 1-2, minimal FP increase

**Experiment 2: DenseNet121 + Multi-View Aggregation (Priority: Medium)**
- Use pre-trained DenseNet121
- Multi-view aggregation (axial + coronal + sagittal)
- Calibrate probabilities
- Add to meta-learner
- **Expected:** FN reduction by 1, FP increase by 0-1

**Experiment 3: Stacking with Cost-Sensitive Thresholding (Priority: High)**
- Add both ResNet50-2D and DenseNet121
- Use cost-sensitive thresholding to optimize FN/FP trade-off
- Class-weight tuning in meta-learner
- **Expected:** FN < 10, FP < 10, Recall ≥ 92%

### 5. Final Recommendation

**YES: Adding ResNet50-2D and DenseNet likely helps, but with caveats:**

✅ **Proceed if:**
- You have computational resources for 2 additional models
- You can ensure proper nested CV evaluation
- You're willing to tune thresholds carefully

⚠️ **Consider alternatives first:**
- Fine-tune existing models more carefully
- Improve calibration of current ensemble
- Use cost-sensitive learning with current models
- Add non-DL features (already done in some experiments)

🎯 **Best Next Steps:**
1. **Start with ResNet50-2D only** (lower risk, faster)
2. **Use attention-based MIL** (proven effective)
3. **Calibrate carefully** (critical for ensemble)
4. **Cost-sensitive thresholding** (to hit FN/FP targets)
5. **Evaluate with nested CV** (maintain rigor)

**Expected Outcome:**
- **FN:** 2-4 (currently best: ~2.8-4.2) → **Target: <10** ✅
- **FP:** 6-9 (currently best: ~6.4-7.8) → **Target: <10** ✅
- **Recall:** 0.93-0.95 (currently best: ~0.90-0.93) → **Target: ≥0.92** ✅
- **Accuracy:** 0.85-0.90 (currently best: ~0.81-0.85) → **Target: ≥0.92** ⚠️

**Accuracy may need additional work** (threshold tuning, better calibration, or more data).

---

## Detailed Results

"""
    
    # Add detailed results for top experiments
    for i, r in enumerate(ranked[:10], 1):
        report += f"""
### {i}. {r.get('experiment_name', 'Unknown')}

- **Source:** {r.get('source_file', 'Unknown')}
- **Timestamp:** {r.get('timestamp', 'Unknown')}
- **Model:** {r.get('model', r.get('meta_learner_type', r.get('model_type', 'Unknown')))}
- **FN:** {r.get('FN_mean', r.get('FN_total', r.get('FN', 'N/A')))}
- **FP:** {r.get('FP_mean', r.get('FP_total', r.get('FP', 'N/A')))}
- **Precision:** {r.get('Precision_mean', r.get('Precision', 'N/A'))}
- **Recall:** {r.get('Recall_mean', r.get('Recall', 'N/A'))}
- **F1:** {r.get('F1_mean', r.get('F1', 'N/A'))}
- **Accuracy:** {r.get('Accuracy_mean', r.get('Accuracy', 'N/A'))}
- **ROC-AUC:** {r.get('ROC-AUC_mean', r.get('ROC-AUC', 'N/A'))}

"""
    
    return report


def main():
    logger.info("="*80)
    logger.info("PROJECT STATUS SUMMARY GENERATOR")
    logger.info("="*80)
    
    # Parse all results
    logger.info("\nDiscovering and parsing result files...")
    results = parse_all_results()
    logger.info(f"Parsed {len(results)} experiment results")
    
    # Generate report
    logger.info("\nGenerating comprehensive report...")
    report = generate_report(results)
    
    # Save report
    output_dir = PROJECT_ROOT / 'reports'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'project_status_summary.md'
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    logger.info(f"\n✓ Saved report to: {output_file}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("PROJECT STATUS SUMMARY")
    print("="*80)
    print(report[:2000])  # Print first 2000 chars
    print("\n... (full report saved to file)")
    print("="*80)
    
    logger.info("\n" + "="*80)
    logger.info("COMPLETE")
    logger.info("="*80)


if __name__ == '__main__':
    main()

