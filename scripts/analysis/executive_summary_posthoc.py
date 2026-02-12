#!/usr/bin/env python3
"""
Executive Summary: Post-Hoc Improvement of Swin-1

This script generates a final executive summary combining results from:
- Part A: Uncertainty-aware thresholding
- Part B: Feature-level rescue

It provides a clear GO/NO-GO decision based on target constraints.
"""

import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

THRESHOLDING_RESULTS = PROJECT_ROOT / 'ensemble' / 'results' / 'posthoc_thresholding' / 'thresholding_results.json'
RESCUE_RESULTS = PROJECT_ROOT / 'ensemble' / 'results' / 'feature_rescue' / 'rescue_results.json'
OUTPUT_DIR = PROJECT_ROOT / 'ensemble' / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FN_MAX = 10
TARGET_FP_MAX = 10
TARGET_PRECISION_MIN = 0.90
TARGET_RECALL_MIN = 0.90


def load_results():
    """Load results from both parts."""
    thresholding = {}
    rescue = {}
    
    if THRESHOLDING_RESULTS.exists():
        with open(THRESHOLDING_RESULTS, 'r') as f:
            thresholding = json.load(f)
    
    if RESCUE_RESULTS.exists():
        with open(RESCUE_RESULTS, 'r') as f:
            rescue = json.load(f)
    
    return thresholding, rescue


def find_best_method(all_results: dict) -> tuple:
    """Find best method that meets all constraints."""
    best_method = None
    best_agg = None
    
    for method_name, method_data in all_results.items():
        agg = method_data.get('aggregated', {})
        if agg.get('meets_all_constraints', False):
            if best_method is None or agg.get('fn_mean', 999) < best_agg.get('fn_mean', 999):
                best_method = method_name
                best_agg = agg
    
    return best_method, best_agg


def main():
    thresholding, rescue = load_results()
    
    # Combine all results
    all_results = {}
    all_results.update(thresholding)
    all_results.update(rescue)
    
    # Find best method
    best_method, best_agg = find_best_method(all_results)
    
    # Generate executive summary
    summary = f"""# Executive Summary: Post-Hoc Improvement of Swin-1

**Date:** 2026-02-10  
**Objective:** Improve Swin-1 performance using ONLY post-hoc decision logic and lightweight feature-based rescue, WITHOUT retraining Swin-1.

---

## Target Constraints

All constraints must be met **simultaneously**:

- **FN < {TARGET_FN_MAX}** (FN < 5 is excellent)
- **FP < {TARGET_FP_MAX}**
- **Precision ≥ {TARGET_PRECISION_MIN}**
- **Recall ≥ {TARGET_RECALL_MIN}**

---

## Methods Evaluated

### Part A: Uncertainty-Aware Thresholding
1. **Baseline** (threshold=0.5)
2. **Reject-band policy** (prob in [0.35, 0.65] → HGG)
3. **Confidence-aware thresholding** (entropy-based)
4. **Fold-specific calibrated threshold**

### Part B: Feature-Level Rescue
1. **Rule-based rescue** (flip LGG→HGG based on high-risk features)
2. **Lightweight logistic regression** (if rule-based helps)

---

## Results

"""
    
    if best_method and best_agg:
        summary += f"""## ✅ SUCCESS: Target Constraints Achieved

**Best Method:** {best_agg.get('method', best_method)}

### Performance Metrics

- **FN:** {best_agg.get('fn_mean', 0):.1f} ± {best_agg.get('fn_std', 0):.1f} (target: <{TARGET_FN_MAX})
- **FP:** {best_agg.get('fp_mean', 0):.1f} ± {best_agg.get('fp_std', 0):.1f} (target: <{TARGET_FP_MAX})
- **Precision:** {best_agg.get('precision_mean', 0):.4f} ± {best_agg.get('precision_std', 0):.4f} (target: ≥{TARGET_PRECISION_MIN})
- **Recall:** {best_agg.get('recall_mean', 0):.4f} ± {best_agg.get('recall_std', 0):.4f} (target: ≥{TARGET_RECALL_MIN})

"""
        if best_agg.get('fn_excellent', False):
            summary += "### ✅ **EXCELLENT: FN < 5 achieved!**\n\n"
        else:
            summary += f"### ⚠️ FN < 5 not achieved (FN < {TARGET_FN_MAX} is acceptable)\n\n"
        
        summary += "## Recommendation\n\n"
        summary += f"**GO:** Proceed with {best_agg.get('method', best_method)} for Swin-1 post-hoc improvement.\n\n"
        summary += "This method achieves all target constraints and can be deployed without retraining Swin-1.\n"
    else:
        summary += """## ❌ NO METHOD MEETS ALL CONSTRAINTS

### Conclusion

None of the evaluated post-hoc methods achieve:
- FN < 10 AND
- FP < 10 AND
- Precision ≥ 0.90 AND
- Recall ≥ 0.90

**All methods evaluated:**
"""
        for method_name, method_data in all_results.items():
            agg = method_data.get('aggregated', {})
            summary += f"- **{agg.get('method', method_name)}:** "
            summary += f"FN={agg.get('fn_mean', 0):.1f}, FP={agg.get('fp_mean', 0):.1f}, "
            summary += f"Precision={agg.get('precision_mean', 0):.4f}, Recall={agg.get('recall_mean', 0):.4f}\n"
        
        summary += "\n### Recommendation\n\n"
        summary += "**NO-GO:** Post-hoc methods alone cannot achieve target constraints.\n\n"
        summary += "**Next Steps:**\n"
        summary += "1. Consider model retraining (e.g., Swin-2 with Focal Loss and hard example mining)\n"
        summary += "2. Evaluate ensemble methods (combining Swin-1 with other models)\n"
        summary += "3. Consider additional data or data augmentation\n"
        summary += "4. Review target constraints (may be too strict for current dataset size)\n"
    
    summary += "\n---\n\n"
    summary += "*Generated: 2026-02-10*  \n"
    summary += "*Evaluation: Strict 5-fold OOF (no data leakage)*\n"
    
    # Save summary
    summary_path = OUTPUT_DIR / 'posthoc_improvement_executive_summary.md'
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(summary)
    print(f"\n✓ Saved executive summary to: {summary_path}")


if __name__ == '__main__':
    main()

