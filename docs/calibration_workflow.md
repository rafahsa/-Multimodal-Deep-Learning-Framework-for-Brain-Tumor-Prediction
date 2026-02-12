# Probability Calibration and Threshold Re-selection Workflow

## Overview

This document describes the probability calibration workflow for the ensemble meta-learner. Calibration improves the reliability of predicted probabilities, and threshold re-selection on calibrated probabilities can lead to better operating points.

## Scientific Design

### Data Split Strategy

To prevent data leakage, we use a **stratified split** of the OOF (out-of-fold) predictions:

- **70% Calibration Set**: Used to fit the calibration model (Platt scaling or Isotonic regression)
- **30% Threshold Selection Set**: Used exclusively for threshold sweep and evaluation

This ensures that:
1. Calibration is trained on independent data
2. Threshold selection is performed on held-out data
3. All metrics (Brier score, ECE, classification metrics) are computed on the threshold selection set

### Calibration Methods

- **Platt Scaling (sigmoid)**: Parametric method, works well with limited data
- **Isotonic Regression**: Non-parametric method, more flexible but requires more data
- **None**: Baseline (uncalibrated probabilities) for comparison

## Usage

### Basic Usage (Platt Calibration)

```bash
python scripts/ensemble/calibrate_and_sweep_thresholds.py --calibration platt
```

### Sanity Check (No Calibration)

```bash
python scripts/ensemble/calibrate_and_sweep_thresholds.py --calibration none
```

This should produce threshold recommendations similar to the original uncalibrated analysis.

### Isotonic Calibration

```bash
python scripts/ensemble/calibrate_and_sweep_thresholds.py --calibration isotonic
```

### Custom Parameters

```bash
python scripts/ensemble/calibrate_and_sweep_thresholds.py \
    --calibration platt \
    --split-seed 42 \
    --calibration-fraction 0.70 \
    --recall-target 0.94 \
    --sweep-start 0.05 \
    --sweep-end 0.95 \
    --sweep-step 0.01 \
    --n-bins 10 \
    --save-calibrator
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--calibration` | `{none,platt,isotonic}` | `platt` | Calibration method |
| `--split-seed` | `int` | `42` | Random seed for data split |
| `--calibration-fraction` | `float` | `0.70` | Fraction for calibration (30% held out) |
| `--recall-target` | `float` | `0.94` | Target recall for high-sensitivity threshold |
| `--sweep-start` | `float` | `0.05` | Start threshold for sweep |
| `--sweep-end` | `float` | `0.95` | End threshold for sweep |
| `--sweep-step` | `float` | `0.01` | Step size for threshold sweep |
| `--n-bins` | `int` | `10` | Number of bins for calibration curve/ECE |
| `--out-root` | `str` | `ensemble/results/calibration` | Root directory for outputs |
| `--save-calibrator` | `flag` | `False` | Save calibrator joblib file |
| `--plot-format` | `{png}` | `png` | Plot format |

## Output Structure

Each run creates a timestamped directory:

```
ensemble/results/calibration/
  {timestamp}_{calibration_mode}_seed{seed}/
    calibration_summary.json          # Brier scores, ECE, split info
    reliability_diagram_{mode}.png    # Calibration curve plot
    threshold_sweep_{mode}.json       # Full threshold sweep results
    recommended_thresholds_{mode}.json # Selected thresholds (balanced & high-sensitivity)
    calibrator_{mode}.joblib          # Saved calibrator (if --save-calibrator)
```

### Example Output Directory

```
ensemble/results/calibration/
  2026-02-08_14-30-15_platt_seed42/
    calibration_summary.json
    reliability_diagram_platt.png
    threshold_sweep_platt.json
    recommended_thresholds_platt.json
    calibrator_platt.joblib
```

## Output Files

### `calibration_summary.json`

Contains:
- Timestamp and calibration mode
- Split information (sizes, seed)
- Brier scores (pre/post calibration)
- ECE (Expected Calibration Error, pre/post)
- Improvement metrics
- Arguments used

### `reliability_diagram_{mode}.png`

Visualization showing:
- Uncalibrated calibration curve
- Calibrated calibration curve
- Perfect calibration line (diagonal)
- Closer to diagonal = better calibration

### `threshold_sweep_{mode}.json`

Array of results for each threshold in the sweep:
- Threshold value
- Confusion matrix (TN, FP, FN, TP)
- Precision, Recall, F1, Accuracy

### `recommended_thresholds_{mode}.json`

Selected thresholds using two policies:
1. **Balanced**: Maximizes F1 score
2. **High-sensitivity**: Maximum Precision subject to Recall ≥ target (default 0.94)

Each includes full metrics at the selected threshold.

## Metrics Explained

### Brier Score
- **Range**: 0 (perfect) to 1 (worst)
- **Formula**: `BS = mean((y_true - y_proba)^2)`
- **Interpretation**: Lower is better. Measures probability accuracy.

### Expected Calibration Error (ECE)
- **Range**: 0 (perfect) to 1 (worst)
- **Formula**: `ECE = sum(|acc_bin - conf_bin| * n_bin) / N`
- **Interpretation**: Lower is better. Measures how well predicted probabilities match observed frequencies.

### Reliability Diagram
- Visual representation of calibration
- X-axis: Mean predicted probability (confidence)
- Y-axis: Fraction of positives (actual frequency)
- Perfect calibration: Points on diagonal line

## Threshold Selection Policies

### Policy A: Balanced (Maximize F1)
- Selects threshold that maximizes F1 score
- Balanced trade-off between Precision and Recall
- Recommended for general use

### Policy B: High-Sensitivity (Recall ≥ Target)
- Among thresholds meeting `Recall ≥ recall_target` (default 0.94)
- Selects the one with maximum Precision
- Fallback: If no threshold meets target, uses highest Recall
- Recommended when minimizing False Negatives is critical

## Validation

### Sanity Check

Run with `--calibration none` to verify:
- Threshold recommendations are similar to original uncalibrated analysis
- No data leakage (split is working correctly)
- Script runs without errors

```bash
python scripts/ensemble/calibrate_and_sweep_thresholds.py --calibration none
```

## Important Notes

1. **Non-destructive**: This script does NOT modify existing files. All outputs are saved to timestamped directories.

2. **Data Leakage Prevention**: Calibration and threshold selection use disjoint subsets of OOF data.

3. **Reproducibility**: Use `--split-seed` to ensure reproducible splits across runs.

4. **Inference Integration**: Calibration in inference is deferred. The calibrator can be loaded later if needed (see `calibrator_{mode}.joblib`).

5. **Backward Compatibility**: Default behavior uses Platt calibration. Use `--calibration none` for uncalibrated baseline.

## Troubleshooting

### Missing Files
- Ensure `ensemble/oof_predictions/merged_oof_predictions.csv` exists
- Ensure `ensemble/models/meta_learner_logistic_regression.joblib` exists
- Run `train_meta_learner.py` first if needed

### Low Sample Size
- If threshold selection set is too small (< 50 samples), results may be unstable
- Consider adjusting `--calibration-fraction` (e.g., 0.65 for more threshold samples)

### Calibration Not Improving Metrics
- Check reliability diagram: if already well-calibrated, improvement may be minimal
- Try isotonic calibration if Platt shows no improvement
- Verify split is working correctly (check class distributions in logs)

## References

- Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning.
- Guo, C., et al. (2017). On calibration of modern neural networks.

