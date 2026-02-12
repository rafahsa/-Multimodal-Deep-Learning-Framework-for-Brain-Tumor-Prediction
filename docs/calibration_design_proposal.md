# Probability Calibration Design Proposal

## Overview

This document proposes a design for adding optional probability calibration to the ensemble meta-learner, followed by threshold re-selection on calibrated probabilities. The design prioritizes **non-destructive changes**, **backward compatibility**, and **scientific rigor**.

---

## Decision 1: Calibration Data Split Strategy

### Proposed Approach: **Single Stratified Split (70% calibration, 30% threshold selection)**

**Rationale:**
- **Simplicity**: One split is easy to understand, implement, and reproduce
- **Sample size**: With ~285 OOF samples, 70/30 split gives:
  - ~200 samples for calibration training (sufficient for Platt/Isotonic)
  - ~85 samples for threshold selection (adequate for threshold sweep)
- **Scientific validity**: Calibration and threshold selection are on independent sets, preventing overfitting
- **Practical**: Avoids nested CV complexity while maintaining rigor

**Alternative Considered: Nested CV**
- **Pros**: More robust, uses all data
- **Cons**: More complex, harder to interpret, computationally heavier
- **Decision**: Reject - overkill for our sample size and use case

**Implementation:**
```python
from sklearn.model_selection import train_test_split

# Stratified split to preserve class distribution
X_cal, X_thresh, y_cal, y_thresh = train_test_split(
    X_oof, y_oof, 
    test_size=0.30, 
    stratify=y_oof, 
    random_state=42
)
```

**Recommendation: ✅ Single stratified split (70/30)**

---

## Decision 2: Calibration Method

### Proposed Approach: **sklearn CalibratedClassifierCV with method='sigmoid' (Platt) or 'isotonic'**

**Rationale:**
- **Standard practice**: sklearn's `CalibratedClassifierCV` is the standard approach
- **Works with existing model**: We can wrap the existing LogisticRegression meta-learner
- **Two options**: 
  - `method='sigmoid'` (Platt scaling): Parametric, works well with small data, fast
  - `method='isotonic'` (Isotonic regression): Non-parametric, more flexible, needs more data
- **No model modification**: Calibration is applied as a post-processing step, original model untouched

**Alternative Considered: Direct probability calibration**
- **Pros**: More control
- **Cons**: Non-standard, harder to maintain, reinvents the wheel
- **Decision**: Reject - use sklearn's proven implementation

**Implementation:**
```python
from sklearn.calibration import CalibratedClassifierCV

# Wrap existing meta-learner
calibrated_model = CalibratedClassifierCV(
    base_estimator=meta_learner,  # Already trained LogisticRegression
    method='sigmoid',  # or 'isotonic'
    cv='prefit'  # Use pre-fitted model
)
calibrated_model.fit(X_cal, y_cal)
```

**Recommendation: ✅ sklearn CalibratedClassifierCV (Platt/Isotonic)**

---

## Decision 3: Metrics & Plots

### Proposed Metrics:

1. **Brier Score** (required)
   - Lower is better (0 = perfect, 1 = worst)
   - Compare: `brier_pre` vs `brier_post`
   - Formula: `BS = mean((y_true - y_proba)^2)`

2. **Reliability Diagram** (required)
   - Visual calibration curve
   - Shows predicted probability vs observed frequency
   - Well-calibrated model: points on diagonal

3. **Expected Calibration Error (ECE)** (optional but recommended)
   - Single-number summary of calibration quality
   - Easy to compute: bin probabilities, compute weighted absolute difference
   - Formula: `ECE = sum(|acc_bin - conf_bin| * n_bin) / N`

4. **Standard metrics** (still tracked):
   - Precision, Recall, F1, Accuracy, FN, FP (at operating points)

**Implementation:**
```python
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Brier score
brier_pre = brier_score_loss(y_true, y_proba_uncalibrated)
brier_post = brier_score_loss(y_true, y_proba_calibrated)

# Reliability diagram data
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_true, y_proba_calibrated, n_bins=10
)

# ECE (custom function, ~10 lines)
ece = compute_ece(y_true, y_proba_calibrated, n_bins=10)
```

**Recommendation: ✅ Brier Score + Reliability Diagram + ECE**

---

## Decision 4: Threshold Selection Policy After Calibration

### Proposed Approach: **Constraint-based with Recall ≥ 0.94**

**Rationale:**
- **Clinical interpretability**: "We target 94% recall to minimize missed HGG cases" is clear
- **Paper-friendly**: Easy to explain and justify
- **Matches existing analysis**: Your earlier threshold tuning already identified 0.19 as achieving ~0.94 recall
- **Flexible**: Can adjust target if needed (e.g., 0.90, 0.95)

**Policy A (Balanced): Maximize F1**
- Simple, standard metric
- Works well for general use

**Policy B (High-sensitivity): Constraint-based**
- Target: `Recall ≥ 0.94`
- Among thresholds meeting this, choose the one that **maximizes Precision**
- Rationale: Minimize FN while maintaining as much precision as possible

**Alternative Considered: Cost-based (5*FN + FP)**
- **Pros**: Flexible, can encode clinical costs
- **Cons**: 
  - Requires justification of weights (why 5:1?)
  - Less interpretable in papers
  - Harder to compare across studies
- **Decision**: Reject for now - can add later if needed

**Implementation:**
```python
# Policy A: Max F1
best_f1_idx = np.argmax([m['f1_score'] for m in sweep_results])
threshold_balanced = sweep_results[best_f1_idx]['threshold']

# Policy B: Constraint-based (Recall ≥ 0.94, maximize Precision)
candidates = [m for m in sweep_results if m['recall'] >= 0.94]
if candidates:
    best_precision_idx = np.argmax([m['precision'] for m in candidates])
    threshold_high_sens = candidates[best_precision_idx]['threshold']
else:
    # Fallback: highest recall achievable
    threshold_high_sens = max(sweep_results, key=lambda x: x['recall'])['threshold']
```

**Recommendation: ✅ Constraint-based (Recall ≥ 0.94, maximize Precision)**

---

## Decision 5: CLI Interface

### Proposed Approach: **Option A (Simple) with slight enhancement**

**Proposed CLI:**
```bash
python scripts/ensemble/calibrate_and_sweep_thresholds.py \
    --calibration none|platt|isotonic \
    --threshold <float>  # Optional override for single-threshold evaluation
```

**Rationale:**
- **Simplicity**: Minimal flags, easy to use
- **Flexibility**: `--threshold` allows single-threshold evaluation if needed
- **Default behavior**: `--calibration none` means no calibration (backward compatible)
- **Clear output**: Results clearly labeled with calibration mode

**Alternative Considered: Option B (Structured with --operating-point)**
- **Pros**: More explicit about which operating point to use
- **Cons**: Adds complexity, less flexible, requires maintaining a mapping
- **Decision**: Reject - simplicity wins, and `--threshold` override covers edge cases

**Enhancement: Add `--split-seed` for reproducibility**
```bash
--split-seed 42  # Random seed for calibration/threshold split (default: 42)
```

**Recommendation: ✅ Option A (Simple) with `--split-seed`**

---

## Decision 6: Files / Scripts Organization

### Proposed Approach: **New dedicated script**

**New Script:** `scripts/ensemble/calibrate_and_sweep_thresholds.py`

**Rationale:**
- **Separation of concerns**: Calibration is a distinct analysis step
- **Non-invasive**: Doesn't modify existing `train_meta_learner.py`
- **Clear workflow**: 
  1. Train meta-learner (existing script)
  2. Calibrate and re-sweep thresholds (new script)
  3. Use calibrated model in inference (future enhancement)
- **Easier maintenance**: Self-contained, focused responsibility

**What it does:**
1. Loads `merged_oof_predictions.csv`
2. Loads trained `meta_learner_logistic_regression.joblib`
3. Splits OOF data (70/30)
4. Calibrates on 70% (if `--calibration` != 'none')
5. Evaluates on 30% (threshold sweep)
6. Saves all outputs with clear naming

**Alternative Considered: Integrate into `train_meta_learner.py`**
- **Pros**: Single script for meta-learner workflow
- **Cons**: 
  - Makes script complex and harder to maintain
  - Mixes training and analysis concerns
  - Harder to run calibration independently
- **Decision**: Reject - keep separation

**Recommendation: ✅ New script: `calibrate_and_sweep_thresholds.py`**

---

## Decision 7: Output Artifacts Naming

### Proposed Naming Convention:

**Directory Structure:**
```
ensemble/results/calibration/
  {timestamp}_{calibration_mode}_seed{seed}/
    calibration_summary.json          # Brier scores, ECE, split info
    reliability_diagram_{mode}.png    # Calibration curve
    threshold_sweep_{mode}.json          # Full sweep results
    recommended_thresholds_{mode}.json # Selected thresholds (A & B)
    calibrator_{mode}.joblib           # Saved calibrator (if not 'none')
```

**Example:**
```
ensemble/results/calibration/
  2026-02-08_14-30-15_platt_seed42/
    calibration_summary.json
    reliability_diagram_platt.png
    threshold_sweep_platt.json
    recommended_thresholds_platt.json
    calibrator_platt.joblib
```

**File Contents:**

**`calibration_summary.json`:**
```json
{
  "timestamp": "2026-02-08T14:30:15",
  "calibration_mode": "platt",
  "split_seed": 42,
  "n_calibration": 200,
  "n_threshold_selection": 85,
  "brier_pre": 0.1234,
  "brier_post": 0.0987,
  "ece_pre": 0.0456,
  "ece_post": 0.0234,
  "improvement_brier": 0.0247,
  "improvement_ece": 0.0222
}
```

**`recommended_thresholds_{mode}.json`:**
```json
{
  "calibration_mode": "platt",
  "timestamp": "2026-02-08T14:30:15",
  "thresholds": {
    "balanced": {
      "threshold": 0.23,
      "precision": 0.9100,
      "recall": 0.9100,
      "f1": 0.9100,
      "accuracy": 0.8600,
      "fn": 20,
      "fp": 20
    },
    "high_sensitivity": {
      "threshold": 0.18,
      "precision": 0.8400,
      "recall": 0.9500,
      "f1": 0.8900,
      "accuracy": 0.8200,
      "fn": 10,
      "fp": 35
    }
  }
}
```

**Recommendation: ✅ Timestamp + calibration mode + seed in directory name**

---

## Decision 8: Inference Usage (Future)

### Proposed Approach: **Optional calibration in inference script**

**Design:**
- Add `--calibration-mode` flag to `test_ensemble_on_new_patients.py`
- If provided, load corresponding `calibrator_{mode}.joblib` from latest calibration run
- Apply calibration to probabilities before thresholding
- Default: `--calibration-mode none` (backward compatible)

**Rationale:**
- **Flexibility**: Users can choose calibrated or uncalibrated inference
- **Backward compatible**: Default is no calibration (existing behavior)
- **Clear**: Explicit flag makes behavior obvious

**Implementation (future):**
```python
# In test_ensemble_on_new_patients.py
parser.add_argument(
    '--calibration-mode',
    type=str,
    default='none',
    choices=['none', 'platt', 'isotonic'],
    help='Calibration mode (default: none, backward compatible)'
)

# Load calibrator if needed
if args.calibration_mode != 'none':
    calibrator_path = find_latest_calibrator(args.calibration_mode)
    calibrator = joblib.load(calibrator_path)
    ensemble_proba = calibrator.predict_proba([features])[0, 1]
else:
    ensemble_proba = meta_learner.predict_proba([features])[0, 1]
```

**Recommendation: ✅ Optional calibration in inference (future enhancement)**

---

## Implementation Checklist (After Approval)

### Phase 1: Core Calibration Script
- [ ] Create `scripts/ensemble/calibrate_and_sweep_thresholds.py`
- [ ] Implement data loading (OOF predictions + meta-learner model)
- [ ] Implement stratified split (70/30)
- [ ] Implement calibration (Platt/Isotonic using CalibratedClassifierCV)
- [ ] Compute Brier score (pre/post)
- [ ] Compute ECE (pre/post)
- [ ] Generate reliability diagram plot
- [ ] Save calibration artifacts

### Phase 2: Threshold Re-selection
- [ ] Implement threshold sweep on calibrated probabilities (0.05 to 0.95, step 0.01)
- [ ] Implement Policy A: Maximize F1
- [ ] Implement Policy B: Constraint-based (Recall ≥ 0.94, maximize Precision)
- [ ] Save threshold sweep results
- [ ] Save recommended thresholds

### Phase 3: Output & Documentation
- [ ] Save all outputs with proper naming (timestamp, mode, seed)
- [ ] Create summary JSON files
- [ ] Update documentation (README or new calibration doc)
- [ ] Test with `--calibration none` (should match existing behavior)

### Phase 4: Integration (Optional, Future)
- [ ] Add `--calibration-mode` to inference script
- [ ] Implement calibrator loading in inference
- [ ] Test end-to-end workflow

---

## Summary of Recommendations

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| 1. Data Split | Single stratified 70/30 | Simple, sufficient, scientifically valid |
| 2. Calibration Method | sklearn CalibratedClassifierCV | Standard, proven, works with existing model |
| 3. Metrics | Brier + Reliability + ECE | Comprehensive, standard metrics |
| 4. Threshold Policy | Constraint-based (Recall ≥ 0.94) | Clinically interpretable, paper-friendly |
| 5. CLI | Simple with `--calibration` + `--threshold` | Minimal, flexible, backward compatible |
| 6. Scripts | New dedicated script | Separation of concerns, non-invasive |
| 7. Output Naming | Timestamp + mode + seed | Clear, non-overwriting, traceable |
| 8. Inference | Optional calibration (future) | Flexible, backward compatible |

---

## Questions for Approval

1. **Recall target**: Is 0.94 appropriate, or should we use 0.90 or 0.95?
2. **Calibration methods**: Should we support both Platt and Isotonic, or just one?
3. **Output location**: Is `ensemble/results/calibration/` acceptable?
4. **Inference integration**: Should we implement Phase 4 now, or defer?

---

## Next Steps

After approval, I will:
1. Implement the calibration script following this design
2. Test with `--calibration none` to ensure backward compatibility
3. Generate example outputs for review
4. Update documentation

**Ready for your feedback and approval!**

