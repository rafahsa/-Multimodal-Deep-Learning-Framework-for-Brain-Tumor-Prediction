# Swin-2 Training Fixes and Complementarity Evaluation

**Date:** 2026-02-10  
**Issue:** Swin-2 training collapsed into degenerate solution (predicting all HGG)  
**Solution:** Minimal fixes to prevent collapse + complementarity-focused evaluation

---

## Training Collapse Fixes

### 1. Replace Single-Layer Classifier with 2-Layer MLP + Dropout

**Change:** Set `use_hidden_layer=True` in `SwinUNETREncoderClassifier`

**Rationale:** Single-layer classifier is too simple and prone to collapse. 2-layer MLP with dropout provides better regularization.

**Location:** `scripts/training/train_swin2_unetr_3d.py` line ~504

### 2. Reduce Focal Loss Gamma

**Change:** Default `--focal-gamma` from 2.0 → 1.0

**Rationale:** High gamma (2.0) over-emphasizes hard examples, leading to collapse. Gamma=1.0 provides balanced focus.

**Location:** `scripts/training/train_swin2_unetr_3d.py` line ~280

### 3. Cap Hard-Mining Oversampling

**Change:** Cap hard cases to max 30% of each batch

**Rationale:** Unlimited oversampling of FN cases causes model to overfit to hard examples and collapse.

**Location:** `scripts/training/train_swin2_unetr_3d.py` lines ~427-443

### 4. Add Temperature Scaling

**Change:** Add `--temperature` parameter (default 1.0) for logits during evaluation

**Rationale:** Temperature scaling can help calibrate predictions and prevent overconfident outputs.

**Location:** `scripts/training/train_swin2_unetr_3d.py` lines ~282, ~219, ~637

---

## Updated Training Command

```bash
cd /workspace/brain_tumor_project

python scripts/training/train_swin2_unetr_3d.py \
  --fold 0 \
  --epochs 60 \
  --batch-size 1 \
  --lr 5e-5 \
  --classifier-lr 1e-4 \
  --focal-alpha 0.25 \
  --focal-gamma 1.0 \
  --hard-mining \
  --hard-mining-multiplier 2 \
  --oof-predictions-file ensemble/oof_predictions/merged_oof_predictions.csv \
  --seed 42 \
  --temperature 1.0
```

**Note:** The fixes are applied automatically:
- `use_hidden_layer=True` (2-layer MLP)
- `focal_gamma=1.0` (reduced from 2.0)
- Hard-mining capped at 30% of batch
- Temperature scaling applied during evaluation

---

## Complementarity Evaluation

### Primary Goal: New Information, NOT Accuracy

After retraining, evaluate whether Swin-2 provides complementary signal to Swin-1:

### Evaluation Script

```bash
cd /workspace/brain_tumor_project

# Find the latest Swin-2 run
SWIN2_RUN_DIR=$(ls -td results/SwinUNETR-3D-Swin2/fold_0/run_* | head -1)
SWIN2_PREDICTIONS="${SWIN2_RUN_DIR}/predictions/swin2_predictions.csv"

# Run complementarity evaluation
python scripts/analysis/evaluate_swin2_complementarity.py \
    --swin1-oof ensemble/oof_predictions/merged_oof_predictions.csv \
    --swin2-predictions "${SWIN2_PREDICTIONS}" \
    --fold-id 0 \
    --output-dir ensemble/results/swin2_complementarity
```

### GO/NO-GO Criteria

**All three criteria must be met:**

1. **FN Ranking AUC Improvement ≥ 0.05**
   - Swin-2 AUC (FN vs rest) ≥ Swin-1 AUC + 0.05
   - Measures: Can Swin-2 better distinguish Swin-1 FN cases from others?

2. **Correlation < 0.70**
   - Pearson correlation(Swin-1 prob, Swin-2 prob) < 0.70
   - Measures: Does Swin-2 provide non-redundant signal?

3. **Clear FN/TN Separation**
   - Swin-2 mean(FN) > Swin-2 mean(TN)
   - Measures: Does Swin-2 rank Swin-1 FN cases higher than TN cases?

### Decision Rules

- **CONTINUE:** All 3 criteria met → Swin-2 provides complementary signal
- **STOP:** Any criterion fails → Swin-2 does not add value, stop permanently

---

## What Changed vs Original Swin-2

| Aspect | Original | Fixed |
|--------|----------|-------|
| **Classifier** | Single-layer | 2-layer MLP + dropout |
| **Focal Gamma** | 2.0 | 1.0 |
| **Hard Mining** | Unlimited oversampling | Capped at 30% of batch |
| **Temperature** | None | 1.0 (configurable) |
| **Evaluation Focus** | Accuracy/Recall | Complementarity |

---

## Expected Outcomes

### If Fixes Work

- Model should NOT collapse (Recall < 1.0, Precision > 0.74)
- Swin-2 should learn meaningful patterns, not degenerate solution
- Complementarity evaluation will determine if Swin-2 adds value

### If Complementarity Criteria Met

✅ **CONTINUE:** Proceed with full 5-fold CV for Swin-2

### If Complementarity Criteria Failed

❌ **STOP:** Swin-2 does not provide complementary signal. Do not proceed.

---

## Files Modified

1. `scripts/training/train_swin2_unetr_3d.py`
   - Added `use_hidden_layer=True`
   - Changed default `focal_gamma=1.0`
   - Added hard-mining cap (30% of batch)
   - Added temperature scaling

2. `scripts/analysis/evaluate_swin2_complementarity.py` (NEW)
   - Complementarity-focused evaluation
   - GO/NO-GO decision based on 3 criteria

---

## Scientific Validation

This is a **scientific validation step**, not an optimization exercise:

- **Goal:** Determine if Swin-2 provides new information vs Swin-1
- **Method:** Strict complementarity criteria (not accuracy)
- **Decision:** Binary CONTINUE/STOP based on evidence
- **No hand-waving:** If criteria fail, stop permanently

---

*Created: 2026-02-10*  
*Purpose: Fix Swin-2 training collapse and evaluate complementarity*


