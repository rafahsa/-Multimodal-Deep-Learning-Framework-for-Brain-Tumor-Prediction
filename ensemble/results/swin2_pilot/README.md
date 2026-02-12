# Swin-2 Pilot Experiment: Single-Fold Training

**Status:** Standalone pilot experiment, fully isolated from Swin-1  
**Goal:** Test conservative Swin-2 modifications on a single fold to evaluate if it should proceed to full 5-fold CV

---

## What is Swin-2?

Swin-2 is a **complementary model** designed to target Swin-1's false negatives (missed HGG cases). It uses memory-optimized architecture modifications:

### 1. Memory-Optimized Architecture
- **Patch Size = 2** (MANDATORY: patch_size=1 creates ~2M tokens → OOM in 3D attention)
- **Window Size = 4** (instead of 7): Smaller local attention for subtle patterns
- **Feature Size = 24** (reduced from 48): Memory efficiency (must be divisible by 12)
- **Depths = [2, 2, 2, 1]** (reduced from [2, 2, 2, 2]): Memory efficiency
- **Why:** patch_size=1 is infeasible (creates 2,097,152 tokens, attention softmax OOM)
- **Research Intent Preserved:** window_size=4 focuses on local subtle patterns that Swin-1's global attention (window_size=7) might miss, still targets FN cases

### 2. Focal Loss (instead of CrossEntropyLoss)
- **Why:** Focuses on hard examples (FN cases) rather than easy ones
- **Parameters:** alpha=0.25, gamma=2.0
- **Effect:** Model learns to better distinguish subtle HGG from LGG

### 3. Hard Example Mining
- **Why:** Oversample Swin-1 FN cases during training
- **Method:** Identify HGG cases where Swin-1 predicted < 0.5, oversample by 2x
- **Effect:** Model sees more difficult cases, learns to catch what Swin-1 misses

---

## How Swin-2 Differs from Swin-1

| Aspect | Swin-1 | Swin-2 |
|--------|--------|--------|
| **Patch Size** | 2 | 2 (MANDATORY: patch_size=1 → OOM) |
| **Window Size** | 7 (global attention) | 4 (local attention for subtle patterns) |
| **Feature Size** | 48 | 24 (memory-optimized, divisible by 12) |
| **Depths** | [2, 2, 2, 2] | [2, 2, 2, 1] (memory-optimized) |
| **Loss Function** | CrossEntropyLoss | Focal Loss (α=0.25, γ=2.0) |
| **Training Strategy** | Class-balanced sampling | Hard example mining + class balancing |
| **Output Directory** | `results/SwinUNETR-3D/` | `results/SwinUNETR-3D-Swin2/` |
| **Training Script** | `train_swin_unetr_3d.py` | `train_swin2_unetr_3d.py` (separate) |

**Key Point:** Swin-2 is **completely isolated** from Swin-1. They share no code, no checkpoints, no dependencies.

---

## Exact Commands

### Step 1: Train Swin-2 Pilot (Fold 0)

```bash
cd /workspace/brain_tumor_project

python scripts/training/train_swin2_unetr_3d.py \
  --fold 0 \
  --epochs 60 \
  --batch-size 1 \
  --lr 5e-5 \
  --classifier-lr 1e-4 \
  --focal-alpha 0.25 \
  --focal-gamma 2.0 \
  --hard-mining \
  --hard-mining-multiplier 2 \
  --oof-predictions-file ensemble/oof_predictions/merged_oof_predictions.csv \
  --seed 42
```

**Expected Output:**
- Checkpoint: `results/SwinUNETR-3D-Swin2/fold_0/run_YYYYMMDD_HHMMSS/checkpoints/best.pt`
- Predictions: `results/SwinUNETR-3D-Swin2/fold_0/run_YYYYMMDD_HHMMSS/predictions/swin2_predictions.csv`
- Metrics: `results/SwinUNETR-3D-Swin2/fold_0/run_YYYYMMDD_HHMMSS/metrics/metrics.json`

**Training Time:** ~2-4 hours on GPU (depending on hardware)  
**Expected Peak GPU Memory:** ~12-16GB with batch_size=1 (memory-optimized architecture)

---

### Step 2: Evaluate Decision Gate

After training completes, find the latest run directory and evaluate:

```bash
cd /workspace/brain_tumor_project

# Find the latest run directory
SWIN2_RUN_DIR=$(ls -td results/SwinUNETR-3D-Swin2/fold_0/run_* | head -1)
SWIN2_PREDICTIONS="${SWIN2_RUN_DIR}/predictions/swin2_predictions.csv"

# Run evaluation
python scripts/analysis/evaluate_swin2_pilot.py \
    --swin1-oof ensemble/oof_predictions/merged_oof_predictions.csv \
    --swin2-predictions "${SWIN2_PREDICTIONS}" \
    --fold-id 0 \
    --output-dir ensemble/results/swin2_pilot
```

**Expected Output:**
- JSON: `ensemble/results/swin2_pilot/fold_0_pilot_metrics.json`
- Markdown: `ensemble/results/swin2_pilot/fold_0_pilot_metrics.md`
- Console: GO/NO-GO decision with detailed metrics

---

## GO/NO-GO Decision Criteria

The evaluation script makes a decision based on **two criteria**:

### Criterion 1: FN Reduction >= 30%
- **Calculation:** `(FN_swin1 - FN_swin2) / FN_swin1 >= 0.30`
- **Example:** Swin-1 has 10 FN, Swin-2 has 7 FN → 30% reduction ✅

### Criterion 2: Correlation < 0.70
- **Calculation:** Pearson correlation between Swin-1 and Swin-2 probabilities < 0.70
- **Why:** Ensures Swin-2 provides complementary signal, not redundant
- **Example:** Correlation = 0.65 ✅

### Decision Rules
- **GO:** Both criteria met → Proceed to full 5-fold CV
- **NO-GO:** Either criterion fails → Stop, reconsider approach

---

## Interpreting Results

### GO Decision (Proceed)
```
Decision: GO
Reason: Both criteria met: FN reduction >= 30% and correlation < 0.70
```

**Next Steps:**
1. Train Swin-2 on all 5 folds
2. Generate full OOF predictions
3. Integrate into ensemble
4. Evaluate ensemble performance

### NO-GO Decision (Stop)
```
Decision: NO_GO
Reason: FN reduction 15.00% < 30%; Correlation 0.75 >= 0.70
```

**Next Steps:**
1. Review why criteria failed
2. Consider alternative modifications:
   - Different patch size
   - Different loss function parameters
   - Different hard mining strategy
3. May need different approach (e.g., different architecture)

---

## Output Files

### Training Outputs
- **Checkpoint:** `results/SwinUNETR-3D-Swin2/fold_0/run_*/checkpoints/best.pt`
- **Predictions CSV:** `results/SwinUNETR-3D-Swin2/fold_0/run_*/predictions/swin2_predictions.csv`
  - Format: `patient_id, fold, swin2_prob, label`
- **Metrics:** `results/SwinUNETR-3D-Swin2/fold_0/run_*/metrics/metrics.json`

### Evaluation Outputs
- **JSON:** `ensemble/results/swin2_pilot/fold_0_pilot_metrics.json`
- **Markdown:** `ensemble/results/swin2_pilot/fold_0_pilot_metrics.md`

---

## Isolation Guarantees

✅ **Swin-2 is fully isolated from Swin-1:**
- Separate training script (`train_swin2_unetr_3d.py`)
- Separate output directory (`results/SwinUNETR-3D-Swin2/`)
- No modifications to Swin-1 code
- No dependencies on Swin-1 checkpoints
- Can delete Swin-1 and Swin-2 still trains

✅ **Import errors fixed:**
- Explicit `PROJECT_ROOT` setup at top of script
- `sys.path.insert(0, str(PROJECT_ROOT))` before any imports
- Absolute imports from project root
- No environment variables or PYTHONPATH hacks required

✅ **Reproducible:**
- Fixed random seed (42)
- Clear output directories
- Saved configuration in metrics.json

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'utils.dataset_3d'"
- **Fixed:** Script now sets `PROJECT_ROOT` explicitly at the top
- **Verify:** Check that `PROJECT_ROOT` is set correctly (should be 2 levels up from script)

### Error: "OOF predictions file not found"
- **Solution:** Ensure `ensemble/oof_predictions/merged_oof_predictions.csv` exists
- **Check:** Run from project root: `cd /workspace/brain_tumor_project`

### Error: "No Swin-1 FN cases in training split"
- **Meaning:** Fold 0 has no Swin-1 FN cases in training
- **Effect:** Hard mining automatically disabled, training proceeds with normal class balancing
- **Action:** This is fine, training will still work

### Error: "Patient ID mismatch" in evaluation
- **Check:** Ensure same fold is used for training and evaluation
- **Verify:** Patient IDs match between Swin-1 OOF and Swin-2 predictions

---

## Files Created

1. **`scripts/training/train_swin2_unetr_3d.py`** - Standalone Swin-2 training script
2. **`scripts/analysis/evaluate_swin2_pilot.py`** - Evaluation script with GO/NO-GO decision
3. **`ensemble/results/swin2_pilot/README.md`** - This documentation

---

## Memory Optimization Details

### Why patch_size=1 is Infeasible
- **Token Count:** With img_size=(128,128,128), patch_size=1 creates (128/1)³ = **2,097,152 tokens**
- **Memory Explosion:** 3D window attention's softmax operation requires O(tokens²) memory
- **OOM Location:** Confirmed in first Swin block's attention softmax, even with batch_size=1
- **Solution:** patch_size=2 creates (128/2)³ = **262,144 tokens** (8× reduction), making attention feasible

### Why window_size=4 Preserves Research Intent
- **Swin-1:** window_size=7 (global attention) → captures large, clear patterns
- **Swin-2:** window_size=4 (local attention) → focuses on subtle, local patterns
- **Complementarity:** Local detail focus still targets FN cases (small/diffuse tumors)
- **Research Goal Unchanged:** FN reduction + complementarity via local detail focus

### Memory-Optimized Architecture
- **patch_size:** 2 (mandatory, cannot use 1)
- **window_size:** 4 (smaller local attention)
- **feature_size:** 24 (reduced from 48, 50% reduction, must be divisible by 12)
- **depths:** [2, 2, 2, 1] (reduced from [2, 2, 2, 2], 25% reduction)
- **Expected Memory:** ~12-16GB peak with batch_size=1

## Research Notes

- **Pilot Experiment:** This is a single-fold experiment to validate the approach
- **Memory-Optimized:** Architecture adjusted to fit <20GB GPU memory while preserving research intent
- **Decision Gate:** Strict criteria ensure we only proceed if Swin-2 adds value
- **Isolation:** Complete separation from Swin-1 ensures clean research structure
- **GO/NO-GO Criteria:** Unchanged (FN reduction >= 30%, correlation < 0.70)

---

*Created: 2026-02-10*  
*Purpose: Conservative Swin-2 pilot experiment for single-fold validation*  
*Isolation: Fully isolated from Swin-1, no shared code or dependencies*

