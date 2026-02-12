# Loss Function Modifications for MIL Training

## Summary

Modified `train_dual_stream_mil.py` to support:
1. **Asymmetric/Focal Loss** with configurable false negative (FN) penalty
2. **Single-fold training mode** for rapid prototyping
3. **Backward compatibility** with existing CrossEntropyLoss

## Changes Made

### 1. New Loss Functions

#### AsymmetricBCELoss
- **Location**: Added as a class before `train_epoch()` function
- **Purpose**: Penalizes false negatives (missed HGG cases) more heavily than false positives
- **Formula**: `loss = -[pos_weight * y * log(p) + (1-y) * log(1-p)]`
- **Parameters**:
  - `pos_weight`: Multiplier for positive class (default: 3.0, recommended: 3.0-5.0)
  - Higher values = stronger FN penalty

#### FocalLoss
- **Location**: Added as a class before `train_epoch()` function
- **Purpose**: Focuses on hard examples while penalizing false negatives
- **Formula**: 
  - Positive: `-alpha * pos_weight * (1-p)^gamma * log(p)`
  - Negative: `-(1-alpha) * p^gamma * log(1-p)`
- **Parameters**:
  - `alpha`: Class weighting (default: 0.25)
  - `gamma`: Focusing parameter (default: 2.0, higher = more focus on hard examples)
  - `pos_weight`: Additional FN penalty multiplier (default: 1.0)

### 2. Modified `train_epoch()` Function

**Changes**:
- Added `loss_type` parameter to distinguish between loss types
- Modified loss computation to use new loss functions when `loss_type != 'ce'`
- For asymmetric/focal loss: uses loss function directly (FN penalty built-in)
- For CrossEntropyLoss: maintains existing adaptive label smoothing and class weights

**Location**: Lines ~512-590

### 3. Command-Line Arguments

**New Arguments**:
```bash
--loss-type {ce,asymmetric,focal}
    Loss function type:
    - ce: CrossEntropyLoss (default, existing behavior)
    - asymmetric: AsymmetricBCELoss (FN penalty)
    - focal: FocalLoss (hard examples + FN penalty)

--pos-weight FLOAT (default: 3.0)
    Positive class weight multiplier for asymmetric/focal loss.
    Higher values penalize false negatives more.
    Recommended: 3.0-5.0 for medical screening.

--gamma FLOAT (default: 2.0)
    Focal loss focusing parameter (gamma).
    Only used with --loss-type focal.
    Higher values focus more on hard examples.

--single-fold
    Train only the specified fold (for rapid prototyping).
    Default: train all 5 folds sequentially.

--fold INT (default: 0)
    Fold number (0-4).
    When --single-fold is used, only this fold is trained.
```

### 4. Single-Fold Training Mode

**Implementation**:
- Added `--single-fold` flag
- When enabled, only trains the fold specified by `--fold`
- When disabled (default), trains all 5 folds sequentially
- Each fold gets its own model instance, optimizer, and output directory

**Location**: Lines ~1150-1170 (fold selection logic)

### 5. Loss Function Selection in `main()`

**Changes**:
- Loss function is created per fold (inside fold loop)
- Selection based on `args.loss_type`:
  - `'asymmetric'`: Creates `AsymmetricBCELoss(pos_weight=args.pos_weight)`
  - `'focal'`: Creates `FocalLoss(alpha=0.25, gamma=args.gamma, pos_weight=args.pos_weight)`
  - `'ce'`: Creates `CrossEntropyLoss` with adaptive label smoothing (existing behavior)

**Location**: Lines ~1264-1281

## Usage Examples

### Example 1: Single-fold training with Asymmetric Loss
```bash
python scripts/training/train_dual_stream_mil.py \
    --fold 0 \
    --single-fold \
    --loss-type asymmetric \
    --pos-weight 4.0 \
    --sampling-strategy entropy \
    --epochs 30
```

### Example 2: Single-fold training with Focal Loss
```bash
python scripts/training/train_dual_stream_mil.py \
    --fold 0 \
    --single-fold \
    --loss-type focal \
    --pos-weight 3.0 \
    --gamma 2.0 \
    --sampling-strategy entropy \
    --epochs 30
```

### Example 3: Full 5-fold training with Asymmetric Loss
```bash
python scripts/training/train_dual_stream_mil.py \
    --loss-type asymmetric \
    --pos-weight 3.5 \
    --sampling-strategy entropy \
    --epochs 60
```

### Example 4: Default behavior (CrossEntropyLoss, all folds)
```bash
python scripts/training/train_dual_stream_mil.py \
    --fold 0 \
    --epochs 60
```

## How FN Penalty Works

### Asymmetric Loss
- **False Negative (HGG missed)**: Loss = `-pos_weight * log(p_pos)`
  - If `pos_weight=3.0`, FN contributes 3x more to loss than FP
- **False Positive (LGG misclassified)**: Loss = `-log(1-p_pos)`
  - Standard penalty

### Focal Loss
- **False Negative**: Loss = `-alpha * pos_weight * (1-p_pos)^gamma * log(p_pos)`
  - `(1-p_pos)^gamma` down-weights easy negatives
  - `pos_weight` adds additional FN penalty
- **False Positive**: Loss = `-(1-alpha) * p_neg^gamma * log(1-p_pos)`
  - Standard focal loss term

## Expected Effects

### On Recall
- **Goal**: Increase recall from 0.52 → ≥0.80
- **Mechanism**: Higher FN penalty forces model to be more conservative (predict HGG more often)
- **Trade-off**: May increase false positives, but acceptable for medical screening

### On Training
- Loss values may be higher initially (due to FN penalty)
- Model should learn to prioritize HGG detection
- Validation recall should increase over epochs

## Compatibility

### Backward Compatibility
- **Default behavior unchanged**: `--loss-type ce` (or no argument) uses existing CrossEntropyLoss
- All existing arguments still work
- Predictions saved in same format (`predictions/val_probs.npy`)
- Compatible with `compare_mil_models.py`

### Threshold Tuning Readiness
- Probabilities are saved exactly as before
- No changes to prediction format
- Threshold tuning can be done post-hoc on saved probabilities
- Full nested CV threshold tuning can be added later without breaking changes

## Safety Checks

### What Was NOT Changed
- ✅ Model architecture (no changes to `dual_stream_mil.py`)
- ✅ Feature extractor (ResNet18/EfficientNet unchanged)
- ✅ MIL pooling logic (attention and selection unchanged)
- ✅ Prediction saving format (compatible with evaluation scripts)
- ✅ Existing CrossEntropyLoss behavior (when `--loss-type ce`)

### What Was Changed
- ✅ Loss function selection (new options added)
- ✅ Training loop structure (fold loop added for single-fold mode)
- ✅ Loss computation in `train_epoch()` (conditional based on loss type)

## Next Steps

1. **Run single-fold training** with entropy sampling + asymmetric loss:
   ```bash
   python scripts/training/train_dual_stream_mil.py \
       --fold 0 \
       --single-fold \
       --loss-type asymmetric \
       --pos-weight 4.0 \
       --sampling-strategy entropy \
       --epochs 30
   ```

2. **Regenerate OOF predictions** for that fold

3. **Run comparison** using `compare_mil_models.py` to evaluate improvement

4. **Iterate**: Adjust `--pos-weight` or try `--loss-type focal` if needed

5. **If successful**: Train all 5 folds with best settings

## Notes

- **Rapid Prototyping**: Single-fold mode allows fast iteration without retraining all folds
- **FN Penalty Strength**: Start with `--pos-weight 3.0-4.0`, adjust based on results
- **Loss Stability**: Monitor training loss - should decrease normally (may be higher initially due to FN penalty)
- **Medical Priority**: Recall improvement is more important than precision for screening applications

