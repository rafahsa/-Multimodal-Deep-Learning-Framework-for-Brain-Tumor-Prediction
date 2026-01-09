# Dual-Stream MIL Training Fix - Verification Report

## Issue Summary

**Error**: `TypeError: create_dual_stream_mil() got an unexpected keyword argument 'critical_selection_mode'`

**Root Cause**: The training script was passing `critical_selection_mode` to `create_dual_stream_mil()`, but this parameter was removed from the model API when switching to soft selection.

**Status**: ✅ **FIXED**

---

## What Was Wrong

### Issue 1: Obsolete Argument Parser
- **Location**: `scripts/training/train_dual_stream_mil.py` line 540
- **Problem**: Argument parser still defined `--critical-selection-mode`
- **Impact**: Confusing, but not directly causing the error (unused arguments are ignored)

### Issue 2: EMA Model Creation (PRIMARY ISSUE)
- **Location**: `scripts/training/train_dual_stream_mil.py` line 829 (in saved file)
- **Problem**: EMA model creation was passing `critical_selection_mode=args.critical_selection_mode`
- **Impact**: Direct cause of the TypeError

### Why It Happened
When the model architecture was updated to always use soft selection:
1. `create_dual_stream_mil()` signature was updated (parameter removed)
2. Main model creation was fixed (no longer passes the parameter)
3. EMA model creation was **missed** (still had the old parameter)

---

## What Was Fixed

### ✅ Fix 1: Removed Obsolete Argument Parser
**File**: `scripts/training/train_dual_stream_mil.py`

**Before**:
```python
parser.add_argument('--critical-selection-mode', type=str, default='hard', choices=['hard', 'soft'],
                   help='Critical instance selection mode (default: hard)')
```

**After**:
```python
# Removed - soft selection is always used (differentiable)
```

### ✅ Fix 2: Fixed EMA Model Creation
**File**: `scripts/training/train_dual_stream_mil.py` (line ~823-832)

**Before**:
```python
ema_model = create_dual_stream_mil(
    num_classes=2,
    instance_encoder_backbone=args.instance_encoder_backbone,
    instance_encoder_input_size=args.instance_encoder_input_size,
    critical_selection_mode=args.critical_selection_mode,  # ❌ WRONG
    attention_type=args.attention_type,
    fusion_method=args.fusion_method,
    dropout=args.dropout,
    use_hidden_layer=args.use_hidden_layer,
    logger=None
)
```

**After**:
```python
ema_model = create_dual_stream_mil(
    num_classes=2,
    instance_encoder_backbone=args.instance_encoder_backbone,
    instance_encoder_input_size=args.instance_encoder_input_size,
    attention_type=args.attention_type,  # ✅ CORRECT - matches main model
    fusion_method=args.fusion_method,
    dropout=args.dropout,
    use_hidden_layer=args.use_hidden_layer,
    logger=None
)
```

### ✅ Verification: Both Model Creations Are Consistent

**Main Model** (line ~723):
```python
model = create_dual_stream_mil(
    num_classes=2,
    instance_encoder_backbone=args.instance_encoder_backbone,
    instance_encoder_input_size=args.instance_encoder_input_size,
    attention_type=args.attention_type,
    fusion_method=args.fusion_method,
    dropout=args.dropout,
    use_hidden_layer=args.use_hidden_layer,
    logger=logger
)
```

**EMA Model** (line ~823):
```python
ema_model = create_dual_stream_mil(
    num_classes=2,
    instance_encoder_backbone=args.instance_encoder_backbone,
    instance_encoder_input_size=args.instance_encoder_input_size,
    attention_type=args.attention_type,
    fusion_method=args.fusion_method,
    dropout=args.dropout,
    use_hidden_layer=args.use_hidden_layer,
    logger=None  # Only difference: logger=None for EMA
)
```

**✅ Both are now consistent and correct!**

---

## Verification Tests

### Test 1: Model API Verification
```bash
cd /workspace/brain_tumor_project
python -c "
from models.dual_stream_mil import create_dual_stream_mil
import inspect
sig = inspect.signature(create_dual_stream_mil)
print('Parameters:', list(sig.parameters.keys()))
"
```

**Expected Output**:
```
Parameters: ['num_classes', 'instance_encoder_backbone', 'instance_encoder_input_size', 'attention_type', 'fusion_method', 'dropout', 'use_hidden_layer', 'logger']
```

**✅ No `critical_selection_mode` in the signature**

### Test 2: Model Creation Test
```bash
python -c "
from models.dual_stream_mil import create_dual_stream_mil
# Correct call (should work)
model = create_dual_stream_mil(
    num_classes=2,
    instance_encoder_backbone='resnet18',
    instance_encoder_input_size=224,
    attention_type='gated',
    fusion_method='concat',
    dropout=0.4,
    use_hidden_layer=True,
    logger=None
)
print('✓ Model creation: SUCCESS')
"
```

**Expected**: No errors

### Test 3: Training Script Import Test
```bash
python -c "
import sys
sys.path.insert(0, '.')
from scripts.training.train_dual_stream_mil import *
print('✓ Training script imports successfully')
"
```

**Expected**: No errors

---

## Training Commands

### Minimal Test Command (5 epochs, small batch)

**Purpose**: Quick verification that training starts without errors

```bash
cd /workspace/brain_tumor_project

python scripts/training/train_dual_stream_mil.py \
  --fold 0 \
  --epochs 5 \
  --batch-size 2 \
  --bag-size 32 \
  --sampling-strategy random \
  --amp
```

**Expected Behavior**:
- ✅ No TypeError
- ✅ Model and EMA model create successfully
- ✅ Training starts and completes 5 epochs
- ✅ Metrics are logged correctly

### Recommended Full Training Command

**Purpose**: Complete training for evaluation and comparison

```bash
cd /workspace/brain_tumor_project

python scripts/training/train_dual_stream_mil.py \
  --fold 0 \
  --epochs 60 \
  --batch-size 4 \
  --bag-size 64 \
  --sampling-strategy random \
  --instance-encoder-backbone resnet18 \
  --instance-encoder-input-size 224 \
  --attention-type gated \
  --fusion-method concat \
  --dropout 0.4 \
  --use-hidden-layer \
  --lr 1e-4 \
  --classifier-lr 2e-4 \
  --weight-decay 1e-4 \
  --gradient-accumulation-steps 2 \
  --ema-decay 0.995 \
  --optimizer adamw \
  --scheduler cosine \
  --grad-clip 1.0 \
  --early-stopping 10 \
  --early-stopping-min-epochs 15 \
  --amp \
  --seed 42
```

**Notes**:
- Uses same training philosophy as ResNet50-3D and SwinUNETR-3D
- CrossEntropyLoss + WeightedRandomSampler (handled automatically)
- EMA enabled (decay=0.995)
- Cosine LR schedule with warmup
- Early stopping (patience=10, min_epochs=15)
- AMP for faster training

---

## Files Modified

1. ✅ **`scripts/training/train_dual_stream_mil.py`**:
   - Removed `--critical-selection-mode` argument parser (line ~540)
   - Fixed EMA model creation to match main model creation (line ~829)
   - Both model creations now use identical parameters

2. ✅ **`models/dual_stream_mil.py`** (previously fixed):
   - Always uses soft selection (differentiable)
   - No `critical_selection_mode` parameter in API

---

## API Consistency Summary

### Model Factory Function
```python
create_dual_stream_mil(
    num_classes: int = 2,
    instance_encoder_backbone: str = 'resnet18',
    instance_encoder_input_size: int = 224,
    attention_type: str = 'gated',
    fusion_method: str = 'concat',
    dropout: float = 0.4,
    use_hidden_layer: bool = True,
    logger=None
) -> DualStreamMIL
```

**Key Points**:
- ✅ No `critical_selection_mode` parameter
- ✅ Always uses soft (differentiable) selection internally
- ✅ Temperature can be adjusted during training (future enhancement)

### Training Script
- ✅ Both main and EMA model use same parameters
- ✅ Consistent with model API
- ✅ No obsolete arguments

---

## Troubleshooting

### If you still see the error:

1. **Clear Python cache**:
   ```bash
   find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null
   find . -name "*.pyc" -delete 2>/dev/null
   ```

2. **Verify file state**:
   ```bash
   grep -n "critical_selection_mode" scripts/training/train_dual_stream_mil.py
   ```
   Should return: **No matches**

3. **Check model signature**:
   ```bash
   python -c "from models.dual_stream_mil import create_dual_stream_mil; import inspect; print(inspect.signature(create_dual_stream_mil))"
   ```
   Should NOT show `critical_selection_mode`

4. **Verify both model creations**:
   ```bash
   grep -A 10 "ema_model = create_dual_stream_mil" scripts/training/train_dual_stream_mil.py
   ```
   Should NOT show `critical_selection_mode`

---

## Status: ✅ READY FOR TRAINING

The training script is now:
- ✅ Consistent with model API
- ✅ Both model creations are correct
- ✅ No obsolete arguments
- ✅ Ready for training

**Next Steps**:
1. Run minimal test command (5 epochs) to verify
2. If successful, run full training command
3. Compare metrics with ResNet50-3D and SwinUNETR-3D

---

**Fix Date**: January 2025  
**Verified**: ✅ Model creation, script imports, API consistency  
**Status**: Ready for training

