# Dual-Stream MIL Training Stability Fix - Implementation Guide

## Summary of Changes

This document provides step-by-step implementation of fixes for MIL training instability.

### Primary Fixes Implemented

1. ✅ **Soft Selection with Temperature** (Model Architecture)
   - Removed hard selection mode
   - Always use differentiable soft selection
   - Added temperature parameter for annealing

2. 🔄 **Temperature Annealing** (Training Script - TODO)
   - Add temperature schedule (start high, anneal to low)
   - Pass temperature to model forward pass

3. 🔄 **Label Smoothing** (Training Script - TODO)
   - Add label_smoothing parameter to CrossEntropyLoss

4. 🔄 **Instance-Level Regularization** (Training Script - TODO)
   - Add attention entropy loss
   - Add selection confidence loss
   - Combine with bag-level loss

5. 🔄 **Gradient Clipping** (Training Script - TODO)
   - Reduce default from 1.0 to 0.5

## Files Modified

### ✅ models/dual_stream_mil.py
- ✅ Removed `selection_mode` parameter
- ✅ Always use soft selection (differentiable)
- ✅ Added temperature parameter to forward pass
- ✅ Updated CriticalInstanceSelector to accept temperature

### 🔄 scripts/training/train_dual_stream_mil.py (In Progress)
- 🔄 Remove `--critical-selection-mode` argument
- 🔄 Add `--temperature-start` and `--temperature-end` arguments
- 🔄 Add `--label-smoothing` argument
- 🔄 Add `--reg-weight-entropy` and `--reg-weight-confidence` arguments
- 🔄 Implement temperature annealing schedule
- 🔄 Update train_epoch to use temperature and compute regularization
- 🔄 Update validate to use temperature
- 🔄 Update model creation (remove critical_selection_mode)

## Implementation Steps

### Step 1: Model Architecture (✅ COMPLETE)

**File**: `models/dual_stream_mil.py`

**Changes Made**:
- CriticalInstanceSelector always uses soft selection
- Temperature parameter can be passed to forward() for annealing
- Removed hard selection mode

### Step 2: Training Script Updates (🔄 IN PROGRESS)

#### 2.1 Remove Hard Selection Argument
```python
# REMOVE:
parser.add_argument('--critical-selection-mode', ...)

# Model creation (already updated):
# Remove critical_selection_mode parameter
```

#### 2.2 Add Temperature Arguments
```python
parser.add_argument('--temperature-start', type=float, default=10.0,
                   help='Initial temperature for soft selection (default: 10.0)')
parser.add_argument('--temperature-end', type=float, default=1.0,
                   help='Final temperature for soft selection (default: 1.0)')
```

#### 2.3 Add Label Smoothing
```python
parser.add_argument('--label-smoothing', type=float, default=0.1,
                   help='Label smoothing factor (default: 0.1)')

# Loss function:
loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
```

#### 2.4 Add Regularization Weights
```python
parser.add_argument('--reg-weight-entropy', type=float, default=0.01,
                   help='Weight for attention entropy regularization (default: 0.01)')
parser.add_argument('--reg-weight-confidence', type=float, default=0.01,
                   help='Weight for selection confidence regularization (default: 0.01)')
```

#### 2.5 Temperature Annealing Schedule
```python
def get_temperature(epoch: int, total_epochs: int, temp_start: float, temp_end: float) -> float:
    """
    Linear temperature annealing schedule.
    
    Starts at temp_start (high = softer distribution) and anneals to temp_end (low = sharper).
    """
    if total_epochs <= 1:
        return temp_end
    
    progress = epoch / (total_epochs - 1)
    temperature = temp_start * (1 - progress) + temp_end * progress
    return max(temp_end, temperature)  # Never go below temp_end
```

#### 2.6 Update train_epoch Function
```python
def train_epoch(model, train_loader, loss_fn, optimizer, device, scaler, epoch, logger,
                grad_clip=0.0, gradient_accumulation_steps=1, ema_model=None, ema_decay=0.0,
                temperature=1.0, reg_weight_entropy=0.0, reg_weight_confidence=0.0):
    """
    Train for one epoch with temperature annealing and regularization.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    for batch_idx, (bags, labels, _) in enumerate(train_loader):
        bags = bags.to(device)
        labels = labels.to(device)
        
        with autocast(enabled=scaler is not None):
            # Forward pass with temperature
            logits, interpretability = model(bags, return_interpretability=True, temperature=temperature)
            
            # Bag-level loss
            bag_loss = loss_fn(logits, labels)
            
            # Instance-level regularization
            selection_weights = interpretability['selection_weights']  # (B, N)
            attention_weights = interpretability['attention_weights']  # (B, N)
            instance_scores = interpretability['instance_scores']  # (B, N)
            
            reg_loss = 0.0
            
            # Attention entropy loss (encourage diverse attention)
            if reg_weight_entropy > 0:
                # Entropy: -sum(p * log(p))
                entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-10), dim=1)
                entropy_loss = -torch.mean(entropy)  # Negative: we want HIGH entropy
                reg_loss += reg_weight_entropy * entropy_loss
            
            # Selection confidence loss (encourage confident selection)
            if reg_weight_confidence > 0:
                max_score = torch.max(instance_scores, dim=1)[0]
                min_score = torch.min(instance_scores, dim=1)[0]
                confidence_loss = -torch.mean(max_score - min_score)  # Negative: want large separation
                reg_loss += reg_weight_confidence * confidence_loss
            
            # Total loss
            loss = bag_loss + reg_loss
            loss = loss / gradient_accumulation_steps
        
        # ... rest of training loop ...
```

#### 2.7 Update validate Function
```python
def validate(model, val_loader, loss_fn, device, epoch, logger, temperature=1.0):
    """
    Validate with fixed temperature (use temperature_end for evaluation).
    """
    model.eval()
    # ... existing validation code ...
    
    with torch.no_grad():
        for bags, labels, _ in val_loader:
            bags = bags.to(device)
            labels = labels.to(device)
            
            # Forward pass with temperature
            logits = model(bags, temperature=temperature)
            loss = loss_fn(logits, labels)
            # ... rest of validation ...
```

#### 2.8 Update Main Training Loop
```python
# Compute temperature for current epoch
current_temperature = get_temperature(
    epoch, args.epochs, args.temperature_start, args.temperature_end
)

# Train
train_loss, train_acc = train_epoch(
    model, train_loader, loss_fn, optimizer, device, scaler, epoch, logger,
    args.grad_clip, args.gradient_accumulation_steps, ema_model, args.ema_decay,
    temperature=current_temperature,
    reg_weight_entropy=args.reg_weight_entropy,
    reg_weight_confidence=args.reg_weight_confidence
)

# Validate (use final temperature)
val_loss, val_metrics = validate(
    model, val_loader, loss_fn, device, epoch, logger,
    temperature=args.temperature_end
)
```

#### 2.9 Update Model Creation
```python
# REMOVE critical_selection_mode parameter
model = create_dual_stream_mil(
    num_classes=2,
    instance_encoder_backbone=args.instance_encoder_backbone,
    instance_encoder_input_size=args.instance_encoder_input_size,
    # critical_selection_mode removed
    attention_type=args.attention_type,
    fusion_method=args.fusion_method,
    dropout=args.dropout,
    use_hidden_layer=args.use_hidden_layer,
    logger=logger
)
```

#### 2.10 Update Loss Function
```python
# Loss function with label smoothing
loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
logger.info(f"Loss function: CrossEntropyLoss (label_smoothing={args.label_smoothing})")
```

#### 2.11 Update Gradient Clipping Default
```python
parser.add_argument('--grad-clip', type=float, default=0.5,  # Changed from 1.0
                   help='Gradient clipping norm (default: 0.5)')
```

## Testing Command

After implementation, test with:

```bash
python scripts/training/train_dual_stream_mil.py \
  --fold 0 \
  --epochs 60 \
  --batch-size 4 \
  --bag-size 64 \
  --sampling-strategy random \
  --temperature-start 10.0 \
  --temperature-end 1.0 \
  --label-smoothing 0.1 \
  --reg-weight-entropy 0.01 \
  --reg-weight-confidence 0.01 \
  --grad-clip 0.5 \
  --amp
```

## Expected Improvements

1. **Batch Loss Variance**: Reduced by 10-50x (from 500x variation to 10x)
2. **Validation Stability**: Fluctuation reduced from ±0.45 to ±0.05
3. **AUC Improvement**: 0.8786 → 0.90+ (2-3% improvement)
4. **F1-Score**: 0.66-0.70 → 0.75-0.85 (10-15% improvement)
5. **Training Stability**: Smooth convergence, no erratic oscillations

## Implementation Status

- ✅ Model architecture updated (soft selection, temperature support)
- 🔄 Training script updates (in progress)
- ⏳ Testing and validation (pending)

---

**Document Status**: Implementation Guide  
**Last Updated**: January 2025

