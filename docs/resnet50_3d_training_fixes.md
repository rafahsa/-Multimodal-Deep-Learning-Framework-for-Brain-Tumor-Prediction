# ResNet50-3D Training Pipeline: Critical Fixes for Stability and Performance

## Executive Summary

The training pipeline was experiencing poor convergence and unstable validation metrics (AUC ~0.72 peak, collapsing to ~0.51) despite correct execution. A comprehensive audit identified **6 critical issues** in the training strategy that were preventing effective learning from pretrained MedicalNet weights.

## Root Causes Identified

### 1. **No Differential Learning Rates** ⚠️ CRITICAL
**Problem**: All model parameters (pretrained backbone + new classifier) used the same learning rate (2e-4), which is too high for fine-tuning pretrained weights.

**Impact**: 
- Pretrained features were being destroyed by excessive updates
- Classifier couldn't adapt quickly enough relative to backbone changes
- Training instability and poor convergence

**Fix**: Implemented differential learning rates:
- Backbone (pretrained): 1e-4 (10x lower, preserves pretrained features)
- Classifier (new): 1e-3 (10x higher, allows rapid adaptation)

### 2. **LDAM Loss Too Aggressive** ⚠️ HIGH PRIORITY
**Problem**: LDAM parameters (max_m=0.3, s=20) combined with DRW were too aggressive for pretrained weights.

**Impact**:
- Excessive margin penalties causing gradient instability
- Loss values too high and unstable
- Model struggling to learn from pretrained initialization

**Fix**: Reduced LDAM parameters:
- max_m: 0.3 → 0.2 (33% reduction)
- s: 20 → 15 (25% reduction)
- More conservative margins preserve pretrained feature quality

### 3. **DRW Starting Too Early** ⚠️ HIGH PRIORITY
**Problem**: DRW re-weighting started at epoch 15, before model learned basic features.

**Impact**:
- Premature class re-weighting disrupted feature learning
- Model hadn't adapted to task before applying aggressive re-weighting
- Validation metrics collapsed when DRW activated

**Fix**: Delayed DRW start:
- Epoch 15 → Epoch 25 (67% delay)
- Allows 25 epochs of stable feature learning before re-weighting

### 4. **Gradient Clipping Too Aggressive** ⚠️ MEDIUM PRIORITY
**Problem**: Gradient clipping at 0.5 was too restrictive, limiting learning capacity.

**Impact**:
- Important gradients being clipped
- Slower convergence
- Reduced ability to fine-tune pretrained features

**Fix**: Increased gradient clipping threshold:
- 0.5 → 1.0 (2x increase)
- Allows larger gradient updates while maintaining stability

### 5. **EMA Decay Too High** ⚠️ MEDIUM PRIORITY
**Problem**: EMA decay of 0.999 updated too slowly, interfering with training dynamics.

**Impact**:
- EMA model lagging too far behind training model
- Poor representation of current training state
- Validation metrics using outdated model state

**Fix**: Reduced EMA decay:
- 0.999 → 0.995 (faster adaptation)
- Better balance between stability and responsiveness

### 6. **No Gradual Unfreezing Strategy** ⚠️ OPTIONAL
**Problem**: All layers trained from start, no staged fine-tuning approach.

**Impact**:
- Potential for catastrophic forgetting of pretrained features
- Less control over fine-tuning process

**Fix**: Added optional backbone freezing:
- New argument: `--freeze-backbone-epochs` (default: 0)
- Can freeze backbone for initial epochs, then unfreeze gradually
- Provides fine-grained control over fine-tuning

## Code Changes Summary

### 1. Model Architecture (`models/resnet50_3d_fast/model.py`)

**Added methods for differential learning rates**:
```python
def get_backbone_params(self):
    """Get parameters from pretrained backbone (all except classifier)"""
    for name, param in self.model.named_parameters():
        if 'fc' not in name:
            yield param

def get_classifier_params(self):
    """Get parameters from classification head"""
    for name, param in self.model.named_parameters():
        if 'fc' in name:
            yield param
```

### 2. Training Script (`scripts/training/train_resnet50_3d.py`)

#### A. New Arguments
- `--classifier-lr`: Separate LR for classifier (default: 10x backbone LR)
- `--freeze-backbone-epochs`: Optional backbone freezing (default: 0)
- Updated defaults:
  - `--lr`: 2e-4 → 1e-4 (backbone)
  - `--max-m`: 0.3 → 0.2
  - `--s`: 20 → 15
  - `--drw-start-epoch`: 15 → 25
  - `--grad-clip`: 0.5 → 1.0
  - `--ema-decay`: 0.999 → 0.995

#### B. Differential Learning Rate Optimizer
```python
# Separate parameter groups
backbone_params = list(model.get_backbone_params())
classifier_params = list(model.get_classifier_params())

optimizer = torch.optim.Adam(
    [
        {'params': backbone_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': classifier_params, 'lr': classifier_lr, 'weight_decay': args.weight_decay}
    ],
    betas=(0.9, 0.999),
    eps=1e-8
)
```

#### C. Differential Learning Rate Scheduler
```python
# Separate LR schedules for backbone and classifier
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=[lr_lambda_backbone, lr_lambda_classifier]
)
```

#### D. Gradual Unfreezing Support
```python
# Freeze backbone if requested
if args.freeze_backbone_epochs > 0:
    for param in backbone_params:
        param.requires_grad = False

# Unfreeze during training
if epoch == args.freeze_backbone_epochs + 1:
    for param in backbone_params:
        param.requires_grad = True
```

## Expected Improvements

### Training Stability
- ✅ **Smoother loss curves**: Reduced LDAM aggressiveness prevents gradient explosions
- ✅ **Stable validation metrics**: Delayed DRW prevents premature metric collapse
- ✅ **Better convergence**: Differential LRs allow proper fine-tuning

### Performance Metrics
- **Target**: Validation accuracy ~90%, AUC >0.90
- **Previous**: AUC peaked at ~0.72, collapsed to ~0.51
- **Expected**: Consistent AUC >0.85 with stable training

### Training Dynamics
- **Backbone**: Slow, conservative updates preserve pretrained features
- **Classifier**: Fast adaptation to task-specific patterns
- **Loss**: More stable with reduced LDAM margins
- **DRW**: Activates after sufficient feature learning

## Recommended Training Command

```bash
python scripts/training/train_resnet50_3d.py \
    --fold 0 \
    --pretrained-path pretrained/medicalnet_resnet50_3d.pth \
    --lr 1e-4 \
    --classifier-lr 1e-3 \
    --batch_size 6 \
    --epochs 60 \
    --optimizer adam \
    --loss LDAM \
    --scheduler cosine \
    --max-m 0.2 \
    --s 15 \
    --drw-start-epoch 25 \
    --grad-clip 1.0 \
    --ema-decay 0.995 \
    --dropout 0.4 \
    --gradient_accumulation_steps 2 \
    --multi_gpu \
    --gpu \
    --amp
```

## Key Principles Applied

1. **Preserve Pretrained Features**: Lower LR for backbone prevents catastrophic forgetting
2. **Rapid Classifier Adaptation**: Higher LR for classifier allows quick task adaptation
3. **Staged Learning**: Delayed DRW allows feature learning before re-weighting
4. **Conservative Loss**: Reduced LDAM margins prevent gradient instability
5. **Balanced Regularization**: Gradient clipping and EMA tuned for stability

## Monitoring During Training

### Success Indicators
- ✅ Loss decreases smoothly (no spikes)
- ✅ Validation AUC increases steadily
- ✅ No sudden metric collapses
- ✅ Backbone LR < Classifier LR (check logs)
- ✅ DRW activates at epoch 25 (check logs)

### Warning Signs
- ⚠️ Loss spikes or NaN values
- ⚠️ Validation metrics decreasing after DRW activation
- ⚠️ AUC not improving beyond 0.70
- ⚠️ Large gap between train and validation metrics

## Ablation Study Recommendations

To verify improvements, consider:

1. **Baseline**: Original settings (LR=2e-4, max_m=0.3, DRW=15)
2. **Differential LR only**: Test impact of separate LRs
3. **Reduced LDAM only**: Test impact of lower margins
4. **Delayed DRW only**: Test impact of later DRW activation
5. **Combined**: All fixes together (recommended)

## Technical Details

### Parameter Counts
- **Backbone**: ~46.16M parameters (pretrained)
- **Classifier**: ~4K parameters (new, trainable)
- **Ratio**: ~11,500:1 (backbone:classifier)

### Learning Rate Ratio
- **Backbone LR**: 1e-4 (preserves pretrained features)
- **Classifier LR**: 1e-3 (10x higher for adaptation)
- **Ratio**: 10:1 (classifier:backbone)

### Loss Function Evolution
- **Epochs 1-24**: LDAM only (no DRW re-weighting)
- **Epoch 25+**: LDAM + DRW (class re-weighting activated)

## Conclusion

These fixes address fundamental issues in fine-tuning pretrained models:
1. **Differential learning rates** are essential for pretrained weights
2. **Conservative loss functions** prevent gradient instability
3. **Staged training strategies** (delayed DRW) improve stability
4. **Proper regularization** balances learning and stability

The pipeline should now achieve stable, high-performance training with MedicalNet pretrained weights.

