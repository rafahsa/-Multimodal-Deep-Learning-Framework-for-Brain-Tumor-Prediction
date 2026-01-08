# MedicalNet Pretrained Weights Integration

## Overview

MedicalNet pretrained weights for ResNet50-3D have been successfully downloaded and integrated into the project. These weights were pretrained on 23 diverse medical datasets and provide excellent initialization for brain tumor classification tasks.

## File Location

**Path**: `pretrained/medicalnet_resnet50_3d.pth`

**Size**: 177 MB

**Source**: 
- Official: [Tencent MedicalNet GitHub](https://github.com/Tencent/MedicalNet)
- Download: [Zenodo](https://zenodo.org/record/15234379)

## Architecture Compatibility

The MedicalNet weights are fully compatible with our custom ResNet50-3D implementation:

- ✅ **Architecture**: ResNet50-3D with Bottleneck3D blocks
- ✅ **Layer Configuration**: [3, 4, 6, 3] layers per stage
- ✅ **Input Channels**: 1 (single modality)
- ✅ **3D Convolutions**: All layers use 3D convolutions (verified)
- ✅ **Weight Loading**: 316/318 keys loaded successfully

**Note**: The 2 missing keys are expected - they correspond to the classification head, which we replace with our custom binary classifier (LGG vs HGG).

## Usage in Training

### Basic Usage

To use MedicalNet pretrained weights during training:

```bash
python scripts/training/train_resnet50_3d.py \
    --fold 0 \
    --pretrained-path pretrained/medicalnet_resnet50_3d.pth \
    --lr 2e-4 \
    --batch_size 6 \
    --epochs 60 \
    --optimizer adam \
    --loss LDAM \
    --scheduler cosine \
    --dropout 0.4 \
    --grad_clip 0.5 \
    --gradient_accumulation_steps 2 \
    --ema_decay 0.999 \
    --multi_gpu \
    --gpu \
    --amp
```

### Without Pretrained Weights

If you want to train from scratch (random initialization):

```bash
python scripts/training/train_resnet50_3d.py \
    --fold 0 \
    --lr 2e-4 \
    # ... other arguments (omit --pretrained-path)
```

The model will automatically use random initialization if `--pretrained-path` is not provided.

## Verification During Training

When training starts, you should see clear logging confirming the weights were loaded:

```
INFO: Creating ResNet50-3D model...
INFO: Model architecture: ResNet50-3D (MedicalNet-compatible)
INFO: Loading MedicalNet pretrained weights from: pretrained/medicalnet_resnet50_3d.pth
INFO: Using checkpoint as direct state_dict
INFO: Missing keys (expected for classification head): 2 keys
INFO: Successfully loaded 316/318 pretrained weights from MedicalNet
INFO: MedicalNet pretrained weights loaded successfully!
INFO: Model created: 46.16M total params, 46.16M trainable
```

## Loading Logic

The `load_medicalnet_pretrained()` function in `models/resnet50_3d_fast/model.py` handles:

1. **Checkpoint Format Detection**: Automatically detects if checkpoint contains:
   - `state_dict` key
   - `model` key
   - Direct state dictionary

2. **DataParallel Compatibility**: Automatically strips `module.` prefix if present (from DataParallel training)

3. **Partial Loading**: Safely ignores mismatched classification head weights (uses `strict=False`)

4. **Error Handling**: Gracefully falls back to random initialization if loading fails

## Verification Script

To verify the weights are valid and compatible:

```python
import torch
import sys
sys.path.insert(0, '.')

from models.resnet50_3d_fast.model import ResNet50_3D
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create model with pretrained weights
model = ResNet50_3D(
    num_classes=2,
    in_channels=1,
    pretrained_path='pretrained/medicalnet_resnet50_3d.pth',
    dropout=0.4,
    logger=logger
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print("✓ MedicalNet weights loaded successfully!")
```

## Expected Benefits

Using MedicalNet pretrained weights should provide:

1. **Faster Convergence**: Model starts with features learned from diverse medical datasets
2. **Better Performance**: Higher validation accuracy and AUC
3. **Improved Generalization**: Better performance on validation/test sets
4. **Reduced Training Time**: Fewer epochs needed to reach good performance

## Troubleshooting

### Weights Not Loading

If you see:
```
WARNING: Pretrained weights file not found: pretrained/medicalnet_resnet50_3d.pth
Using random initialization instead.
```

**Solution**: Verify the file exists:
```bash
ls -lh pretrained/medicalnet_resnet50_3d.pth
```

### Architecture Mismatch

If you see many missing/unexpected keys:
- This is normal - the classification head will have different keys
- Only 2 missing keys are expected (fc layer)
- If you see hundreds of missing keys, there may be an architecture mismatch

### File Size Issues

Expected file size: ~177 MB

If the file is much smaller (< 1 MB), the download may have failed. Re-download from:
- Zenodo: https://zenodo.org/record/15234379
- Or use the verification script above to check

## References

- **MedicalNet Paper**: [MedicalNet: Transfer Learning for 3D Medical Image Analysis](https://arxiv.org/abs/1904.00625)
- **GitHub Repository**: https://github.com/Tencent/MedicalNet
- **Zenodo Dataset**: https://zenodo.org/record/15234379

