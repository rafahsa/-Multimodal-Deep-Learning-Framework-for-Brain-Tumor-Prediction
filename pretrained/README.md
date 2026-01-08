# Pretrained Weights Directory

This directory contains pretrained model weights for the brain tumor classification project.

## MedicalNet ResNet50-3D Weights

**File**: `medicalnet_resnet50_3d.pth`

**Source**: MedicalNet (Tencent) - ResNet50-3D pretrained on 23 diverse medical datasets

**Download**: 
- Original source: [Tencent MedicalNet GitHub](https://github.com/Tencent/MedicalNet)
- Alternative source: [Zenodo](https://zenodo.org/record/15234379)
- File size: ~177 MB

**Architecture Compatibility**:
- ResNet50-3D with Bottleneck3D blocks
- Layer configuration: [3, 4, 6, 3]
- Compatible with our custom ResNet50-3D implementation
- 316/318 keys loaded (classification head uses custom weights for binary classification)

**Usage in Training**:

```bash
python scripts/training/train_resnet50_3d.py \
    --fold 0 \
    --pretrained-path pretrained/medicalnet_resnet50_3d.pth \
    --lr 2e-4 \
    --batch_size 6 \
    --epochs 60 \
    --optimizer adam \
    --loss LDAM \
    --multi_gpu \
    --gpu \
    --amp
```

**Verification**:

The training script will log confirmation when pretrained weights are loaded:

```
INFO: Loading MedicalNet pretrained weights from: pretrained/medicalnet_resnet50_3d.pth
INFO: Using checkpoint as direct state_dict
INFO: Missing keys (expected for classification head): 2 keys
INFO: Successfully loaded 316/318 pretrained weights from MedicalNet
INFO: MedicalNet pretrained weights loaded successfully!
```

**Note**: The missing keys are expected - they correspond to the classification head, which we replace with our custom binary classifier (LGG vs HGG).

## File Verification

To verify the weights file is valid:

```python
import torch

checkpoint = torch.load('pretrained/medicalnet_resnet50_3d.pth', map_location='cpu', weights_only=False)
print(f"Checkpoint keys: {len(checkpoint)}")
print(f"Sample keys: {list(checkpoint.keys())[:5]}")
```

Expected output:
- Total keys: 318
- Contains 3D convolution layers (conv1.weight shape: [64, 1, 7, 7, 7])
- No classification head (fc layer) - we use custom head

