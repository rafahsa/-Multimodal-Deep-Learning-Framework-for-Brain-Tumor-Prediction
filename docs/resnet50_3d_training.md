# ResNet50-3D Training Guide

## Overview

This document describes the ResNet50-3D training pipeline for brain tumor classification (HGG vs LGG) on the BraTS 2018 dataset.

The pipeline implements:
- **ResNet50-3D Architecture**: 3D CNN for full volume classification
- **MedicalNet Pretrained Weights**: Support for loading pretrained weights from MedicalNet
- **LDAM + DRW Loss**: Label-Distribution-Aware Margin loss with Deferred Re-Weighting
- **GPU-optimized Training**: AMP, TF32, efficient data loading, multi-GPU support
- **K-fold Cross-Validation**: 5-fold CV with comprehensive metrics

## Architecture

### ResNet50-3D Model

The ResNet50-3D model (`models/resnet50_3d_fast/model.py`) consists of:

1. **3D Convolutional Backbone**: Custom ResNet50-3D implementation compatible with MedicalNet
   - Processes full 3D volumes (128x128x128)
   - Uses Bottleneck3D blocks with [3, 4, 6, 3] layers (ResNet50 architecture)
   - Supports single-channel input (1 channel)
   - Fully compatible with MedicalNet pretrained weights
   - Note: We implement our own ResNet50-3D as torchvision.models.video doesn't include r3d_50

2. **Classification Head**: Binary classification (HGG vs LGG)
   - Dropout regularization
   - Linear layer for 2-class output

**Model Size**: ~46.2M parameters (ResNet50-3D with Bottleneck blocks)

**Input**: Full 3D volumes (1, 128, 128, 128) - no slice selection

## Loss Function: LDAM + DRW

### LDAM (Label-Distribution-Aware Margin) Loss

LDAM loss addresses class imbalance by applying class-dependent margins:
- **Formula**: L = -log(exp(s * (logit_y - m_y)) / Σ exp(s * logit_j))
- **Margin**: m_y = C / n_y^(1/4) (larger margins for minority classes)
- **Scaling**: s = 30 (temperature scaling)

### DRW (Deferred Re-Weighting)

DRW applies class weights only after the model has learned basic representations:
- **Before `drw_start_epoch`**: No class weights (standard LDAM)
- **After `drw_start_epoch`**: Apply weights proportional to sqrt(n_max / n_j)

**Default**: `drw_start_epoch = 15` (start re-weighting at epoch 15)

## Training Configuration

### Default Hyperparameters

```python
epochs = 50
batch_size = 4
learning_rate = 1e-4
weight_decay = 1e-4
dropout = 0.5
scheduler = 'cosine'
early_stopping_patience = 7
early_stopping_min_epochs = 10

# LDAM + DRW
max_m = 0.5
s = 30
drw_start_epoch = 15

# Modality
modality = 'flair'
```

### Performance Optimizations

- **Mixed Precision (AMP)**: Enabled by default (faster training, lower memory)
- **TF32**: Enabled by default (faster on Ampere+ GPUs)
- **cuDNN Benchmark**: Enabled for optimized convolutions
- **Pinned Memory**: Enabled for faster CPU→GPU transfers
- **Persistent Workers**: Enabled to avoid worker restart overhead
- **Prefetch Factor**: 4 (prefetch batches while training)

### Data Loading

- **Workers**: Auto (min(8, cpu_count()))
- **Pinned Memory**: True (faster transfers)
- **Persistent Workers**: True (reuse workers across epochs)
- **Prefetch Factor**: 4 (prefetch batches)
- **Drop Last**: True for training (consistent batch sizes)

## Usage

### Single Fold Training

```bash
python scripts/training/train_resnet50_3d.py \
    --fold 0 \
    --modality flair \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4 \
    --amp \
    --tf32
```

### With MedicalNet Pretrained Weights

```bash
python scripts/training/train_resnet50_3d.py \
    --fold 0 \
    --modality flair \
    --pretrained-path /path/to/medicalnet/resnet_50_23dataset.pth \
    --epochs 50 \
    --batch-size 4 \
    --lr 1e-4
```

### Multi-GPU Training

```bash
python scripts/training/train_resnet50_3d.py \
    --fold 0 \
    --modality flair \
    --multi-gpu \
    --batch-size 4 \
    --epochs 50
```

### Custom Hyperparameters

```bash
python scripts/training/train_resnet50_3d.py \
    --fold 0 \
    --modality flair \
    --epochs 50 \
    --batch-size 4 \
    --lr 5e-4 \
    --drw-start-epoch 20 \
    --max-m 0.6 \
    --early-stopping 10 \
    --grad-clip 1.0
```

### Full Command (as specified in requirements)

```bash
python scripts/training/train_resnet50_3d.py \
    --fold 0 \
    --lr 1e-4 \
    --batch-size 4 \
    --epochs 50 \
    --early-stopping 7 \
    --optimizer adam \
    --multi-gpu \
    --gpu
```

## Output Organization

### Per-Fold Run Structure

```
results/ResNet50-3D/runs/fold_{k}/run_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── best.pt          # Best model (by val AUC, tie by F1)
│   └── last.pt          # Last epoch checkpoint
├── metrics/
│   └── metrics.json     # Final validation metrics + full training history (including loss)
├── plots/
│   ├── loss_curve_epoch_X.png       # Per-epoch LDAM loss curve
│   ├── learning_curves_epoch_X.png  # Per-epoch learning curves
│   ├── confusion_matrix_epoch_X.png # Per-epoch confusion matrix
│   ├── roc_curve_epoch_X.png       # Per-epoch ROC curve
│   ├── pr_curve_epoch_X.png         # Per-epoch PR curve
│   ├── loss_curve.png               # Final LDAM loss curve
│   ├── learning_curves.png          # Final learning curves
│   ├── confusion_matrix.png         # Final confusion matrix
│   ├── roc_curve.png                # Final ROC curve
│   └── pr_curve.png                 # Final PR curve
├── predictions/
│   ├── val_probs.npy    # Validation probabilities
│   ├── val_preds.npy    # Validation predictions
│   └── val_labels.npy   # Validation labels
├── config/
│   └── config.json      # Training configuration
└── logs/
    └── train_YYYYMMDD_HHMMSS.log
```

## Metrics

### Loss Tracking

**IMPORTANT**: All loss values tracked are **LDAM loss with DRW**, not cross-entropy.

- **Train Loss (LDAM)**: Average LDAM loss over training set for each epoch
- **Val Loss (LDAM)**: Average LDAM loss over validation set for each epoch
- **Loss History**: Full per-epoch loss history saved in `metrics.json`
- **Loss Summary**: Statistics (min, max, mean, std) for train and val loss
- **Loss Visualization**: Dedicated `loss_curve.png` plot showing train/val loss over epochs

Loss values are:
- Logged to console at each epoch with 6 decimal precision
- Saved to `metrics.json` with full per-epoch history
- Plotted in `loss_curve.png` (dedicated loss plot)
- Also included in `learning_curves.png` (comprehensive plot)
- Stored in checkpoints for analysis

### Classification Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Macro-averaged precision
- **Recall**: Macro-averaged recall
- **F1 Score**: Macro-averaged F1
- **AUC-ROC**: Area under ROC curve

### Per-Class Metrics

- Precision, Recall, F1 per class (LGG, HGG)
- Support (number of samples per class)

### Medical Context

- **False Negatives (FN)**: Missing HGG diagnosis (more serious)
- **False Positives (FP)**: Incorrectly predicting HGG
- **Balanced Metrics**: Macro-averaged metrics consider both classes equally

## Early Stopping

- **Monitor**: Validation AUC (primary), F1 (tie-breaker)
- **Patience**: 7 epochs (default)
- **Min Epochs**: 10 (never stop before this)
- **Min Delta**: 0.0 (any improvement counts)

Early stopping saves the best model (by val AUC, tie by F1).

## Reproducibility

- **Random Seed**: 42 (default, configurable)
- **Checkpoints**: Model state, optimizer state, scheduler state saved
- **Config**: All hyperparameters saved in `config/config.json`

## Performance

### Expected Training Time

- **Single fold**: ~2-4 hours (depends on GPU and data)
- **5-fold CV**: ~10-20 hours (sequential)

### GPU Memory

- **Batch size 4**: ~8-12 GB VRAM (with AMP)
- **Batch size 8**: ~16-20 GB VRAM (with AMP)

### Speed Optimizations

- **AMP**: ~1.5-2x speedup
- **TF32**: ~1.2x speedup (on Ampere+ GPUs)
- **Multi-GPU**: ~Nx speedup (N = number of GPUs)
- **Persistent Workers**: Reduces data loading overhead
- **Prefetch**: Hides data loading latency

## Troubleshooting

### Out of Memory

- Reduce `--batch-size` (try 2 or 1)
- Ensure AMP is enabled (`--amp`)
- Check if other processes are using GPU

### Slow Training

- Increase `--num-workers` (if CPU allows)
- Enable `--tf32` (on Ampere+ GPUs)
- Use `--multi-gpu` if multiple GPUs available
- Check data loading bottleneck (monitor GPU utilization)

### Missing Pretrained Weights

- If `--pretrained-path` is not provided, model uses random initialization
- MedicalNet weights can be downloaded from: https://github.com/Tencent/MedicalNet
- Model will still train without pretrained weights (may need more epochs)

### Poor Performance

- Check class imbalance (should be handled by LDAM+DRW)
- Verify data preprocessing (Stages 1-4)
- Consider adjusting `--drw-start-epoch` or `--max-m`
- Try different learning rates or schedulers

## MedicalNet Pretrained Weights

MedicalNet provides pretrained ResNet50-3D weights trained on 23 diverse medical datasets.

### Download

1. Visit: https://github.com/Tencent/MedicalNet
2. Download the ResNet50-3D pretrained weights
3. Extract and provide path via `--pretrained-path`

### Usage

```bash
python scripts/training/train_resnet50_3d.py \
    --fold 0 \
    --pretrained-path /path/to/medicalnet/resnet_50_23dataset.pth \
    ...
```

The model will automatically:
- Load pretrained weights (allowing partial loading if classification head differs)
- Fine-tune for binary classification task
- Handle weight initialization for modified first conv layer (1 channel input)

## Differences from MIL Training

### Data Processing

- **ResNet50-3D**: Processes full 3D volumes (no slice selection)
- **MIL**: Uses entropy-based slice selection (top-k slices)

### Model Architecture

- **ResNet50-3D**: 3D CNN with spatial convolutions
- **MIL**: 2D CNN with instance-level aggregation

### Input Format

- **ResNet50-3D**: (batch_size, 1, 128, 128, 128) - full volume
- **MIL**: (batch_size, num_slices, 1, H, W) - bag of slices

## Summary

The ResNet50-3D training pipeline provides:

- **Production-grade**: Fast, GPU-optimized, reproducible
- **Medical-focused**: LDAM+DRW for class imbalance, MedicalNet pretrained weights
- **Comprehensive**: Metrics, plots, checkpoints, logging
- **Flexible**: Configurable hyperparameters, K-fold CV, multi-GPU support
- **Full-volume processing**: No slice selection, processes complete 3D volumes

For questions or issues, refer to the training logs in `results/ResNet50-3D/runs/fold_{k}/run_*/logs/`.

