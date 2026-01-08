# Dual-Stream MIL Training Pipeline

## Overview

This document describes the Dual-Stream Multiple Instance Learning (MIL) training pipeline for brain tumor classification (HGG vs LGG) on the BraTS2018 dataset.

The pipeline implements:
- **Dual-Stream MIL Architecture**: Max-pooling + Gated Attention aggregation
- **Entropy-based Slice Selection**: Always enabled for MIL (top-k informative slices)
- **LDAM + DRW Loss**: Label-Distribution-Aware Margin loss with Deferred Re-Weighting
- **GPU-optimized Training**: AMP, TF32, efficient data loading
- **K-fold Cross-Validation**: 5-fold CV with aggregated results

## Architecture

### Dual-Stream MIL Model

The Dual-Stream MIL model (`models/dual_stream_mil/model.py`) consists of:

1. **Instance Encoder**: ResNet34 adapted for 1-channel 2D slices (UPGRADED from ResNet18)
   - Converts each slice to a 512-dimensional feature vector
   - Pretrained ImageNet weights used by default
   - Richer feature representations for complex slice-level patterns

2. **Stream 1: Max-Pooling Aggregation**
   - Takes maximum across all instance features
   - Captures strongest responses across slices

3. **Stream 2: Gated Attention Aggregation**
   - Learns attention weights for each slice
   - Computes weighted aggregation of instance features
   - Provides interpretability through attention weights

4. **Fusion**: Concatenates features from both streams (1024-dim)

5. **Classifier**: Binary classification head (HGG vs LGG)

**Model Size**: ~21.3M parameters (ResNet34 encoder)

**Multi-Modality Version**: See `train_mil_multi_modal.py` for FLAIR+T1ce multi-modality MIL

### Why Dual-Stream?

- **Max-pooling**: Captures most discriminative features
- **Gated Attention**: Learns which slices are most relevant
- **Fusion**: Combines complementary information from both streams
- **Interpretability**: Attention weights show which slices the model focuses on

## Entropy-based Slice Selection

### Always Enabled for MIL

**IMPORTANT**: Entropy-based slice selection is **ALWAYS enabled** for MIL training (not optional).

The pipeline:
1. Loads pre-computed entropy metadata from `data/entropy/<patient_id>_entropy.json`
2. Selects top-k slices (default: 16) based on entropy scores
3. Uses only these slices for training (reduces computation from 128 to 16 slices)

### Why Entropy for MIL?

- **MIL Architecture**: MIL models process 2D slices as instances
- **Computational Efficiency**: Reduces computation by focusing on informative slices
- **Discriminative Power**: High-entropy slices contain diverse tissue patterns
- **Medical Relevance**: Tumor regions typically have high entropy

### Why NOT for 3D CNNs?

- **ResNet50-3D** and **Swin UNETR** process full 3D volumes
- Slice selection would destroy 3D spatial context
- 3D convolutions require complete volumes
- Entropy selection is MIL-specific by design

See `docs/stage_entropy_mil.md` for detailed explanation.

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

### Why LDAM + DRW?

- **LDAM**: Handles imbalance through margins (works from epoch 1)
- **DRW**: Additional re-weighting after model convergence (avoids early instability)
- **Medical Context**: HGG vs LGG is imbalanced (210 vs 75 cases)
- **Best Practice**: Combines margin-based and re-weighting strategies

## Training Configuration

### Default Hyperparameters

```python
epochs = 30
batch_size = 2
learning_rate = 1e-3
weight_decay = 1e-4
dropout = 0.5
scheduler = 'cosine'
early_stopping_patience = 7
early_stopping_min_epochs = 10

# LDAM + DRW
max_m = 0.5
s = 30
drw_start_epoch = 15

# Entropy
top_k = 16  # Always enabled
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
python scripts/training/train_mil.py \
    --fold 0 \
    --modality flair \
    --top-k 16 \
    --epochs 30 \
    --batch-size 2 \
    --amp \
    --tf32
```

### K-Fold Cross-Validation

```bash
python scripts/training/run_mil_kfold.py \
    --folds 0,1,2,3,4 \
    --modality flair \
    --top-k 16 \
    --epochs 30 \
    --batch-size 2 \
    --amp \
    --tf32
```

### Custom Hyperparameters

```bash
python scripts/training/train_mil.py \
    --fold 0 \
    --modality flair \
    --top-k 16 \
    --epochs 50 \
    --batch-size 4 \
    --lr 5e-4 \
    --drw-start-epoch 20 \
    --max-m 0.6 \
    --early-stopping-patience 10
```

## Output Organization

### Per-Fold Run Structure

```
results/MIL/runs/fold_{k}/run_YYYYMMDD_HHMMSS/
├── checkpoints/
│   ├── best.pt          # Best model (by val AUC, tie by F1)
│   └── last.pt          # Last epoch checkpoint
├── metrics/
│   └── metrics.json     # Final validation metrics
├── plots/
│   ├── learning_curves.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── predictions/
│   ├── val_probs.npy    # Validation probabilities
│   ├── val_preds.npy    # Validation predictions
│   └── val_labels.npy   # Validation labels
├── config/
│   └── config.json      # Training configuration
└── logs/
    └── train_YYYYMMDD_HHMMSS.log
```

### Latest Run Pointer

```
results/MIL/latest/fold_{k}/LATEST_RUN.txt
```

Contains path to the most recent run directory for each fold.

### K-Fold Summary

```
results/MIL/kfold_summary/
├── summary.json         # Detailed per-fold metrics
├── summary.csv          # Table format (per-fold)
└── mean_std.json        # Aggregated statistics (mean±std)
```

## Metrics

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
- **Deterministic**: Entropy slice selection is deterministic
- **Checkpoints**: Model state, optimizer state, scheduler state saved
- **Config**: All hyperparameters saved in `config/config.json`

## Performance

### Expected Training Time

- **Single fold**: ~30-60 minutes (depends on GPU and data)
- **5-fold CV**: ~2.5-5 hours (sequential)

### GPU Memory

- **Batch size 2**: ~4-6 GB VRAM (with AMP)
- **Batch size 4**: ~8-12 GB VRAM (with AMP)

### Speed Optimizations

- **AMP**: ~1.5-2x speedup
- **TF32**: ~1.2x speedup (on Ampere+ GPUs)
- **Persistent Workers**: Reduces data loading overhead
- **Prefetch**: Hides data loading latency

## Troubleshooting

### Out of Memory

- Reduce `--batch-size` (try 1 or 2)
- Ensure AMP is enabled (`--amp`)
- Check if other processes are using GPU

### Slow Training

- Increase `--num-workers` (if CPU allows)
- Enable `--tf32` (on Ampere+ GPUs)
- Check data loading bottleneck (monitor GPU utilization)

### Missing Entropy Files

- Run entropy analysis first:
  ```bash
  python scripts/analysis/run_entropy_analysis.py --modality flair --axis axial --top-k 16
  ```

### Poor Performance

- Check class imbalance (should be handled by LDAM+DRW)
- Verify entropy files are correct
- Check data preprocessing (Stages 1-4)
- Consider adjusting `--drw-start-epoch` or `--max-m`

## Integration with Other Models

### Separation of Concerns

- **MIL**: Uses `utils/mil_dataset.py` with entropy (always enabled)
- **ResNet50-3D**: Uses separate dataset (full 3D volumes, no entropy)
- **Swin UNETR**: Uses separate dataset (full 3D volumes, no entropy)

### No Cross-Contamination

- Entropy selection is MIL-only
- 3D CNN datasets remain unchanged
- No modifications to preprocessing pipeline
- Each model uses data in its optimal format

## References

- **LDAM Loss**: Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss" (NeurIPS 2019)
- **Gated Attention MIL**: Ilse et al., "Attention-based Deep Multiple Instance Learning" (ICML 2018)
- **DRW Strategy**: Standard practice for class imbalance in medical imaging
- **BraTS2018**: https://www.med.upenn.edu/cbica/brats2018/

## Summary

The Dual-Stream MIL training pipeline provides:

- **Production-grade**: Fast, GPU-optimized, reproducible
- **Medical-focused**: LDAM+DRW for class imbalance, entropy for efficiency
- **Comprehensive**: Metrics, plots, checkpoints, logging
- **Flexible**: Configurable hyperparameters, K-fold CV
- **Separated**: MIL-only entropy, no impact on 3D CNNs

For questions or issues, refer to the training logs in `results/MIL/runs/fold_{k}/run_*/logs/`.

