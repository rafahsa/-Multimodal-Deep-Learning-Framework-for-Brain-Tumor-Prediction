# Scripts Directory

This directory contains all executable scripts for the project.

## Structure

- `preprocessing/`: Data preprocessing scripts
  - Each preprocessing stage should have its own script
  - Scripts read from `../../data/raw/` and write to `../../data/processed/`
  
- `training/`: Model training scripts
  - Scripts for training ResNet50-3D, Swin UNETR, and MIL models
  
- `evaluation/`: Model evaluation scripts
  - Scripts for evaluating trained models
  - Generate metrics, plots, and predictions
  
- `utils/`: Utility functions
  - Shared functions used across preprocessing, training, and evaluation
  - Data loaders, visualization tools, etc.

## Best Practices

- All scripts should use relative paths from the project root
- Log all operations to `../../logs/`
- Use configuration files from `../../configs/` for parameters
- Ensure reproducibility by setting random seeds

