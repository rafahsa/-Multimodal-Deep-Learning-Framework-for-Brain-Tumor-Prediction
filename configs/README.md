# Configuration Files Directory

This directory contains configuration files for preprocessing, training, and evaluation.

## Recommended Format

- YAML files for human-readable configurations
- JSON files for programmatic configurations

## Example Structure

- `preprocessing.yaml`: Preprocessing parameters (N4, normalization, cropping, etc.)
- `training.yaml`: Training hyperparameters (learning rate, batch size, epochs, etc.)
- `model_configs/`: Model-specific configurations
  - `resnet50_3d.yaml`
  - `swin_unetr.yaml`
  - `mil.yaml`
- `data_configs/`: Data loading configurations
  - `kfold_split.yaml`: K-fold cross-validation settings

## Best Practices

- Use version control for configuration files
- Document all parameters
- Keep default configurations that work out of the box
- Use environment-specific overrides when needed

