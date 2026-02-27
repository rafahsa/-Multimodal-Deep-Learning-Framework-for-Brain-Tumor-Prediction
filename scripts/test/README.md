# Test Scripts

## run_final_ensemble_inference.py

Final ensemble inference script matching the MICCAI 2026 paper configuration.

### Requirements

- Trained base model checkpoints at:
  - `results/ResNet50-3D/runs/fold_0/run_*/checkpoints/best.pt` (or best_ema.pt)
  - `results/SwinUNETR-3D/runs/fold_0/run_*/checkpoints/best.pt` (or best_ema.pt)
  - `results/DualStreamMIL-3D/runs/fold_0/run_*/checkpoints/best.pt` (or best_ema.pt)
- `ensemble/results/meta_learner_metrics.json`
- `ensemble/results/calibration/2026-02-07_22-29-29_platt_seed42/calibrator_platt.joblib`

### Usage

```bash
# Single patient
python scripts/test/run_final_ensemble_inference.py test/DATA_FOR_TEST/UCSF-PDGM-0004

# Both patients
python scripts/test/run_final_ensemble_inference.py test/DATA_FOR_TEST/UCSF-PDGM-0004 test/DATA_FOR_TEST/UCSF-PDGM-0005

# Dry-run (check paths, no inference)
python scripts/test/run_final_ensemble_inference.py test/DATA_FOR_TEST/UCSF-PDGM-0004 --dry-run
```

### Output

- **Console**: Per-model probabilities, baseline ensemble (uncalibrated and calibrated), predictions at τ=0.41 and τ=0.38
- **CSV**: `test/outputs/final_ensemble_inference_results.csv`

| Column | Meaning |
|--------|---------|
| patient_id | Patient identifier |
| p_hgg_resnet50_3d | ResNet50-3D HGG probability |
| p_hgg_swinunetr_3d | SwinUNETR-3D HGG probability |
| p_hgg_mil_entropy | DualStreamMIL-3D HGG probability (entropy-based slice selection, k=16) |
| ensemble_prob_baseline_uncalibrated | Baseline LR ensemble (meta_learner_metrics.json formula) |
| ensemble_prob_baseline_calibrated | Platt-calibrated ensemble probability |
| pred_balanced_tau_0_41 | Prediction at calibrated threshold 0.41 (balanced) |
| pred_high_sens_tau_0_38 | Prediction at calibrated threshold 0.38 (high-sensitivity) |

### Notes

- Uses baseline formula from meta_learner_metrics.json (coefficients), NOT a joblib meta-learner
- MIL uses entropy-based slice selection with k=16 (paper configuration)
- No ground-truth labels required; outputs probabilities and predicted class at both thresholds
