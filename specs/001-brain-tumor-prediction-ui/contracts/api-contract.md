# API Contract: Brain Tumor Prediction Backend

**Branch**: `001-brain-tumor-prediction-ui` | **Date**: 2026-05-21

## Base URL

- Development: `http://localhost:8000`
- Production: Configurable via `API_BASE_URL` environment variable

## Endpoints

### POST /api/predict

Upload 4 NIfTI modality files and receive an ensemble tumor grade prediction.

**Content-Type**: `multipart/form-data`

#### Request

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| t1 | File (.nii / .nii.gz) | Yes | T1-weighted MRI modality |
| t1ce | File (.nii / .nii.gz) | Yes | T1 contrast-enhanced MRI modality |
| t2 | File (.nii / .nii.gz) | Yes | T2-weighted MRI modality |
| flair | File (.nii / .nii.gz) | Yes | FLAIR MRI modality |
| patient_label | string | No | Optional display label (default: auto-generated from filename) |

**Constraints**:
- Each file must be a valid NIfTI volume (.nii or .nii.gz extension)
- Individual file size: ≤ 200 MB
- Total upload size: ≤ 500 MB
- All 4 files are required — partial uploads are rejected

#### Response (200 OK)

```json
{
  "prediction_id": "uuid-v4",
  "patient_label": "UCSF-PDGM-0004",
  "calibrated_probability": 0.8723,
  "uncalibrated_probability": 0.8451,
  "model_probabilities": {
    "resnet": 0.9102,
    "swinunetr": 0.8856,
    "mil": 0.7934
  },
  "ensemble_logit": 1.923,
  "meta_learner_coefficients": {
    "resnet": 0.537,
    "swinunetr": 4.063,
    "mil": 0.890,
    "intercept": -2.405
  },
  "thresholds": {
    "balanced": {
      "value": 0.41,
      "classification": "HGG",
      "mode_name": "Balanced Screening"
    },
    "high_sensitivity": {
      "value": 0.38,
      "classification": "HGG",
      "mode_name": "High-Sensitivity Triage"
    }
  },
  "processing_duration_ms": 45230,
  "timestamp": "2026-05-21T18:05:23.456Z",
  "device_used": "cuda"
}
```

| Field | Type | Description |
|-------|------|-------------|
| prediction_id | string (UUID v4) | Unique prediction identifier |
| patient_label | string | Display label for the scan |
| calibrated_probability | float [0, 1] | Platt-calibrated P(HGG) |
| uncalibrated_probability | float [0, 1] | Raw ensemble P(HGG) before Platt |
| model_probabilities | object | Per-model P(HGG) values |
| model_probabilities.resnet | float [0, 1] | ResNet50-3D prediction |
| model_probabilities.swinunetr | float [0, 1] | SwinUNETR-3D prediction |
| model_probabilities.mil | float [0, 1] | DualStreamMIL-3D prediction |
| ensemble_logit | float | Raw logit before sigmoid: intercept + Σ(coef × p) |
| meta_learner_coefficients | object | Logistic regression coefficients (static, for display) |
| thresholds | object | Pre-computed classification at both operating points |
| thresholds.balanced | object | Result at τ=0.41 |
| thresholds.high_sensitivity | object | Result at τ=0.38 |
| processing_duration_ms | integer | Total backend processing time |
| timestamp | ISO 8601 | Server-side prediction completion time |
| device_used | "cuda" \| "cpu" | Which compute device was used |

#### Error Responses

**422 Unprocessable Entity** — Validation failure

```json
{
  "detail": [
    {
      "loc": ["body", "t1"],
      "msg": "Missing required modality file: T1",
      "type": "value_error.missing"
    }
  ]
}
```

**400 Bad Request** — Invalid NIfTI file

```json
{
  "error": "invalid_file",
  "message": "File 'scan_t2.nii.gz' is not a valid NIfTI volume. The file could not be parsed as a NIfTI image.",
  "field": "t2",
  "suggestion": "Ensure the file is a valid NIfTI (.nii or .nii.gz) brain MRI volume. Files exported from FreeSurfer, FSL, or 3D Slicer are typically compatible."
}
```

**413 Payload Too Large** — File size exceeded

```json
{
  "error": "file_too_large",
  "message": "Total upload size (623 MB) exceeds the 500 MB limit.",
  "max_size_mb": 500,
  "actual_size_mb": 623
}
```

**500 Internal Server Error** — Inference failure

```json
{
  "error": "inference_error",
  "message": "Model inference failed during SwinUNETR-3D prediction. This may be caused by insufficient GPU memory or an incompatible input volume shape.",
  "suggestion": "Try again. If the error persists, ensure the uploaded files are standard brain MRI volumes with typical dimensions."
}
```

**503 Service Unavailable** — Models not loaded

```json
{
  "error": "models_not_ready",
  "message": "The prediction service is still loading models. Please wait and try again in 30 seconds.",
  "retry_after_seconds": 30
}
```

---

### GET /api/health

Health check endpoint for monitoring.

#### Response (200 OK)

```json
{
  "status": "healthy",
  "models_loaded": true,
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 3090",
  "version": "1.0.0"
}
```

#### Response (503 Service Unavailable)

```json
{
  "status": "loading",
  "models_loaded": false,
  "message": "Models are still loading. Estimated time: 30 seconds."
}
```
