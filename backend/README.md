# Brain Tumor Prediction API

FastAPI backend for the NeuroGrade multimodal ensemble (ResNet50-3D, SwinUNETR-3D, DualStreamMIL-3D) with Platt-calibrated HGG/LGG classification.

## Prerequisites

- Python 3.10+
- Model checkpoints under `results/` (or path set via `MODEL_BASE_DIR`)
- Meta-learner metrics: `ensemble/results/meta_learner_metrics.json`
- Platt calibrator: `ensemble/results/calibration/2026-02-07_22-29-29_platt_seed42/calibrator_platt.joblib`
- NVIDIA GPU recommended (CUDA); CPU supported but slower

## Setup

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

First startup loads all three models (typically 30–60 seconds). Check readiness:

```bash
curl http://localhost:8000/api/health
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |
| `MODEL_BASE_DIR` | `../results` | Checkpoint root directory |
| `METRICS_PATH` | `../ensemble/results/meta_learner_metrics.json` | Meta-learner coefficients |
| `CALIBRATOR_PATH` | `../ensemble/results/calibration/.../calibrator_platt.joblib` | Platt scaler path |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated allowed origins |
| `MAX_UPLOAD_MB` | `500` | Maximum total multipart upload size |
| `LOG_LEVEL` | `INFO` | Python logging level |

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Model load status and device info |
| `POST` | `/api/predict` | Upload 4 NIfTI modalities (multipart: `t1`, `t1ce`, `t2`, `flair`) |

Full request/response schemas: [`specs/001-brain-tumor-prediction-ui/contracts/api-contract.md`](../specs/001-brain-tumor-prediction-ui/contracts/api-contract.md)

Interactive docs (when server is running): [http://localhost:8000/docs](http://localhost:8000/docs)

## Project layout

```
backend/
├── app/
│   ├── main.py           # FastAPI app, CORS, lifespan
│   ├── config.py         # Settings from environment
│   ├── api/
│   │   ├── routes.py     # /api/predict, /api/health
│   │   └── schemas.py    # Pydantic models
│   └── services/
│       ├── inference.py  # Model registry
│       ├── preprocessing.py
│       └── calibration.py
└── requirements.txt
```

Inference logic is adapted from `scripts/test/run_final_ensemble_inference.py` at the repository root.

## Quick test

Use sample NIfTI volumes from `test/DATA_FOR_TEST/` (see [quickstart.md](../specs/001-brain-tumor-prediction-ui/quickstart.md)).
