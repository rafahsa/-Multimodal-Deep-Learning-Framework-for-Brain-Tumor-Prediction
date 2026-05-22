# Quickstart: Brain Tumor Prediction Web Interface

**Branch**: `001-brain-tumor-prediction-ui` | **Date**: 2026-05-21

## Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- Trained model checkpoints in `results/` directory (or `archive_minimal_runs/` for demo)
- Platt calibrator at `ensemble/results/calibration/2026-02-07_22-29-29_platt_seed42/calibrator_platt.joblib`
- GPU recommended (NVIDIA with CUDA support); CPU fallback available but slower (~2-5 min per prediction)

## 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server loads all three models at startup (30-60 seconds on first launch). Check readiness:

```bash
curl http://localhost:8000/api/health
```

### Environment Variables (optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `auto` | Force `cuda` or `cpu` (auto-detects by default) |
| `MODEL_BASE_DIR` | `../results` | Path to model checkpoint directories |
| `CORS_ORIGINS` | `http://localhost:5173` | Allowed CORS origins (comma-separated) |
| `MAX_UPLOAD_MB` | `500` | Maximum total upload size |
| `LOG_LEVEL` | `INFO` | Python logging level |

## 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The React app starts at `http://localhost:5173` and connects to the backend at `http://localhost:8000`.

## 3. Usage

1. Open `http://localhost:5173` in your browser
2. Upload 4 NIfTI files — one for each modality slot (T1, T1ce, T2, FLAIR)
3. Select an operating mode (Balanced Screening or High-Sensitivity Triage)
4. Click **Predict** and wait for results
5. View the classification (HGG/LGG), probability score, and model breakdown
6. Previous predictions appear in the session history sidebar

## 4. Test Data

Sample patient data is available at:

```
test/DATA_FOR_TEST/UCSF-PDGM-0004/
├── UCSF-PDGM-0004_T1.nii.gz
├── UCSF-PDGM-0004_T1ce.nii.gz
├── UCSF-PDGM-0004_T2.nii.gz
└── UCSF-PDGM-0004_FLAIR.nii.gz

test/DATA_FOR_TEST/UCSF-PDGM-0005/
├── UCSF-PDGM-0005_T1.nii.gz
├── UCSF-PDGM-0005_T1ce.nii.gz
├── UCSF-PDGM-0005_T2.nii.gz
└── UCSF-PDGM-0005_FLAIR.nii.gz
```

## 5. Production Build

```bash
# Build frontend
cd frontend
npm run build
# Output in frontend/dist/

# Serve backend (production)
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

For production, serve the frontend static files from `frontend/dist/` behind a reverse proxy (nginx) alongside the FastAPI backend, both on the same origin to avoid CORS.
