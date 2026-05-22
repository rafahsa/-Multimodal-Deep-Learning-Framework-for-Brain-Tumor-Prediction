# Implementation Plan: Brain Tumor Prediction Web Interface

**Branch**: `001-brain-tumor-prediction-ui` | **Date**: 2026-05-21 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/001-brain-tumor-prediction-ui/spec.md`

## Summary

Build a React + Python web application that allows users to upload 4 NIfTI brain MRI modalities (T1, T1ce, T2, FLAIR), run ensemble inference through the existing three-model pipeline (ResNet50-3D, SwinUNETR-3D, DualStreamMIL-3D), and display calibrated tumor grade predictions (HGG vs LGG) with operating mode selection and model breakdown. The React frontend adopts the NeuroGrade visual identity from `index.html`. The Python backend wraps the existing CLI inference script (`scripts/test/run_final_ensemble_inference.py`) as a FastAPI service.

## Technical Context

**Language/Version**: Python 3.10+ (backend), TypeScript/JavaScript (React frontend)

**Primary Dependencies**:
- Backend: FastAPI, uvicorn, python-multipart, torch, monai, SimpleITK, numpy, scikit-learn, joblib, pandas
- Frontend: React 18, TypeScript, Vite, Axios

**Storage**: N/A — no database. Files are processed in-memory/tempdir and discarded. Session history is browser-side only (React state).

**Testing**: pytest (backend unit + integration), React Testing Library + Vitest (frontend)

**Target Platform**: Web application — Python backend on Linux/Windows server with GPU (or CPU fallback), React SPA served by Vite dev server or static build

**Project Type**: Web application (frontend + backend)

**Performance Goals**: End-to-end inference (upload → result) under 2 minutes on GPU, under 5 minutes on CPU. Frontend interactions (mode switching, history navigation) under 200ms.

**Constraints**: Single-user inference (no queue), max 500 MB upload, models loaded once at startup and kept in GPU memory

**Scale/Scope**: Single-user research/demo tool, ~5 React pages/views, 1 API endpoint for prediction

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Code Quality & Reproducibility | PASS | Seeds used in inference pipeline (seed=42); preprocessing is deterministic; config externalized to YAML/JSON |
| I. Patient-level data isolation | PASS | Web app does inference only — no training splits involved |
| I. No magic numbers | PASS | Thresholds (0.41, 0.38), coefficients, and model params loaded from `meta_learner_metrics.json` and config files |
| II. Testing Standards | PASS | Plan includes unit tests for API, contract tests for model I/O shapes, integration test for upload→predict flow |
| III. User Experience Consistency | PASS | Error messages include file path + suggested remediation; progress reporting during inference; consistent output format |
| IV. Performance Requirements | PASS | Single-patient inference targets <30s on GPU (per constitution); web latency budget is 2 min total including upload + preprocessing |
| Technical Constraints: Python 3.10+ | PASS | Backend is Python |
| Technical Constraints: PyTorch + MONAI | PASS | Existing model code uses both |
| Technical Constraints: NIfTI format | PASS | Upload accepts .nii/.nii.gz only |
| Technical Constraints: Config in YAML/JSON | PASS | Model config from JSON; preprocessing config from YAML; API config from environment variables |
| Technical Constraints: Logging (not print) | PASS | Backend uses Python `logging` module |
| Technical Constraints: Version pinning | NEEDS ACTION | `requirements.txt` must be created with pinned versions |
| Technical Constraints: Patient privacy | PASS | No patient identifiers stored; files processed transiently; privacy disclaimer displayed |
| Development Workflow: Feature branch | PASS | Working on `001-brain-tumor-prediction-ui` branch |

**Gate result**: PASS (one action item: create `requirements.txt` with pinned versions during implementation)

## Project Structure

### Documentation (this feature)

```text
specs/001-brain-tumor-prediction-ui/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── api-contract.md  # REST API specification
└── tasks.md             # Phase 2 output (/speckit-tasks command)
```

### Source Code (repository root)

```text
backend/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration (paths, thresholds, device)
│   ├── api/
│   │   ├── routes.py        # POST /predict endpoint
│   │   └── schemas.py       # Pydantic request/response models
│   ├── services/
│   │   ├── inference.py     # Model loading + ensemble inference (wraps existing code)
│   │   ├── preprocessing.py # NIfTI validation + preprocessing (wraps existing stages)
│   │   └── calibration.py   # Platt calibration wrapper
│   └── utils/
│       └── file_handling.py # Temp file management, NIfTI validation
├── tests/
│   ├── test_api.py          # API endpoint tests
│   ├── test_inference.py    # Model inference contract tests
│   ├── test_preprocessing.py # Preprocessing unit tests
│   └── conftest.py          # Fixtures (synthetic NIfTI volumes)
├── requirements.txt         # Pinned Python dependencies
└── README.md

frontend/
├── src/
│   ├── App.tsx              # Root component with routing
│   ├── main.tsx             # Entry point
│   ├── theme/
│   │   └── neurograde.ts    # Design tokens from index.html (colors, fonts, radii)
│   ├── components/
│   │   ├── Layout/          # Nav, Footer, PageContainer
│   │   ├── Upload/          # ModalityUploadZone (×4 slots), UploadProgress
│   │   ├── Prediction/      # ResultCard, ProbabilityGauge, ConfidenceBadge
│   │   ├── OperatingMode/   # ModeSelector (balanced / high-sensitivity)
│   │   ├── ModelBreakdown/  # EnsembleFormula, ModelContributionBar
│   │   ├── History/         # SessionHistorySidebar, HistoryItem
│   │   └── common/          # Button, Card, GlassPanel, Spinner, ErrorBanner
│   ├── pages/
│   │   ├── PredictPage.tsx  # Main prediction workflow page
│   │   └── LandingBridge.tsx # Optional bridge to index.html
│   ├── services/
│   │   └── api.ts           # Axios client for POST /predict
│   ├── hooks/
│   │   ├── usePrediction.ts # Upload + predict mutation hook
│   │   └── useSessionHistory.ts # Session-only prediction history
│   └── types/
│       └── prediction.ts    # TypeScript interfaces for API responses
├── public/
│   └── favicon.ico
├── index.html               # Vite entry HTML
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

**Structure Decision**: Web application with separate `backend/` and `frontend/` directories at repository root. The backend wraps existing model code (via `sys.path` to project root for `models/`, `utils/` imports). The frontend is a standalone React SPA that communicates with the backend via REST API. The existing `index.html` landing page remains untouched at project root.

## Complexity Tracking

No constitution violations requiring justification. The project structure uses 2 directories (frontend + backend) which is the standard web application pattern.
