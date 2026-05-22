# Research: Brain Tumor Prediction Web Interface

**Branch**: `001-brain-tumor-prediction-ui` | **Date**: 2026-05-21

## R1: Backend Framework — FastAPI vs Flask

**Decision**: FastAPI

**Rationale**: FastAPI provides native async support (critical for handling file uploads without blocking), automatic OpenAPI documentation, built-in Pydantic validation for request/response schemas, and `python-multipart` integration for file uploads. It aligns with the project's need for a typed, documented API contract. Performance is significantly better than Flask for I/O-bound operations (file upload receiving).

**Alternatives considered**:
- Flask: Simpler but lacks native async, requires manual schema validation, no auto-generated docs. Would require Flask-CORS, Flask-RESTful, marshmallow for equivalent functionality.
- Django REST Framework: Too heavy for a single-endpoint API. Brings ORM, admin, migrations — none needed here.

## R2: Model Loading Strategy — Startup vs On-Demand

**Decision**: Load all three models at application startup, keep in GPU/CPU memory

**Rationale**: The inference pipeline requires all three models (ResNet50-3D, SwinUNETR-3D, DualStreamMIL-3D) for every prediction. Loading weights from disk takes 10-30 seconds per model. Pre-loading at startup eliminates this latency from the user-facing request path, meeting the <2 minute end-to-end target. Memory cost is acceptable for a single-user research tool (~2-4 GB GPU memory total).

**Alternatives considered**:
- On-demand loading: Would add 30-90 seconds to each prediction. Unacceptable for UX.
- Lazy loading (first request): Better than on-demand but creates a poor first-user experience. Startup loading is predictable.

## R3: File Upload Transport — Multipart Form vs Chunked

**Decision**: Standard multipart/form-data with 4 file fields

**Rationale**: NIfTI files for a single patient are typically 30-150 MB total (4 modalities). Standard multipart upload handles this well within the 500 MB limit. FastAPI's `UploadFile` with `python-multipart` streams files to temp storage efficiently. No need for chunked upload complexity for a single-user tool.

**Alternatives considered**:
- Chunked upload (tus protocol): Adds resume-on-failure capability but significant frontend/backend complexity. Overkill for single-user, same-network usage.
- Base64 encoding in JSON: Bloats payload by 33%, no streaming. Rejected.

## R4: Frontend Build Tool — Vite vs CRA vs Next.js

**Decision**: Vite with React 18 + TypeScript

**Rationale**: Vite provides fast HMR during development, optimized production builds, and native TypeScript support. It's the modern standard for React SPAs. The app is a single-page tool with no SSR needs, so Next.js complexity is unnecessary. Create React App is deprecated/unmaintained.

**Alternatives considered**:
- Next.js: SSR/SSG capabilities are unused here. Adds routing complexity and server-side concerns that conflict with the SPA model.
- Create React App: Deprecated, slow builds, no longer recommended by React team.

## R5: Design System Approach — CSS-in-JS vs CSS Modules vs Tailwind

**Decision**: CSS Modules with design tokens extracted from `index.html` into a TypeScript theme file

**Rationale**: The NeuroGrade design system in `index.html` uses CSS custom properties (variables) extensively. Extracting these into a TypeScript theme constant (`neurograde.ts`) and using CSS Modules preserves the exact design language while providing type-safe access in components. CSS Modules avoid global style collisions and keep component styles co-located. No additional runtime dependency needed.

**Alternatives considered**:
- Tailwind CSS: Would require translating the entire NeuroGrade design system into Tailwind config. The existing design uses precise custom values (specific rgba, gradients, backdrop-filter) that don't map cleanly to Tailwind utilities.
- styled-components: Adds runtime overhead and a dependency. CSS Modules achieve the same scoping without runtime cost.
- Global CSS replicating index.html: Fragile, no scoping, harder to maintain.

## R6: Backend-Frontend Communication — REST vs WebSocket

**Decision**: REST API with a single `POST /api/predict` endpoint

**Rationale**: The prediction workflow is request-response: upload files → wait → get result. No real-time streaming or bi-directional communication needed. REST is simpler, stateless, and matches the single-user model. The frontend shows a loading state during the request.

**Alternatives considered**:
- WebSocket with progress updates: Would enable real-time preprocessing progress (stage 1/4, 2/4...). Added complexity not justified for v1. Can be retrofitted later by adding a `/ws/predict` endpoint alongside the REST endpoint.
- Server-Sent Events (SSE): Same progress benefit as WebSocket but one-directional. Same complexity concern for v1.

## R7: Wrapping Existing Inference Code

**Decision**: Import and refactor `run_final_ensemble_inference.py` functions into `backend/app/services/inference.py`

**Rationale**: The existing script (`scripts/test/run_final_ensemble_inference.py`) contains all necessary functions: `_load_patient`, `_preprocess_volume`, `_select_slices_entropy`, `_load_resnet`, `_load_swin`, `_load_mil`, `_predict_*`, `_baseline_ensemble`, `_apply_platt`. These can be extracted into a service class that loads models once and exposes a `predict(nifti_paths)` method. The service imports from `models/` and `utils/` via `sys.path` to project root.

**Key adaptation points**:
- `_load_patient` currently expects a patient directory with naming convention `{patient_id}_{modality}.nii.gz`. The web upload will receive 4 arbitrary files mapped to modalities by upload slot. The service will accept 4 file paths explicitly rather than scanning a directory.
- Models are loaded per-call in the script. The service will load once at startup and reuse.
- The script writes to CSV. The service returns a Python dict/Pydantic model.

**Alternatives considered**:
- Subprocess call to the existing script: Fragile, hard to capture structured output, model reloading on each call.
- Complete rewrite: Unnecessary — the existing code is well-structured and tested.

## R8: Temporary File Handling

**Decision**: Use Python `tempfile.TemporaryDirectory` for uploaded files, clean up after inference

**Rationale**: Uploaded NIfTI files need to be written to disk briefly for SimpleITK to read them (SimpleITK requires file paths, not byte streams). A temporary directory per request ensures isolation and automatic cleanup. The directory is deleted after inference completes (or on error).

**Alternatives considered**:
- In-memory processing with `io.BytesIO`: SimpleITK's `ReadImage` requires a file path. Would need to write to temp file anyway.
- Persistent upload directory: Unnecessary for a stateless, no-history-on-server design. Would require cleanup cron job.

## R9: CORS Configuration

**Decision**: Allow CORS from `localhost:5173` (Vite dev server) in development, configurable origin list for production

**Rationale**: During development, the React dev server (Vite, port 5173) and FastAPI server (port 8000) run on different ports, requiring CORS. FastAPI's `CORSMiddleware` handles this. Production deployments may reverse-proxy both behind the same origin, making CORS unnecessary — but the configuration should remain flexible.
