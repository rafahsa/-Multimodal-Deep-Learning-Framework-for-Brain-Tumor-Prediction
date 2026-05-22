# Tasks: Brain Tumor Prediction Web Interface

**Input**: Design documents from `specs/001-brain-tumor-prediction-ui/`

**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/api-contract.md, quickstart.md

**Tests**: Not explicitly requested in feature specification. Test tasks are omitted.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Web app**: `backend/` (Python FastAPI), `frontend/` (React + Vite + TypeScript)
- Backend imports existing model code from project root `models/`, `utils/` via sys.path

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initialize both backend and frontend project scaffolds

- [x] T001 Create backend directory structure: `backend/app/`, `backend/app/api/`, `backend/app/services/`, `backend/app/utils/`, `backend/tests/`
- [x] T002 Create `backend/requirements.txt` with pinned dependencies: fastapi, uvicorn[standard], python-multipart, torch, monai, SimpleITK, numpy, scikit-learn, joblib, pandas, pydantic
- [x] T003 [P] Initialize React+TypeScript project with Vite in `frontend/` using `npm create vite@latest frontend -- --template react-ts`
- [x] T004 [P] Install frontend dependencies: axios in `frontend/package.json`
- [x] T005 [P] Create NeuroGrade design tokens file at `frontend/src/theme/neurograde.ts` extracting CSS variables from `index.html` (colors: --bg-primary #060810, --accent-cyan #00d4aa, --accent-warm #f0724b, --accent-violet #8b5cf6; fonts: Playfair Display, Manrope, JetBrains Mono; radii, borders, glass effects)
- [x] T006 [P] Create global styles in `frontend/src/theme/global.css` with base reset, body styles, grain overlay, scrollbar, and Google Fonts imports matching `index.html`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core backend infrastructure and shared frontend components that MUST be complete before ANY user story can be implemented

**WARNING**: No user story work can begin until this phase is complete

- [x] T007 Create FastAPI application entry point in `backend/app/main.py` with CORS middleware (origins from env var CORS_ORIGINS, default localhost:5173), lifespan handler for model loading at startup, and 500 MB max upload size
- [x] T008 Create backend configuration module in `backend/app/config.py` with Settings class: DEVICE (auto/cuda/cpu), MODEL_BASE_DIR (../results), METRICS_PATH (../ensemble/results/meta_learner_metrics.json), CALIBRATOR_PATH, CORS_ORIGINS, MAX_UPLOAD_MB (500), LOG_LEVEL (INFO)
- [x] T009 Create Pydantic request/response schemas in `backend/app/api/schemas.py` defining PredictionResponse (prediction_id, patient_label, calibrated_probability, uncalibrated_probability, model_probabilities, ensemble_logit, meta_learner_coefficients, thresholds with both balanced and high_sensitivity, processing_duration_ms, timestamp, device_used) and ErrorResponse models per contracts/api-contract.md
- [x] T010 Create model loading service in `backend/app/services/inference.py` with ModelRegistry class that loads ResNet50-3D, SwinUNETR-3D, DualStreamMIL-3D at startup from checkpoint paths, loads meta_learner_metrics.json coefficients, loads Platt calibrator from joblib — importing `create_resnet50_3d` from `models/resnet50_3d_fast/model.py`, `create_swin_unetr_classifier` from `models/swin_unetr_encoder.py`, `create_dual_stream_mil` from `models/dual_stream_mil.py` via sys.path to project root
- [x] T011 Create preprocessing service in `backend/app/services/preprocessing.py` wrapping `_preprocess_volume()` and `_select_slices_entropy()` from `scripts/test/run_final_ensemble_inference.py` — accepts 4 NIfTI file paths, validates each with SimpleITK, resizes to 128³, z-score normalizes, returns (4,128,128,128) tensor for 3D models and (16,4,224,224) bag for MIL
- [x] T012 Create calibration service in `backend/app/services/calibration.py` wrapping `_baseline_ensemble()` and `_apply_platt()` from the existing inference script — accepts 3 model probabilities, returns uncalibrated and calibrated ensemble P(HGG), plus classifications at both thresholds (0.41 and 0.38)
- [x] T013 [P] Create file handling utility in `backend/app/utils/file_handling.py` with functions: validate_nifti_extension(), save_upload_to_tempdir(), cleanup_tempdir() — validates .nii/.nii.gz extension, writes UploadFile to temp directory, returns paths dict keyed by modality
- [x] T014 [P] Create shared React layout components: `frontend/src/components/Layout/Navbar.tsx` (NeuroGrade brand with Playfair Display italic accent, nav links, glass-morphism backdrop-filter), `frontend/src/components/Layout/Footer.tsx` (matching index.html footer style), `frontend/src/components/Layout/PageContainer.tsx` (max-width wrapper with bg-primary background)
- [x] T015 [P] Create common UI components in `frontend/src/components/common/`: `Button.tsx` (btn-primary and btn-outline variants matching index.html), `Card.tsx` (bg-elevated with border and radius-lg), `GlassPanel.tsx` (bg-glass with backdrop-filter blur), `Spinner.tsx` (cyan animated spinner), `ErrorBanner.tsx` (error display with warm accent and suggestion text)
- [x] T016 Create Axios API client in `frontend/src/services/api.ts` with baseURL from env var VITE_API_URL (default http://localhost:8000), POST /api/predict multipart form-data function accepting 4 File objects keyed by modality, GET /api/health function
- [x] T017 Create TypeScript type definitions in `frontend/src/types/prediction.ts` matching PredictionResponse from backend schemas: PredictionResult, ModelProbabilities, ThresholdResult, OperatingMode interfaces
- [x] T018 Create `frontend/src/App.tsx` with root layout wrapping Navbar + PageContainer + Footer, and main PredictPage route

**Checkpoint**: Foundation ready — backend can start, frontend renders shell with NeuroGrade styling. User story implementation can now begin.

---

## Phase 3: User Story 1 — Upload Brain MRI & Receive Prediction (Priority: P1) — MVP

**Goal**: User uploads 4 NIfTI modality files and receives an HGG/LGG classification with calibrated probability

**Independent Test**: Upload 4 NIfTI files from `test/DATA_FOR_TEST/UCSF-PDGM-0004/`, receive prediction result card showing tumor grade + probability

### Implementation for User Story 1

- [x] T019 [US1] Create the prediction API route in `backend/app/api/routes.py`: POST /api/predict endpoint accepting 4 UploadFile fields (t1, t1ce, t2, flair) + optional patient_label string, calling file_handling → preprocessing → inference → calibration services, returning PredictionResponse JSON; include timing measurement for processing_duration_ms
- [x] T020 [US1] Create GET /api/health endpoint in `backend/app/api/routes.py` returning models_loaded status, device name, GPU name (if cuda), and version
- [x] T021 [US1] Register API routes in `backend/app/main.py` with /api prefix, wire lifespan to load ModelRegistry at startup and expose via app.state
- [x] T022 [US1] Create the 4-slot modality upload component in `frontend/src/components/Upload/ModalityUploadZone.tsx`: 4 labeled drop zones (T1, T1ce, T2, FLAIR) each accepting .nii/.nii.gz via drag-and-drop or click-to-browse, showing file name + size when populated, validation indicator (green check when file added, grey placeholder when empty), "all 4 required" status bar
- [x] T023 [P] [US1] Create upload progress indicator in `frontend/src/components/Upload/UploadProgress.tsx`: shows upload state (uploading → preprocessing → running models → calibrating → done), animated progress bar with cyan accent, elapsed time counter
- [x] T024 [US1] Create prediction result card in `frontend/src/components/Prediction/ResultCard.tsx`: displays tumor grade (HGG in warm accent or LGG in cyan accent) as large prominent text, calibrated probability as a percentage gauge, confidence level badge (High/Medium/Low derived from probability distance from 0.5), applied threshold and mode name, processing duration
- [x] T025 [P] [US1] Create probability gauge component in `frontend/src/components/Prediction/ProbabilityGauge.tsx`: circular or semicircular gauge showing 0-100% with gradient from cyan (LGG) to warm (HGG), pointer at calibrated probability value, numeric display in JetBrains Mono
- [x] T026 [US1] Create the usePrediction hook in `frontend/src/hooks/usePrediction.ts`: manages upload mutation state (idle → uploading → processing → success → error), calls api.ts POST /api/predict with FormData containing 4 files, returns { predict, result, isLoading, error, reset }
- [x] T027 [US1] Build the main PredictPage in `frontend/src/pages/PredictPage.tsx`: composes ModalityUploadZone + OperatingMode selector (Phase 4) + Predict button + UploadProgress + ResultCard; Predict button enabled only when all 4 files are uploaded; shows UploadProgress during processing, ResultCard on success, ErrorBanner on failure
- [x] T028 [US1] Add error handling for backend failures: in routes.py return 400 for invalid NIfTI (caught by SimpleITK), 413 for oversized files, 503 if models not loaded, 500 for inference errors — each with error/message/suggestion fields per api-contract.md; in frontend show ErrorBanner with suggestion text
- [x] T029 [US1] Add privacy disclaimer component in `frontend/src/components/common/PrivacyDisclaimer.tsx`: subtle banner at bottom of upload area stating "Research use only — no authentication enforced. You are responsible for the privacy of uploaded medical data." Render in PredictPage above the upload zone.

**Checkpoint**: MVP complete — user can upload 4 NIfTI files, receive HGG/LGG prediction with probability. Default operating mode is Balanced (τ=0.41). Validate with test data from `test/DATA_FOR_TEST/`.

---

## Phase 4: User Story 2 — Select Clinical Operating Mode (Priority: P2)

**Goal**: User switches between Balanced Screening (τ=0.41) and High-Sensitivity Triage (τ=0.38) and sees classification update

**Independent Test**: After receiving a prediction, switch modes and verify the classification may change while the probability stays the same

### Implementation for User Story 2

- [x] T030 [US2] Create operating mode selector component in `frontend/src/components/OperatingMode/ModeSelector.tsx`: two cards side by side — "Balanced Screening" (τ=0.41, equal precision/recall 0.9365, FN=4/FP=4) and "High-Sensitivity Triage" (τ=0.38, recall 0.9524, FN=3/FP=6) — active card highlighted with cyan border and glow, inactive card with standard border; includes brief clinical implication text per card
- [x] T031 [US2] Add operating mode state to PredictPage in `frontend/src/pages/PredictPage.tsx`: store selected mode in React state (default: balanced), pass to ResultCard, derive classification client-side by comparing calibratedProbability >= threshold — no API re-call needed on mode switch
- [x] T032 [US2] Update ResultCard in `frontend/src/components/Prediction/ResultCard.tsx` to accept operatingMode prop and re-derive classification (HGG/LGG) from calibrated_probability vs selected threshold; show both threshold results from API response with active one highlighted; animate classification change on mode switch

**Checkpoint**: User can toggle operating modes and see instant classification updates. Both modes display their statistical characteristics.

---

## Phase 5: User Story 3 — View Model Breakdown & Ensemble Explanation (Priority: P3)

**Goal**: User expands a detail panel to see per-model probabilities, coefficients, and the ensemble formula with actual values

**Independent Test**: After prediction, click "View Model Details" and verify individual model probabilities and weighted formula display

### Implementation for User Story 3

- [x] T033 [US3] Create model contribution bar component in `frontend/src/components/ModelBreakdown/ModelContributionBar.tsx`: horizontal stacked bar showing weighted contribution of each model (ResNet warm accent, SwinUNETR cyan accent, MIL violet accent), bar width proportional to |coefficient × probability|, numeric labels showing coefficient and probability per model
- [x] T034 [P] [US3] Create ensemble formula display component in `frontend/src/components/ModelBreakdown/EnsembleFormula.tsx`: renders P(HGG) = σ(4.06·p_swin + 0.89·p_mil + 0.54·p_resnet − 2.40) with actual values from current prediction filled in, styled in JetBrains Mono with color-coded coefficients (matching model accent colors), shows intermediate logit value and final sigmoid output
- [x] T035 [US3] Create expandable model detail panel in `frontend/src/components/ModelBreakdown/ModelDetailPanel.tsx`: collapsible section below ResultCard toggled by "View Model Details" button; contains: 3 model cards (one per architecture) each showing model name, type badge (CNN/Transformer/MIL), individual P(HGG), meta-learner coefficient, and weighted contribution; plus the EnsembleFormula and ModelContributionBar
- [x] T036 [US3] Integrate ModelDetailPanel into PredictPage in `frontend/src/pages/PredictPage.tsx`: render below ResultCard when prediction result exists, pass model_probabilities and meta_learner_coefficients from API response; panel collapsed by default, expands on click with smooth animation

**Checkpoint**: Full model transparency — user sees how each model contributed to the ensemble decision.

---

## Phase 6: Session History (Cross-cutting enhancement from clarification)

**Goal**: Previous predictions displayed in session sidebar for comparison during browser session

**Independent Test**: Upload multiple scans sequentially, verify prediction history sidebar shows all results ordered by time

### Implementation for Session History

- [x] T037 Create useSessionHistory hook in `frontend/src/hooks/useSessionHistory.ts`: stores PredictionResult[] in React state, addPrediction() appends to array (max 50, evict oldest), getPredictions() returns descending by timestamp, clearHistory() resets array; state is lost on page refresh per spec
- [x] T038 Create session history sidebar in `frontend/src/components/History/SessionHistorySidebar.tsx`: vertical panel on the right side of PredictPage showing list of previous predictions — each HistoryItem shows patient label, classification badge (HGG/LGG), calibrated probability, timestamp; clicking an item scrolls to and highlights its full result
- [x] T039 [P] Create history item component in `frontend/src/components/History/HistoryItem.tsx`: compact card with patient label, color-coded classification badge (HGG warm / LGG cyan), probability percentage, relative timestamp ("2 min ago"), active state highlighting when selected
- [x] T040 Integrate SessionHistorySidebar into PredictPage in `frontend/src/pages/PredictPage.tsx`: add sidebar to page layout (collapsible on mobile), wire usePrediction result to useSessionHistory.addPrediction on success, allow clicking history items to display that prediction's ResultCard and ModelDetailPanel

**Checkpoint**: Users can compare multiple predictions within the same session.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Visual refinement, responsiveness, error edge cases, documentation

- [x] T041 [P] Add responsive styles to all frontend components for mobile/tablet breakpoints (768px, 1100px) matching index.html responsive patterns — stack upload slots vertically, collapse sidebar, adjust font sizes
- [x] T042 [P] Add loading skeleton states to PredictPage: shimmer placeholders for ResultCard and ModelDetailPanel areas while awaiting first prediction
- [x] T043 [P] Add scroll-reveal animations to frontend components matching index.html `.reveal` pattern — fade-in-up on intersection observer with staggered delays
- [x] T044 Add frontend file validation before upload in `frontend/src/components/Upload/ModalityUploadZone.tsx`: reject files without .nii/.nii.gz extension, warn on individual files >200 MB, block submission if total >500 MB — show inline error messages per slot
- [x] T045 [P] Create `backend/README.md` with setup instructions, environment variables, and API documentation link
- [x] T046 [P] Create `frontend/README.md` with setup instructions, available scripts, and design system reference
- [x] T047 Handle backend unavailability gracefully: frontend calls GET /api/health on mount, shows "Backend connecting..." banner if unhealthy or unreachable, polls every 10 seconds until healthy, then hides banner
- [x] T048 Run quickstart.md validation: start backend + frontend per quickstart.md steps, upload test data from `test/DATA_FOR_TEST/UCSF-PDGM-0004/`, verify full prediction flow works end-to-end

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup (Phase 1) completion — BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational (Phase 2) — MVP
- **User Story 2 (Phase 4)**: Depends on Phase 3 (needs ResultCard and PredictPage to exist)
- **User Story 3 (Phase 5)**: Depends on Phase 3 (needs prediction result data to display)
- **Session History (Phase 6)**: Depends on Phase 3 (needs prediction results to store)
- **Polish (Phase 7)**: Depends on Phases 3-6 being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational — independent MVP
- **User Story 2 (P2)**: Requires US1 ResultCard + PredictPage — adds mode switching overlay
- **User Story 3 (P3)**: Requires US1 prediction result data — adds expandable detail panel
- **Session History**: Requires US1 prediction flow — adds sidebar tracking

### Within Each User Story

- Backend services before API routes
- API routes before frontend API client usage
- Common components before page-level composition
- Core implementation before error handling refinements

### Parallel Opportunities

**Phase 1** (all parallel):
- T003, T004, T005, T006 can all run simultaneously

**Phase 2** (parallel groups):
- T007, T008, T009 (backend core) can run in parallel with T014, T015, T016, T017, T018 (frontend core)
- T010, T011, T012 must be sequential (inference depends on preprocessing, calibration depends on inference)
- T013 can run in parallel with T010-T012

**Phase 3** (parallel groups):
- T022, T023, T024, T025 (frontend components) can run in parallel
- T019 must wait for T010-T013 (backend services)
- T026, T027 depend on both frontend components and API client

**Phase 5** (parallel):
- T033, T034 can run in parallel

---

## Parallel Example: User Story 1

```text
# Backend (sequential dependency chain):
T019 POST /api/predict route (depends on T010, T011, T012, T013)
T020 GET /api/health route (can parallel with T019)
T021 Wire routes into main.py (depends on T019, T020)

# Frontend components (all parallel — different files):
T022 ModalityUploadZone.tsx
T023 UploadProgress.tsx
T024 ResultCard.tsx
T025 ProbabilityGauge.tsx
T029 PrivacyDisclaimer.tsx

# Frontend integration (depends on components + API):
T026 usePrediction.ts hook (depends on T016 api client)
T027 PredictPage.tsx (depends on T022-T026)
T028 Error handling (depends on T019, T027)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (~30 min)
2. Complete Phase 2: Foundational (~3-4 hours)
3. Complete Phase 3: User Story 1 (~3-4 hours)
4. **STOP and VALIDATE**: Upload test data, verify prediction works
5. Deploy/demo if ready — this is a functional product

### Incremental Delivery

1. Setup + Foundational → Backend starts, frontend shell renders
2. Add User Story 1 → Upload + Predict works (MVP!)
3. Add User Story 2 → Operating mode switching (~1 hour)
4. Add User Story 3 → Model breakdown transparency (~2 hours)
5. Add Session History → Multi-scan comparison (~1.5 hours)
6. Polish → Responsive, animations, edge cases (~2 hours)

### Total Estimated Effort

- Phase 1 (Setup): 6 tasks
- Phase 2 (Foundational): 12 tasks
- Phase 3 (US1 — MVP): 11 tasks
- Phase 4 (US2): 3 tasks
- Phase 5 (US3): 4 tasks
- Phase 6 (History): 4 tasks
- Phase 7 (Polish): 8 tasks
- **Total: 48 tasks**

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks in the same phase
- [US1/US2/US3] labels map tasks to specific user stories for traceability
- Backend imports existing model code from project root — no need to copy/rewrite model architectures
- Existing `run_final_ensemble_inference.py` is the reference implementation — services extract its functions
- The `index.html` landing page at project root remains untouched — the React app is a separate interface
- Model checkpoints must exist at `results/` paths or `archive_minimal_runs/` before backend can start
