# Feature Specification: Brain Tumor Prediction Web Interface

**Feature Branch**: `001-brain-tumor-prediction-ui`

**Created**: 2026-05-21

**Status**: Draft

**Input**: User description: "I want to build an interface for this project; I want an interface where I can upload an image and get a result. Use index.html for design and connect with it, I need to use this model (MICCAI 2026 paper results). Use React for UI and Python for backend."

## Clarifications

### Session 2026-05-21

- Q: MRI upload format — single file or multiple files per patient? → A: Upload 4 separate NIfTI files (T1, T1ce, T2, FLAIR), matching BraTS native format
- Q: Authentication & access control model? → A: No authentication for v1 — open access research/demo tool with privacy disclaimer; users responsible for data handling
- Q: Prediction history — persist or ephemeral? → A: Session-only history — predictions listed in sidebar during browser session, lost on refresh; no server-side persistence

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Upload Brain MRI & Receive Tumor Grade Prediction (Priority: P1)

A clinician or researcher navigates to the NeuroGrade web application, uploads a brain MRI scan as 4 separate NIfTI files (one per modality: T1, T1ce, T2, FLAIR), and receives a tumor grade classification result (HGG vs. LGG) along with a calibrated probability score and confidence indicators. The interface design matches the existing NeuroGrade visual identity from `index.html` — dark theme, cyan/violet/warm accent palette, Playfair Display + Manrope typography, glass-morphism cards, and orbital animations.

**Why this priority**: This is the core value proposition — without the ability to upload and classify, the application has no purpose. Everything else builds on this flow.

**Independent Test**: Can be fully tested by uploading a sample NIfTI file and verifying a classification result (HGG/LGG) appears with a probability score. Delivers immediate diagnostic value.

**Acceptance Scenarios**:

1. **Given** the user is on the NeuroGrade prediction page, **When** they drag-and-drop or click-to-select 4 valid NIfTI (.nii or .nii.gz) files — one for each modality (T1, T1ce, T2, FLAIR), **Then** the file is accepted, a processing indicator appears, and within a reasonable timeframe a prediction result card displays: tumor grade (HGG or LGG), calibrated probability score, confidence level indicator, and the selected operating threshold mode.
2. **Given** the user uploads a valid MRI scan, **When** the backend ensemble model completes inference, **Then** the result displays the ensemble decision (combining ResNet50-3D, SwinUNETR-3D, and DualStreamMIL-3D predictions), the calibrated probability via Platt scaling, and the final classification at the selected threshold.
3. **Given** the user uploads an invalid file (wrong format, corrupted, non-brain image), **When** the system attempts to process it, **Then** a clear, user-friendly error message appears explaining the issue and suggesting corrective action, without crashing or leaving the UI in a broken state.

---

### User Story 2 — Select Clinical Operating Mode Before Prediction (Priority: P2)

A clinician selects between two operating modes before or after uploading a scan: "Balanced Screening" (threshold τ=0.41, maximizing F1 with equal precision/recall of 0.9365) or "High-Sensitivity Triage" (threshold τ=0.38, maximizing recall at 0.9524 to minimize missed HGG cases). The selected mode affects how the raw ensemble probability is converted into a final classification.

**Why this priority**: Operating mode selection is a key differentiator of this framework — it enables clinicians to adapt the system to their risk tolerance. Without it, the interface delivers a generic prediction rather than a clinically controllable one.

**Independent Test**: Can be tested by switching between modes and verifying that the displayed threshold, description, and (for the same scan) potentially different classification outcomes reflect the selected mode.

**Acceptance Scenarios**:

1. **Given** the prediction page is loaded, **When** the user views the operating mode selector, **Then** two clearly labeled options are visible — "Balanced Screening (τ=0.41)" and "High-Sensitivity Triage (τ=0.38)" — with brief descriptions of each mode's clinical implications.
2. **Given** a prediction result is displayed with Balanced mode selected, **When** the user switches to High-Sensitivity mode, **Then** the classification is re-evaluated against the new threshold and the result card updates accordingly (same probability, potentially different grade label).

---

### User Story 3 — View Detailed Model Breakdown & Ensemble Explanation (Priority: P3)

After receiving a prediction, the user can expand a details panel to see how each individual model (ResNet50-3D, SwinUNETR-3D, DualStreamMIL-3D) contributed to the ensemble decision, including individual model probabilities, the meta-learner coefficients, and the ensemble formula visualization.

**Why this priority**: Transparency builds clinical trust. Understanding which model drove the decision and how the ensemble combined them helps clinicians interpret and validate results. This is important but not essential for basic functionality.

**Independent Test**: Can be tested by uploading a scan, receiving a result, then expanding the detail panel to verify individual model predictions and coefficient-weighted contributions are displayed.

**Acceptance Scenarios**:

1. **Given** a prediction result is displayed, **When** the user clicks "View Model Details" or expands the detail section, **Then** a breakdown card appears showing each base model's individual probability, its meta-learner coefficient (SwinUNETR: 4.06, MIL: 0.89, ResNet: 0.54), and the weighted contribution to the final ensemble logit.
2. **Given** the detail panel is expanded, **When** the user reviews the ensemble formula, **Then** the formula P(HGG) = σ(4.06·p_swin + 0.89·p_mil + 0.54·p_resnet − 2.40) is displayed with the actual values from the current prediction filled in.

---

### Edge Cases

- What happens when the user uploads a file that is too large (exceeding server memory for 3D volume processing)?
- How does the system handle partial uploads (user provides fewer than 4 modality files)?
- What happens if the backend Python inference service is unavailable or times out?
- How does the system respond when the uploaded MRI has unexpected dimensions or voxel spacing?
- What happens when the user attempts to upload multiple files simultaneously?
- How does the system behave on slow network connections during file upload?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a file upload mechanism accepting 4 separate NIfTI files (.nii, .nii.gz) per patient — one for each modality (T1, T1ce, T2, FLAIR) — with clear labeling of which slot corresponds to which modality
- **FR-002**: System MUST process uploaded MRI scans through the full preprocessing pipeline (N4 bias correction, Z-score normalization, ROI cropping, isotropic resize to 128³)
- **FR-003**: System MUST run inference using the three-model ensemble (ResNet50-3D, SwinUNETR-3D, DualStreamMIL-3D) and combine predictions via the trained logistic regression meta-learner
- **FR-004**: System MUST apply Platt calibration to ensemble predictions before presenting results
- **FR-005**: System MUST display the final classification (HGG or LGG), calibrated probability score, and confidence level for each prediction
- **FR-006**: System MUST support two operating modes — Balanced Screening (τ=0.41) and High-Sensitivity Triage (τ=0.38) — allowing users to switch between them
- **FR-007**: System MUST show individual model contributions (per-model probability and meta-learner coefficient) in an expandable detail view
- **FR-008**: System MUST display the ensemble decision formula with actual prediction values filled in
- **FR-009**: System MUST provide clear error messages for invalid file uploads (wrong format, corrupted files, missing modalities)
- **FR-010**: System MUST show a processing/loading state during MRI preprocessing and model inference
- **FR-011**: The React frontend MUST adopt the existing NeuroGrade visual identity from `index.html` — dark theme (#060810 background), cyan (#00d4aa) / warm (#f0724b) / violet (#8b5cf6) accents, Playfair Display + Manrope typography, glass-morphism cards, and subtle grain overlay
- **FR-012**: System MUST provide a Python backend API that receives uploaded MRI files, runs the preprocessing + inference pipeline, and returns structured prediction results
- **FR-013**: System MUST validate uploaded files on both frontend (file extension, basic size check) and backend (NIfTI structure validation) before processing
- **FR-014**: System MUST display a privacy/research-use disclaimer informing users that no authentication is enforced and that they are responsible for the privacy and handling of any uploaded medical data
- **FR-015**: System MUST maintain a session-only prediction history — displaying a list of previous predictions during the current browser session, allowing users to review and compare results across multiple scans; history is cleared on page refresh or browser close

### Key Entities

- **MRI Scan**: A set of 4 uploaded NIfTI files representing one patient's brain MRI — one file per modality (T1, T1ce, T2, FLAIR). Key attributes: file names (×4), total file size, volume dimensions per modality, upload timestamp. All 4 modalities are required for prediction.
- **Prediction Result**: The output of the ensemble inference pipeline for a given scan. Key attributes: final classification (HGG/LGG), calibrated probability, individual model probabilities (ResNet, SwinUNETR, MIL), applied threshold, operating mode, ensemble logit value, processing duration. Stored in browser session memory only — not persisted server-side.
- **Operating Mode**: A clinical deployment configuration defining the decision threshold. Key attributes: mode name, threshold value (τ), expected precision/recall/F1 characteristics, clinical use case description.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can upload a brain MRI scan and receive a tumor grade prediction within 2 minutes of file upload completion (including preprocessing and inference time)
- **SC-002**: The prediction result clearly communicates the tumor grade, probability, and confidence such that 90% of first-time users understand the output without external guidance
- **SC-003**: Users can switch between Balanced and High-Sensitivity operating modes and see the result update within 2 seconds (threshold re-application only, no re-inference needed)
- **SC-004**: The interface matches the NeuroGrade visual identity — an observer familiar with the existing landing page (`index.html`) recognizes the prediction interface as belonging to the same product
- **SC-005**: Invalid file uploads produce a helpful error message within 5 seconds, guiding the user to correct the issue
- **SC-006**: The system handles files up to 500 MB without crashing or producing unrecoverable errors

## Assumptions

- Users have access to brain MRI data in NIfTI format (.nii or .nii.gz) — the system does not support DICOM or other imaging formats in this version
- The trained model weights (ResNet50-3D, SwinUNETR-3D, DualStreamMIL-3D, meta-learner, Platt calibrator) are available and loadable on the server running the Python backend
- The Python backend runs on a machine with sufficient GPU memory to perform 3D model inference (or falls back to CPU with longer processing times)
- The existing `index.html` serves as a design reference and landing page; the React application is a separate but visually connected interface (same product family)
- No authentication or access control is required for v1 — the application is a research/demo tool with open access
- Single-user inference is sufficient for initial deployment — concurrent multi-user inference queuing is out of scope for v1
- The frontend communicates with the Python backend via REST API over HTTP
- Browser support targets modern evergreen browsers (Chrome, Firefox, Edge, Safari latest 2 versions)
