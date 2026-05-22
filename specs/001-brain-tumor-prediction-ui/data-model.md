# Data Model: Brain Tumor Prediction Web Interface

**Branch**: `001-brain-tumor-prediction-ui` | **Date**: 2026-05-21

## Entities

### MRIScan (Frontend — transient, in-memory)

Represents a set of 4 uploaded NIfTI files for one patient.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | string (UUID) | Auto-generated | Unique identifier for this upload session |
| files | Map<Modality, File> | Exactly 4 entries | Uploaded files keyed by modality |
| totalSizeBytes | number | ≤ 500 MB | Combined size of all 4 files |
| uploadTimestamp | ISO 8601 string | Auto-set | When files were uploaded |
| status | UploadStatus | enum | Current state of the upload |

**Modality enum**: `T1` | `T1ce` | `T2` | `FLAIR`

**UploadStatus enum**: `incomplete` (< 4 files) → `ready` (4 files present) → `uploading` → `processing` → `complete` | `error`

**Validation rules**:
- All 4 modalities must be present before submission
- Each file must have `.nii` or `.nii.gz` extension
- Individual file size ≤ 200 MB; total ≤ 500 MB
- Files are not persisted — held in browser `File` objects until submission

---

### PredictionResult (Backend → Frontend, session-stored)

The output of the ensemble inference pipeline. Returned by the API and stored in frontend session state.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | string (UUID) | Matches MRIScan.id | Links result to its upload |
| patientLabel | string | User-provided or auto | Display label for the scan |
| classification | "HGG" \| "LGG" | Derived from threshold | Final grade at selected threshold |
| calibratedProbability | float | [0.0, 1.0] | Platt-calibrated P(HGG) |
| uncalibratedProbability | float | [0.0, 1.0] | Raw ensemble P(HGG) before Platt |
| modelProbabilities | ModelProbabilities | Required | Per-model HGG probabilities |
| ensembleLogit | float | unbounded | Raw logit before sigmoid |
| appliedThreshold | float | 0.38 or 0.41 | Threshold used for classification |
| operatingMode | OperatingMode | enum | Which mode was active |
| processingDurationMs | number | ≥ 0 | Backend processing time in ms |
| timestamp | ISO 8601 string | Auto-set | When prediction was completed |

**ModelProbabilities**:

| Field | Type | Description |
|-------|------|-------------|
| resnet | float [0,1] | ResNet50-3D P(HGG) |
| swinunetr | float [0,1] | SwinUNETR-3D P(HGG) |
| mil | float [0,1] | DualStreamMIL-3D P(HGG) |

**Derived fields** (computed on frontend from stored data):
- `confidenceLevel`: "High" (p > 0.85 or p < 0.15), "Medium" (0.65-0.85 or 0.15-0.35), "Low" (0.35-0.65)
- `ensembleFormula`: String rendering of `σ(4.06·{swinunetr} + 0.89·{mil} + 0.54·{resnet} − 2.40) = {calibratedProbability}`

---

### OperatingMode (Frontend — static configuration)

Defines a clinical operating threshold. These are constants, not user-created.

| Field | Type | Value (Balanced) | Value (High-Sensitivity) |
|-------|------|-------------------|--------------------------|
| id | string | "balanced" | "high_sensitivity" |
| name | string | "Balanced Screening" | "High-Sensitivity Triage" |
| threshold | float | 0.41 | 0.38 |
| description | string | "Maximizes F1 score with equal precision and recall (0.9365)" | "Minimizes missed HGG cases with recall 0.9524" |
| precision | float | 0.9365 | 0.9091 |
| recall | float | 0.9365 | 0.9524 |
| f1 | float | 0.9365 | 0.9302 |
| expectedFN | number | 4 | 3 |
| expectedFP | number | 4 | 6 |

---

### SessionHistory (Frontend — in-memory array)

| Field | Type | Description |
|-------|------|-------------|
| predictions | PredictionResult[] | Ordered by timestamp descending |

**Lifecycle**: Created when app loads (empty array). Entries added after each successful prediction. Cleared on page refresh / browser close. Maximum 50 entries (oldest evicted).

## Relationships

```text
MRIScan (1) ──uploads──> (1) PredictionResult
PredictionResult (N) ──stored in──> (1) SessionHistory
PredictionResult (1) ──classified by──> (1) OperatingMode
```

## State Transitions

### Upload Flow

```text
[No Files] → (user adds file) → [Incomplete: 1-3 files]
[Incomplete] → (user adds remaining files) → [Ready: 4 files]
[Ready] → (user clicks Predict) → [Uploading]
[Uploading] → (upload complete) → [Processing]
[Processing] → (inference complete) → [Complete] → PredictionResult created
[Processing] → (error occurs) → [Error] → error message displayed
[Error] → (user retries or uploads new) → [No Files]
[Complete] → (user uploads new scan) → [No Files]
```

### Operating Mode Switch (post-prediction)

```text
[Result @ Balanced τ=0.41] → (user switches mode) → [Result @ HighSensitivity τ=0.38]
```

This is a client-side re-evaluation: same `calibratedProbability`, different `appliedThreshold` → potentially different `classification`. No API call needed.
