# NeuroGrade Prediction UI

React + TypeScript + Vite frontend for brain tumor HGG/LGG classification. Visual identity matches the project landing page (`index.html` at repo root).

## Prerequisites

- Node.js 18+
- Backend API running at `http://localhost:8000` (see [`backend/README.md`](../backend/README.md))

## Setup

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

### Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_URL` | `http://localhost:8000` | FastAPI base URL |

Create `frontend/.env.local` to override:

```
VITE_API_URL=http://localhost:8000
```

## Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Vite dev server with HMR |
| `npm run build` | Production build → `dist/` |
| `npm run preview` | Preview production build |
| `npm run lint` | ESLint |

## Design system

Tokens live in `src/theme/neurograde.ts` and `src/theme/global.css`:

| Token | Value | Usage |
|-------|-------|--------|
| Background | `#060810` | Page shell |
| Accent cyan | `#00d4aa` | LGG, primary actions |
| Accent warm | `#f0724b` | HGG, errors |
| Accent violet | `#8b5cf6` | MIL / secondary highlights |
| Display font | Playfair Display | Headings |
| Body font | Manrope | UI copy |
| Mono font | JetBrains Mono | Metrics, formulas |

Responsive breakpoints: **1100px** (sidebar stack), **768px** (single-column upload, compact nav) — see `src/theme/responsive.css`.

## Features

- 4-slot NIfTI upload with client-side validation (`.nii` / `.nii.gz`, 200 MB per file, 500 MB total)
- Ensemble prediction with operating modes (τ = 0.41 / 0.38)
- Expandable model breakdown and meta-learner formula
- Session-only prediction history (sidebar)
- Backend health banner with 10s polling until models are loaded

## Structure

```
frontend/src/
├── components/
│   ├── Upload/          # ModalityUploadZone
│   ├── Prediction/      # ResultCard, ProbabilityGauge
│   ├── ModelBreakdown/  # Ensemble transparency panel
│   ├── History/         # SessionHistorySidebar
│   └── common/          # Button, Reveal, skeletons, banners
├── hooks/               # usePrediction, useSessionHistory, useBackendHealth
├── pages/PredictPage.tsx
├── services/api.ts
└── theme/
```

Specification and API contract: [`specs/001-brain-tumor-prediction-ui/`](../specs/001-brain-tumor-prediction-ui/)
