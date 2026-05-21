<!--
  === Sync Impact Report ===
  Version change: 0.0.0 → 1.0.0 (initial ratification)
  Modified principles: N/A (initial creation)
  Added sections:
    - Principle I: Code Quality & Reproducibility
    - Principle II: Testing Standards
    - Principle III: User Experience Consistency
    - Principle IV: Performance Requirements
    - Section: Technical Constraints
    - Section: Development Workflow
    - Section: Governance
  Removed sections: N/A
  Templates requiring updates:
    - .specify/templates/plan-template.md ✅ aligned (Constitution Check section present)
    - .specify/templates/spec-template.md ✅ aligned (Success Criteria supports performance/UX)
    - .specify/templates/tasks-template.md ✅ aligned (Phase N includes performance, testing, docs)
  Follow-up TODOs: None
-->

# Brain Tumor MRI Classification Constitution

## Core Principles

### I. Code Quality & Reproducibility

Every script, module, and pipeline in this project MUST produce deterministic,
reproducible results given identical inputs and random seeds. This is
non-negotiable for a medical imaging research project where clinical trust
depends on verifiable outputs.

- All random operations MUST use explicit, configurable seeds (default: 42).
  Seeds MUST be logged alongside experiment results.
- Preprocessing stages (1-4) MUST be idempotent: re-running a stage on the
  same input MUST produce byte-identical output.
- Runtime stages (5-6) MUST use seeded transforms so that a given
  (epoch, sample) pair produces the same augmentation when the seed is fixed.
- Every function that performs non-trivial computation MUST include a docstring
  specifying inputs, outputs, and any assumptions about data shape or dtype.
- Magic numbers are prohibited. All hyperparameters, thresholds, and
  configuration values MUST be externalized to YAML/JSON config files under
  `configs/`.
- Dead code, commented-out experiments, and unused imports MUST be removed
  before merging to the main branch.
- Patient-level data isolation MUST be enforced: no patient's data may appear
  in both training and validation/test splits within the same fold.

### II. Testing Standards

All code that transforms data, trains models, or produces predictions MUST
have corresponding validation mechanisms. The testing strategy is tiered
to match the cost and criticality of each component.

- **Unit tests** MUST exist for all utility functions (`utils/`), data
  loaders, and config parsers. Unit tests MUST run in under 60 seconds
  total without GPU access.
- **Integration tests** MUST verify end-to-end pipeline correctness: raw
  NIfTI input through each preprocessing stage to a valid model-ready
  tensor. Integration tests MAY use a small synthetic volume (e.g.,
  64x64x64 zeros with known class label).
- **Contract tests** MUST validate model input/output shapes, dtype
  expectations, and checkpoint load/save round-trips for every model
  architecture (ResNet50-3D, Swin UNETR, MIL).
- **Regression tests** MUST be added whenever a bug is fixed to prevent
  recurrence.
- Test data MUST NOT include real patient data. Use synthetic NIfTI
  volumes or officially provided sample data only.
- All tests MUST pass before any merge to the main branch. Failing tests
  block the merge without exception.

### III. User Experience Consistency

Every user-facing interface -- CLI scripts, inference pipelines, and
output reports -- MUST behave predictably and provide clear, actionable
feedback. Researchers and clinicians interact with this system; ambiguity
in outputs can lead to misinterpretation.

- All CLI scripts MUST use `argparse` with descriptive `--help` text,
  sensible defaults, and consistent flag naming conventions (lowercase,
  hyphen-separated, e.g., `--top-k`, `--threshold`).
- Default behavior MUST be safe and non-destructive. Scripts MUST NOT
  overwrite existing outputs without an explicit `--overwrite` flag.
- Every inference script MUST report: model name, threshold used,
  calibration mode, number of patients processed, and a summary table
  of predictions. Output format MUST be consistent across all models.
- Error messages MUST include the failing file path, the expected
  condition, and a suggested remediation step.
- Progress reporting MUST be provided for any operation exceeding 10
  seconds (use `tqdm` or structured log lines with completion percentage).
- All output files (CSV, JSON, logs) MUST include a metadata header or
  field recording the script version, timestamp, configuration used, and
  random seed.

### IV. Performance Requirements

Model training and inference MUST meet defined latency and resource
budgets. Medical imaging workloads are compute-intensive; uncontrolled
resource usage wastes GPU hours and delays research iteration.

- **Inference latency**: Single-patient inference (4 modalities, full
  pipeline from NIfTI to prediction) MUST complete in under 30 seconds
  on a single GPU (NVIDIA T4 or equivalent baseline).
- **Memory budget**: Training batch processing MUST NOT exceed 80% of
  available GPU memory at the configured batch size. Out-of-memory
  errors during training indicate a configuration defect, not a hardware
  limitation.
- **Data loading**: DataLoader throughput MUST keep GPU utilization above
  70% during training. If GPU utilization drops below this threshold,
  `num_workers`, prefetch factor, or caching strategy MUST be tuned.
- **Preprocessing throughput**: Stages 1-4 MUST process the full dataset
  (285 patients, 4 modalities each) in under 4 hours on a single CPU
  node. Per-patient processing time MUST be logged.
- **Model checkpointing**: Checkpoints MUST be saved at configurable
  intervals (default: every epoch). Checkpoint files MUST include model
  state, optimizer state, epoch number, best metric value, and config
  hash.
- **Disk usage**: Intermediate preprocessing outputs MUST NOT exceed 2x
  the raw dataset size per stage. Scripts MUST log cumulative disk usage
  after each stage completes.

## Technical Constraints

The following constraints govern technology choices and operational
boundaries for this project.

- **Language**: Python 3.10+ is the sole implementation language.
- **Deep learning framework**: PyTorch is the primary framework. MONAI
  is the standard library for medical imaging transforms and model
  components.
- **Data format**: All volumetric data MUST be stored as NIfTI
  (.nii/.nii.gz). No proprietary formats are permitted for data exchange.
- **Configuration**: All experiment configurations MUST be defined in
  YAML files under `configs/`. Command-line arguments MAY override
  config values but MUST NOT be the sole source of configuration.
- **Logging**: Python `logging` module MUST be used (not `print`
  statements) for all operational output. Log level MUST be configurable
  via `--log-level` flag.
- **Version pinning**: All Python dependencies MUST be pinned to exact
  versions in `requirements.txt`. Unpinned dependencies are not
  permitted.
- **Patient privacy**: No real patient identifiers (names, dates of
  birth, hospital IDs) may appear in code, logs, or committed data
  files. BraTS anonymized IDs are the only permitted identifiers.

## Development Workflow

All changes to this project follow a structured workflow designed to
maintain quality and traceability.

- Every feature or fix MUST be developed on a dedicated branch. Direct
  commits to the main branch are prohibited.
- Commit messages MUST follow the format: `type(scope): description`
  where type is one of `feat`, `fix`, `refactor`, `docs`, `test`,
  `perf`, `chore`.
- Code review is REQUIRED for all changes that modify preprocessing
  logic, model architecture, loss functions, evaluation metrics, or
  threshold selection.
- Experiment results (metrics, plots, confusion matrices) MUST be
  recorded in the `experiments/` directory with a timestamped
  subdirectory or entry linked to the specific commit hash.
- Configuration changes that affect model behavior MUST include a
  before/after comparison of at least one evaluation metric on the
  validation set.

## Governance

This constitution is the authoritative reference for all development
decisions in the Brain Tumor MRI Classification project. When a conflict
arises between this constitution and any other document, script, or
convention, this constitution prevails.

- **Amendments**: Any change to this constitution MUST be documented
  with a rationale, reviewed by at least one project contributor, and
  recorded in the Sync Impact Report at the top of this file.
- **Versioning**: The constitution follows semantic versioning
  (MAJOR.MINOR.PATCH). MAJOR increments indicate backward-incompatible
  governance changes; MINOR increments indicate new principles or
  material expansions; PATCH increments indicate clarifications or
  wording fixes.
- **Compliance review**: All pull requests MUST be checked against the
  active constitution principles. Reviewers MUST verify that new code
  does not violate any principle. Violations MUST be resolved before
  merge.
- **Exception process**: If a principle cannot be satisfied for a
  specific change, the contributor MUST document the exception in the
  pull request description with: (a) which principle is violated,
  (b) why the violation is necessary, and (c) a plan to resolve the
  violation in a follow-up.

**Version**: 1.0.0 | **Ratified**: 2026-05-21 | **Last Amended**: 2026-05-21
