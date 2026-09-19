# Implementation Plan — SIGNOVA Phase 21: First Genuine Sequential ISL CTC Training, Evaluation & Live-Model Integration

Phase 21 advances SIGNOVA from diagnostic mode (`STATE_B`, `REAL_CTC = BLOCKED`, `LIVE_CAMERA = OPERATIONAL_DIAGNOSTIC_MODE`) toward `STATE_A_DATA_LIMITED` or `STATE_A` by using ONLY genuine human-annotated sequential Indian Sign Language data. If genuine human data is absent or insufficient, Phase 21 remains scientifically blocked in `STATE_B`, reporting the exact missing conditions without manufacturing synthetic data or fake checkpoints.

## User Review Required

> [!IMPORTANT]
> **Zero Fabrication Policy**: Real CTC training, evaluation, and checkpoint generation are strictly gated on the canonical Phase 19 readiness engine. If genuine annotations are not present in the workspace, Phase 21 will report `STATE_B` with `REAL_CTC_TRAINING = BLOCKED`. All unit tests will test both the current `STATE_B` blocked path and simulated `STATE_A`/`STATE_A_DATA_LIMITED` authorized paths in isolated temporary environments.

> [!NOTE]
> **Protected Reference Files**: The 44 reference files across `ISLTranslate-main` and `isl-translator-main` will remain untouched and verified against their SHA-256 baselines.

---

## Proposed Changes

### 1. Operations & Orchestration Engine (`src/signova/operations/`)

#### [NEW] [phase21_orchestrator.py](file:///g:/SingLang/SIGNOVA/src/signova/operations/phase21_orchestrator.py)
- Reuses canonical `evaluate_phase19_readiness()`.
- Implements `Phase21Orchestrator`:
  - Gate evaluation: `check_training_authorization()`.
  - Dataset construction from genuine human annotations (`VideoAnnotation`), verifying provenance, SHA-256, schema versions.
  - Frozen dataset fingerprint calculation (`dataset_manifest.json`, `dataset_manifest.csv`, `dataset_sha256`).
  - Vocabulary builder with `<BLANK>=0, <UNK>=1`.
  - Splitting hierarchy: `SIGNER_INDEPENDENT` $\to$ `SESSION_INDEPENDENT` $\to$ `SOURCE_GROUP_INDEPENDENT` $\to$ `RANDOM (WITH EXPLICIT WARNING)`.
  - Sequence CTC feasibility checker ($T_{\text{required}} \le T_{\text{features}}$).
  - CTC model trainer & experiment runner using BiGRU encoder + CTC loss.
  - Checkpoint provenance validator (`checkpoint_sha256`, dataset fingerprint, vocabulary fingerprint, input spec matching Phase 20).
  - Test evaluation and error categorization (insertions, deletions, substitutions, repeated tokens, short/long sequences).
  - Live model integration with `LiveModelRegistry`.
- Implements `evaluate_phase21_readiness()`.

---

### 2. Live Runtime & Model Registry Integration (`src/signova/live/`)

#### [MODIFY] [model_registry.py](file:///g:/SingLang/SIGNOVA/src/signova/live/model_registry.py)
- Ensure model discovery dynamically checks `models/experiments/phase21_real_ctc` and `models/checkpoints/` for authorized real models.
- Support loading `ModelMetadata` with complete provenance and CTC model weights.

---

### 3. Python-First Entrypoint Scripts (`scripts/`)

#### [NEW] [check_phase21.py](file:///g:/SingLang/SIGNOVA/scripts/check_phase21.py)
- Safe, non-training CLI reporting supervision state, human data state, dataset state, vocabulary, split, CTC feasibility, model availability, Phase 20 live integration, and next action.

#### [NEW] [prepare_phase21_dataset.py](file:///g:/SingLang/SIGNOVA/scripts/prepare_phase21_dataset.py)
- Checks Phase 19 gate. If authorized, generates frozen dataset manifests and fingerprints. If blocked, exits safely reporting blocker.

#### [NEW] [train_phase21_ctc.py](file:///g:/SingLang/SIGNOVA/scripts/train_phase21_ctc.py)
- Refuses to train unless authorized by canonical gate. If authorized, trains BiGRU CTC model and generates immutable checkpoint and provenance metadata.

#### [NEW] [evaluate_phase21_ctc.py](file:///g:/SingLang/SIGNOVA/scripts/evaluate_phase21_ctc.py)
- Evaluates real model on held-out test split, reporting TER, exact match, token F1, and classified error analysis.

#### [NEW] [verify_phase21_checkpoint.py](file:///g:/SingLang/SIGNOVA/scripts/verify_phase21_checkpoint.py)
- Validates checkpoint hash, dataset fingerprint, vocabulary fingerprint, and Phase 20 input spec compatibility.

#### [NEW] [integrate_phase21_live_model.py](file:///g:/SingLang/SIGNOVA/scripts/integrate_phase21_live_model.py)
- Links authorized checkpoint to Phase 20 live runtime and validates feature compatibility.

#### [NEW] [run_phase21_verification.py](file:///g:/SingLang/SIGNOVA/scripts/run_phase21_verification.py)
- Master end-to-end verification script executing diagnostic check, gate check, test suite execution, protected reference check (44/44), and formatted banner output.

---

### 4. Interactive Console & Launcher (`run_signova.py`)

#### [MODIFY] [run_signova.py](file:///g:/SingLang/SIGNOVA/run_signova.py)
- Add menu entries and CLI options for Phase 21 checks and verification workflows.

---

### 5. Unit & Integration Test Suite (`tests/unit/`)

#### [NEW] [test_phase21_readiness_summary.py](file:///g:/SingLang/SIGNOVA/tests/unit/test_phase21_readiness_summary.py)
#### [NEW] [test_phase21_supervision_gate.py](file:///g:/SingLang/SIGNOVA/tests/unit/test_phase21_supervision_gate.py)
#### [NEW] [test_phase21_dataset.py](file:///g:/SingLang/SIGNOVA/tests/unit/test_phase21_dataset.py)
#### [NEW] [test_phase21_vocabulary.py](file:///g:/SingLang/SIGNOVA/tests/unit/test_phase21_vocabulary.py)
#### [NEW] [test_phase21_split.py](file:///g:/SingLang/SIGNOVA/tests/unit/test_phase21_split.py)
#### [NEW] [test_phase21_ctc_feasibility.py](file:///g:/SingLang/SIGNOVA/tests/unit/test_phase21_ctc_feasibility.py)
#### [NEW] [test_phase21_training.py](file:///g:/SingLang/SIGNOVA/tests/unit/test_phase21_training.py)
#### [NEW] [test_phase21_checkpoint_provenance.py](file:///g:/SingLang/SIGNOVA/tests/unit/test_phase21_checkpoint_provenance.py)
#### [NEW] [test_phase21_evaluation.py](file:///g:/SingLang/SIGNOVA/tests/unit/test_phase21_evaluation.py)
#### [NEW] [test_phase21_error_analysis.py](file:///g:/SingLang/SIGNOVA/tests/unit/test_phase21_error_analysis.py)
#### [NEW] [test_phase21_live_integration.py](file:///g:/SingLang/SIGNOVA/tests/unit/test_phase21_live_integration.py)
#### [NEW] [test_phase21_entrypoints.py](file:///g:/SingLang/SIGNOVA/tests/unit/test_phase21_entrypoints.py)

---

### 6. Documentation (`docs/`)

#### [NEW] [phase21-real-ctc.md](file:///g:/SingLang/SIGNOVA/docs/phase21-real-ctc.md)
- Complete technical design, dataset preparation, split hierarchy, CTC feasibility equation, BiGRU CTC architecture, provenance validation, live runtime integration, and scientific limitation guidelines.

#### [NEW] [phase21-completion-report.md](file:///g:/SingLang/SIGNOVA/docs/phase21-completion-report.md)
- Full verification and execution report detailing Phase 21 status, gate authorization, test results, protected baseline check, and next actions.

---

## Verification Plan

### Automated Tests
- Run complete pytest suite: `pytest`
- Run Phase 21 specific test suite: `pytest tests/unit/test_phase21_*.py -v`
- Verify 44/44 reference integrity files: `python scripts/run_phase21_verification.py`

### Script Verification Commands
```powershell
python scripts/check_phase21.py
python scripts/prepare_phase21_dataset.py
python scripts/train_phase21_ctc.py
python scripts/evaluate_phase21_ctc.py
python scripts/verify_phase21_checkpoint.py
python scripts/integrate_phase21_live_model.py
python scripts/run_phase21_verification.py
```
