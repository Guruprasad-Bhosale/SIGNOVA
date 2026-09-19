# SIGNOVA Phase 1 Starting State Audit

Audit of the pre-existing Phase 0 foundation and boundaries prior to executing Phase 1.

---

## 1. What Already Exists (Phase 0 Deliverables)

- **Modular Project Layout**: `apps/` (`api/`, `web/`), `src/signova/`, `configs/`, `data/`, `notebooks/`, `scripts/`, `tests/`, `docs/`, `outputs/`.
- **Configuration Subsystem**: YAML configurations (`base.yaml`, `dataset.yaml`, `model.yaml`, `training.yaml`, `inference.yaml`, `paths.yaml`) with Pydantic validation schemas.
- **Hardware Profile**: Windows 11 workstation, Python 3.12.10, Node.js 24.20.0, NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM).
- **Core Abstractions**:
  - `DatasetAdapter` base class (`src/signova/data/base.py`).
  - `ISLTranslateAdapter` & `INCLUDEAdapter` initial implementations.
  - MediaPipe Holistic 543-keypoint extractor interface and spatial normalization routines.
  - Model and inference coordinator placeholders.
- **Test Suite**: 15 passing tests across unit and integration suites.
- **Initial Manifest**: 31,222 samples from `ISLTranslate.csv` with deterministic 80/10/10 split (Train: 25,004, Val: 3,116, Test: 3,102).

---

## 2. What Phase 1 Can Reuse
- Deterministic hashing algorithms (`deterministic_split` via SHA-256).
- Configuration manager (`ConfigManager`) and path resolution logic.
- Landmark data structure classes (`HolisticLandmarkFrame`).
- FastAPI `/health` and `/version` endpoints and React frontend.

---

## 3. What Phase 1 Needs to Add
- **Remote Dataset Inventory & Discovery**: Physical verification of `ISLTranslate-main` (confirming 0 local video files) and remote Hugging Face dataset mapping.
- **Remote Smoke Test**: 3–5 representative sample downloads to verify decoding, FPS, resolution, and annotation alignment.
- **Deep Text Analytics**: Vocabulary size, n-gram/word frequencies, sentence lengths, duplicate analysis.
- **Gloss Availability Documentation**: Formal documentation confirming `ISLTranslate.csv` provides English translations only without explicit sign glosses.
- **Signer Identity Audit & Proposed Signer-Independent Split**: Verifying whether UID prefixes denote signers; calculating overlap and saving `proposed_signer_independent_split.csv` as an analysis artifact.
- **Canonical Sample Contract**: Refined `SignSample` schema distinguishing `video_reference` from `local_video_path` across `REMOTE_ONLY`, `LOCAL_AVAILABLE`, and `BOTH`.
- **Validation & Summary CLI Tools**: `scripts/validate_dataset.py` and `scripts/dataset_summary.py`.

---

## 4. Invariant Rules (Must NOT Change)
- `isl-translator-main` and `ISLTranslate-main` are strictly **read-only**.
- No machine learning model training in Phase 1.
- No bulk downloading of the complete 31k remote video dataset.
- No bulk MediaPipe landmark extraction.
