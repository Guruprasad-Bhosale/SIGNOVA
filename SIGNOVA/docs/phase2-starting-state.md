# SIGNOVA Phase 2 Starting State Audit

Audit of the pre-existing Phase 1 deliverables and baseline state prior to executing Phase 2.

---

## 1. Reusable Phase 1 Components
- **Canonical Sample Contract**: `SignSample` and `SampleAvailability` (`REMOTE_ONLY`, `LOCAL_AVAILABLE`, `BOTH`) in `src/signova/data/models.py`.
- **Dataset Manifests**: `data/manifests/isltranslate_manifest.csv` (31,222 samples), `isltranslate_integrity.csv`, `duplicate_report.csv`, and `proposed_signer_independent_split.csv`.
- **Dataset Adapter**: `ISLTranslateAdapter` with deterministic splitting, sample retrieval, and metadata parsing.
- **Computer Vision Dependencies**: OpenCV (`5.0.0.93`) and MediaPipe (`1.0.1`) installed and active.
- **Configuration Subsystem**: `configs/dataset.yaml`, `configs/paths.yaml`, `configs/base.yaml`.

---

## 2. Required Phase 2 Deliverables
1. **Remote Video Resolver (`src/signova/data/remote.py`)**: Abstract layer resolving sample IDs to remote archive endpoints or cached files.
2. **Local Video Cache Subsystem (`src/signova/data/cache.py`)**: Transient storage in `data/cache/videos/` with size limits and eviction.
3. **Robust Video Decoder (`src/signova/preprocessing/video.py`)**: Safe frame streaming, corruption detection, and metadata extraction.
4. **Temporal Frame Sampler (`src/signova/preprocessing/sampling.py`)**: Uniform temporal sampling and target FPS decimation.
5. **MediaPipe Holistic Extractor (`src/signova/features/mediapipe_extractor.py`)**: 543-keypoint extraction (Pose 33, Face 468, Left Hand 21, Right Hand 21) with explicit component detection masks.
6. **Landmark Normalization & Masking (`src/signova/preprocessing/normalization.py`)**: Mid-hip centering and shoulder distance scaling.
7. **Feature Storage & Manifests**: Compact binary `.npz` storage in `data/features/`, `data/manifests/feature_manifest.csv`, and `data/manifests/feature_failures.csv`.
8. **Pilot Extraction & Sanity Checks**: 20–50 sample pilot extraction, visual verification, and storage benchmarks.

---

## 3. Components to Remain Untouched
- `ISLTranslate-main` and `isl-translator-main` remain **100% read-only**.
- No neural network model training (ST-GCN, Transformer, CTC, or NMT).
- No bulk download of the 57.89 GB remote video archive.
- Existing train/val/test split (80/10/10) and proposed session-independent benchmark are strictly preserved.
