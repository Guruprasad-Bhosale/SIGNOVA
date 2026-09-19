# Phase 3 Starting State Audit

**Date:** 2026-09-17  
**Project:** SIGNOVA — Continuous Indian Sign Language (ISL) → English Translation System  
**Phase:** Phase 3 — Isolated / Dynamic Sign Recognition Baseline  

---

## 1. System & Environment Inventory

- **Operating System**: Windows 11 (PowerShell)
- **Python**: 3.12.10 (AMD64)
- **Deep Learning Framework**: PyTorch 2.14.0
- **Hardware Profile**: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM) / 15.7 GB Host RAM
- **Computer Vision**: OpenCV 4.11+, MediaPipe 0.10.35 (verified 543-topology)
- **Active Tests**: 28/28 unit and integration tests passing (`python -m pytest`)

---

## 2. Training-Ready Feature Assets & Manifests

- **Extracted Landmark Features (`data/features/landmarks/`)**:
  - 20 `.npz` feature containers extracted in Phase 2 pilot (`1782bea75c7d-1.npz` through `1782bea75c7d-29.npz`).
  - Shape per sample: $(T \times 543 \times 3)$ normalized coordinates + $(T \times 4)$ binary detection masks + $(T,)$ timestamps.
  - 100% numerical integrity verified: zero NaN/Inf corruptions, coordinate envelope $[-5.0, 5.0]$, monotonic timestamps.
- **Feature Manifest (`data/manifests/feature_manifest.csv`)**: 20 verified entries with SemVer `0.1.0` metadata headers.
- **Feature Failures Log (`data/manifests/feature_failures.csv`)**: 15 unmounted remote archive requests logged with clear diagnostics.

---

## 3. Dataset & Label Availability Audit

| Dataset | Type | Sample Count | Local Video Availability | Label Availability | Isolated Sign Recognition Suitability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ISLTranslate** | Continuous Sentence Translation | 31,222 sentence pairs | 0 raw local videos (57.89 GB unmounted on HF) | Full English sentence translations only; **NO word-level glosses** | **UNSUITABLE** for isolated sign classification. Fabricating isolated signs from continuous sentences is scientifically invalid and prohibited. |
| **INCLUDE** | Isolated Dynamic Sign Recognition | 263 classes (4,287 videos on Zenodo `4010759`) | **0 local video files present on disk**; unmounted remote archive | Isolated word-level sign class labels (e.g., Greetings, Colours, Actions) | **SUITABLE** for isolated dynamic sign recognition baseline once acquired/extracted. |

---

## 4. Missing Prerequisites for Full-Scale Isolated Training

1. **INCLUDE Raw Video Acquisition**: No raw INCLUDE video files are currently downloaded or extracted locally in `data/raw/include/`.
2. **Controlled Pilot Acquisition Strategy**: Rather than downloading the entire multi-gigabyte INCLUDE repository, Phase 3 will acquire a strictly bounded, authentic subset (e.g. Greetings & Colours subset: ~10–20 classes, ~100–200 MB) or generate a controlled pilot isolated benchmark using the verified Phase 2 extraction pipeline.
3. **PyTorch Sequence Collation**: Need `SignLanguageFeatureDataset` and `PadCollate` with explicit temporal padding masks `(B, T_max)` and sequence lengths.
4. **Centralized Landmark Groups**: Need `src/signova/features/feature_groups.py` for structured semantic slicing (`HANDS`, `HANDS_POSE`, `FULL_543`).

---

## 5. Decision & Execution Path

- **Primary Dataset for Phase 3**: INCLUDE (isolated dynamic sign classification).
- **Secondary Reference**: ISLTranslate remains the canonical continuous translation dataset reserved for Phase 5+.
- **Zero Fabrication Rule**: We will NOT synthesize gloss labels from ISLTranslate sentences.
