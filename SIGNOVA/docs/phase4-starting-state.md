# Phase 4 Starting State Audit: SIGNOVA Continuous Sign Sequence Modeling

## Executive Summary
Phase 4 transitions SIGNOVA from isolated/dynamic sign classification toward genuine continuous Indian Sign Language (ISL) video temporal modeling, representation, and segmentation. This audit establishes the baseline capabilities inherited from Phases 0–3, identifies reusable modules, defines extensions needed for continuous modeling, and formalizes the research track based on empirical label availability.

---

## 1. What Phase 3 Already Provides

1. **Centralized Anatomical Landmark Groups**:
   - Standardized slicing for 543 MediaPipe Holistic landmarks (`HANDS` = 42 joints, `HANDS_POSE` = 75 joints, `FULL` = 543 joints, `POSE` = 33 joints, `FACE` = 468 joints) in [`feature_groups.py`](file:///g:/SingLang/SIGNOVA/src/signova/features/feature_groups.py).
2. **Lazy Feature Storage & Loading**:
   - Compressed `.npz` storage format retaining 3D joint coordinates, detection masks (4 channels), timestamps, and frame indices in [`storage.py`](file:///g:/SingLang/SIGNOVA/src/signova/features/storage.py).
   - Lazy on-demand loading in [`feature_dataset.py`](file:///g:/SingLang/SIGNOVA/src/signova/data/feature_dataset.py).
3. **Variable-Length Sequence Collation**:
   - Dynamic batch padding with validity padding masks $(B, T)$ in [`collate.py`](file:///g:/SingLang/SIGNOVA/src/signova/data/collate.py).
4. **Temporal Backbone Architectures**:
   - 2-stage spatial projection + 2-layer BiGRU with packed sequences and masked temporal pooling in [`baseline_rnn.py`](file:///g:/SingLang/SIGNOVA/src/signova/models/baseline_rnn.py).
   - Dilated Residual Temporal Convolutional Network (TCN) with exponential receptive fields in [`baseline_tcn.py`](file:///g:/SingLang/SIGNOVA/src/signova/models/baseline_tcn.py).
5. **Hardware-Optimized Training Engine**:
   - `SignLanguageTrainer` supporting CUDA/CPU, AMP FP16, gradient accumulation, gradient clipping ($1.0$), early stopping, learning rate scheduling, and RTX 3050 VRAM safety in [`trainer.py`](file:///g:/SingLang/SIGNOVA/src/signova/recognition/trainer.py).
6. **Isolated Benchmark Empirical Verification**:
   - 100% test accuracy on the 10-class isolated dynamic sign benchmark using Hands+Pose BiGRU (93.3% on Hands-only and Full-body).
   - 35 unit/integration tests passing.
   - Reference repositories (`ISLTranslate-main`, `isl-translator-main`) SHA-256 verified and intact.

---

## 2. What Can Be Reused vs What Must Be Extended

| Component | Status in Phase 3 | Phase 4 Action | Required Modifications |
|---|---|---|---|
| **Landmark Feature Groups** | Complete (`feature_groups.py`) | **Reuse directly** | None. Standardized slicing applies to all continuous frames. |
| **Storage & npz Format** | Complete (`storage.py`) | **Reuse directly** | Stores $(T, 543, 3)$ continuous arrays seamlessly. |
| **Temporal Collation** | Isolated (`PadCollate`) | **Extend** | Must support continuous metadata, sliding windows, and optional sequence targets. |
| **Temporal Encoders** | Sequence-to-One (`BaselineRNN`, `BaselineTCN`) | **Extend to Sequence-to-Sequence** | Preserve per-frame temporal representation $(B, T, D)$ instead of global temporal pooling. |
| **Boundary Modeling** | None | **Implement New** | Heuristic candidate boundary proposal generator & modular boundary head. |
| **CTC Infrastructure** | Placeholder stub (`ctc.py`) | **Implement New** | Full CTC recognizer model, greedy decoder, and data integrity validator. |
| **Dataset Loader** | Fixed single-class target | **Implement New** | Continuous feature dataset supporting variable-length continuous videos and windowing. |
| **Inference Pipeline** | Isolated sign predictor | **Extend** | Continuous video predictor, windowed offline inference, and diagnostic timeline outputs. |

---

## 3. Current Data Limitations & Label Truth

1. **ISLTranslate Dataset**:
   - Contains ~31,222 continuous video-sentence pairs.
   - Ground-truth annotation is sentence-level English text translations.
   - **Does NOT contain token-level or frame-level ISL gloss annotations.**
   - **Does NOT contain temporal sign boundary annotations.**
2. **Signer Metadata**:
   - UID prefixes must not be assumed to represent signers without explicit validation.
3. **INCLUDE Dataset**:
   - Strictly isolated sign vocabulary (263 classes, single-word clips). Not a continuous sentence dataset.

---

## 4. Phase 4 Research Track Selection

**Selected Track: TRACK B — NO VALID SEQUENTIAL SIGN LABELS AVAILABLE**

- Under strict research guardrails, English sentences will **never** be parsed into artificial sign tokens or treated as visual sign supervision.
- CTC training on real-world data is **BLOCKED** and will be documented as an explicit prerequisite for Phase 5.
- CTC architecture will be verified on deterministic synthetic fixtures.
- Phase 4 focuses on:
  1. Continuous temporal representations $(B, T, D)$ using BiGRU and TCN.
  2. Non-leaking temporal windowing.
  3. Heuristic candidate boundary proposals for visualization.
  4. Real ISLTranslate continuous video empirical extraction & representation profiling.
  5. Phase 3 pretrained encoder transfer analysis.
  6. Windowed offline streaming inference interfaces.
