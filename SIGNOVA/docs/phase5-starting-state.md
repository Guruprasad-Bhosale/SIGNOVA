# Phase 5 Starting State Audit: Sequential ISL Supervision & Continuous Recognition

## Executive Summary
Phase 5 advances SIGNOVA from continuous temporal modeling (Phase 4) toward genuine continuous Indian Sign Language (ISL) sequence recognition. This audit formalizes the starting architecture inherited from Phases 0–4, assesses available dataset annotations, and defines the research track based on empirical supervision availability.

---

## 1. What Phase 4 Delivered & What Is Reusable

1. **Continuous Landmark Dataset & Dynamic Collation**:
   - `ContinuousSignDataset` and `ContinuousPadCollate` in [`continuous_dataset.py`](file:///g:/SingLang/SIGNOVA/src/signova/data/continuous_dataset.py) supporting variable-length continuous sequence batches with boolean padding masks.
2. **Temporal Windowing with Zero Leakage**:
   - `SlidingWindowExtractor` and `verify_window_split_leakage` in [`windowing.py`](file:///g:/SingLang/SIGNOVA/src/signova/data/windowing.py) guaranteeing video-level split isolation.
3. **Continuous Temporal Backbones**:
   - `ContinuousBiGRUEncoder` ($2.1\text{M}$ params) and `ContinuousTCNEncoder` ($2.4\text{M}$ params) preserving per-frame $(B, T, D)$ representations without global sequence collapsing in [`continuous_encoder.py`](file:///g:/SingLang/SIGNOVA/src/signova/models/continuous_encoder.py).
4. **CTC Recognizer Architecture & Greedy Decoder**:
   - `CTCContinuousRecognizer` in [`ctc_recognizer.py`](file:///g:/SingLang/SIGNOVA/src/signova/models/ctc_recognizer.py) and `CTCDecoder` in [`ctc_decoder.py`](file:///g:/SingLang/SIGNOVA/src/signova/recognition/ctc_decoder.py) with Levenshtein Token Error Rate (TER) evaluation.
5. **Continuous Inference Pipelines**:
   - `WindowedOfflineInference` and `ExperimentalRollingBufferStream` in [`streaming.py`](file:///g:/SingLang/SIGNOVA/src/signova/inference/streaming.py).
6. **Pre-flight CTC Validator**:
   - `check_ctc_data.py` enforcing $T_{\text{in}} \ge T_{\text{target}}$ constraints and NaN/Inf sanity checks.
7. **Regression Test Baseline**:
   - 46 unit/integration tests passing. External reference repositories verified 100% SHA-256 identical.

---

## 2. The Supervision Status & Core Research Challenge

### Current Blocker
- **ISLTranslate**: Contains ~31,222 continuous video-sentence pairs annotated with **English sentence translations**.
- **No aligned token-level or frame-level ISL sign gloss annotations exist.**
- **No aligned temporal sign boundaries exist.**
- **INCLUDE**: Strictly an isolated sign vocabulary (263 isolated word classes).

### Scientific Guardrail
Under strict research rules:
- English sentences will **never** be parsed into pseudo-sign tokens or treated as visual ISL supervision.
- Real-data CTC training is recorded as:
  > `CTC training blocked: valid sequential sign targets unavailable.`
- Phase 5 establishes:
  1. A formal Supervision Blocker Report and Dataset Acquisition Specification.
  2. Generic sequential data adapter architecture (tested on deterministic synthetic fixtures).
  3. Syntactic label normalizer and vocabulary engine.
  4. Sequence length and CTC feasibility analysis.
  5. Phase 4 pretrained encoder transfer representation diagnostics on real continuous sequences.
  6. End-to-end continuous sequence evaluation metrics (TER, edit distance, S/I/D breakdown).
