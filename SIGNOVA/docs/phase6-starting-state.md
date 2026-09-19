# Phase 6 Starting-State Audit

## 1. Executive Summary

Phase 6 marks the critical evaluation and integration stage of **Real Sequential Indian Sign Language (ISL) Supervision and Continuous Sign Language Recognition (CSLR)** in SIGNOVA. 

Prior phases (0 through 5) successfully established a robust, modular, and mathematically verified pipeline:
- **Phase 0–3**: Isolated sign classification baseline (100% test accuracy across 10 classes), MediaPipe 543-landmark topology, landmark normalization, cache management, and feature group extraction (`HANDS` 42, `HANDS_POSE` 75, `FULL` 543).
- **Phase 4**: Continuous sequence dataset loaders (`ContinuousSignDataset`, `ContinuousPadCollate`), temporal window slicing (`SlidingWindowExtractor`), continuous encoders (`ContinuousBiGRUEncoder`, `ContinuousTCNEncoder`), CTC recognizer module (`SignLanguageCTCRecognizer`), greedy CTC decoder (`CTCGreedyDecoder`), and heuristic candidate boundary detector (`HeuristicBoundaryDetector`).
- **Phase 5**: Sequential dataset adapter (`GenericSequentialAdapter`), syntactic label normalizer (`LabelNormalizer`), vocabulary management (`SignVocabulary`), sequential CTC trainer (`SequentialCTCTrainer`), Levenshtein sequence metrics (Token Error Rate, Substitutions, Insertions, Deletions, Exact Match, Token F1), and empirical continuous corpus feature analysis (72 continuous streams, 11,980 frames, 100% pose/hand tracking density).

## 2. Inventory of Reusable Architectural Modules

| Layer | Module Path | Purpose / Capability | Phase 6 Status |
|---|---|---|---|
| **Data Adapter** | `signova.data.adapters.generic_sequential` | Canonical ingestion of sequential sign samples | Reused as-is |
| **Label Normalization** | `signova.data.normalization` | Text cleaning, uppercase tokenization, whitespace collapse | Reused as-is |
| **Vocabulary Engine** | `signova.data.vocabulary` | Index mapping with `<BLANK> = 0`, `<UNK> = 1` | Reused as-is |
| **Temporal Dataset** | `signova.data.continuous` | Dynamic padding, sequence lengths, tensor conversion | Reused as-is |
| **Temporal Encoders** | `signova.models.continuous_encoder` | BiGRU (128 hidden) & Dilated TCN (4 residual blocks) | Reused as-is |
| **CTC Recognizer** | `signova.models.ctc_recognizer` | Linear projection to vocabulary logits + PyTorch CTC Loss | Reused as-is |
| **CTC Decoder** | `signova.models.ctc_decoder` | Argmax beam/greedy decode, blank removal, repeat collapse | Reused as-is |
| **Sequence Metrics** | `signova.evaluation.metrics` | Levenshtein alignment breakdown (TER, S, I, D, Exact Match, F1) | Reused as-is |
| **Sequential Trainer** | `signova.training.sequential_trainer` | PyTorch AMP FP16, grad clipping, validation loop, checkpoints | Reused as-is |

## 3. Baseline Verification State

- **Unit & Integration Tests**: 55 passed out of 55 tests (`pytest`) across data loading, landmarks, models, CTC losses, decoders, and metrics.
- **External Reference Repositories**: 44 out of 44 files in `ISLTranslate-main` (5 files) and `isl-translator-main` (39 files) verified cryptographically identical to the baseline SHA-256 hash manifest.
- **Continuous Feature Corpus**: 72 real continuous ISL video streams preprocessed into normalized `.npz` feature tensors totaling 11,980 frames with complete pose and hand landmark availability.

## 4. Scientific Guardrails for Phase 6

1. **No Target Fabrication**: English sentence translations must **never** be tokenized or treated as visual sign gloss sequences.
2. **No LLM Synthesizers**: No language model or heuristic grammar mapping will be used to fabricate sign targets.
3. **Strict Separation of Experiments**: Synthetic CTC validation must run purely on synthetic fixtures. Real continuous feature diagnostics must never be trained on fake CTC targets.
4. **Hard Data Gate**: Real-data CTC training will only proceed if genuine lexical sign gloss annotations exist. Otherwise, the supervision blocker will be formally documented.
