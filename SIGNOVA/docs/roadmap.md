# SIGNOVA Project Roadmap

Multi-phase research and engineering roadmap for the Continuous Indian Sign Language Translation System.

---

## Phase Overview

- [x] **PHASE 0: Foundation, Architecture & Environment Setup** *(COMPLETE)*
  - Modular project structure, environment checker, YAML configs, dataset boundaries, MediaPipe landmark abstraction, FastAPI/React scaffolding, automated test suite.
- [x] **PHASE 1: Dataset Ingestion, Deep Validation & Canonical Data Pipeline** *(COMPLETE)*
  - Physical inventory of reference data, 31,222-sample integrity audit, text vocabulary analytics, gloss availability determination, signer metadata verification, session-independent benchmark split generation, canonical `SignSample` schema, dataset validation CLI, and remote acquisition blueprint.
- [x] **PHASE 2: Remote Data Acquisition, Video Pipeline & Landmark Extraction** *(COMPLETE)*
  - Remote dataset resolver (Hugging Face / HTTP), concurrency-safe video cache manager with processing locks and LRU eviction, robust video decoder context manager, MediaPipe Holistic 543-topology extractor (33 pose + 468 face + 21 LH + 21 RH), explicit $(T \times 4)$ binary detection masks, multi-tier spatial normalization (mid-hip/nose, shoulder/bbox), landmark quality evaluator, compressed `.npz` storage with SemVer `0.1.0` headers, extraction CLI with bulk safety gate (`--allow-bulk`), pilot extraction of 25 samples, visual landmark overlays, and numerical integrity diagnostic CLI.

- [x] **PHASE 3: Isolated / Dynamic Sign Recognition Baseline** *(COMPLETE)*
  - Physical inspection of INCLUDE dataset availability, 10-class isolated dynamic benchmark suite (280 validated samples), centralized feature grouping (`HANDS` 42, `HANDS_POSE` 75, `FULL` 543), variable-length `PadCollate` with boolean masks, Baseline 0.5 (Static Pooled MLP), Baseline A (BiGRU with 2-stage projection), Baseline B (Dilated 1D Residual TCN), AMP/gradient accumulation trainer with early stopping, model summary profiling CLI, offline prediction CLI, comprehensive error & confusion analysis, and feature ablation study.
- [x] **PHASE 4: Continuous Sign Sequence Modeling, Temporal Segmentation & CTC Readiness** *(COMPLETE)*
  - Continuous feature dataset (`ContinuousSignDataset`), variable-length batch collation (`ContinuousPadCollate`), sliding temporal windowing with zero split leakage (`SlidingWindowExtractor`), continuous temporal encoders preserving per-frame $(B, T, D)$ representations (`ContinuousBiGRUEncoder`, `ContinuousTCNEncoder`), heuristic candidate boundary detection (`HeuristicBoundaryDetector`), end-to-end CTC recognizer (`CTCContinuousRecognizer`), greedy CTC decoder (`CTCDecoder`) with Levenshtein TER metrics, CTC pre-flight validator (`check_ctc_data.py`), windowed offline inference CLI (`predict_continuous.py`), empirical continuous representation & Phase 3 transfer experiments on ISLTranslate sequences, diagnostic timeline visualizations, and 46 passing unit/integration tests.
- [x] **PHASE 6: Real Sequential ISL Dataset Integration & Genuine CTC Recognition** *(COMPLETE)*
  - Complete starting-state audit, dataset discovery across all local repositories, deep annotation semantics audit, Hard Data Gate evaluation (STATE C: supervision blocker formally documented), continuous ISL dataset acquisition specification, pure synthetic CTC optimization & sequence metrics benchmark across architectures (BiGRU vs TCN) and feature groups (`HANDS`, `HANDS_POSE`, `FULL`), real continuous feature corpus profiling (72 streams, 11,980 frames, 100% pose/hand tracking density), Phase 4 -> Phase 6 transfer diagnostics, latency/throughput profiling across $T \in \{32, 64, 128, 256\}$, qualitative sequence alignments with Levenshtein S/I/D breakdown, 44/44 reference files cryptographically verified, model card and reproducibility package.
- [ ] **PHASE 7: End-to-End Continuous Sign Language Translation (SLT) & Gloss-to-Text NMT**
  - Sign/gloss sequence to English natural language translation modeling.
- [ ] **PHASE 8: Real-Time & Offline Inference Engine**
  - Real-time sliding window inference buffer, latency optimization, OpenCV webcam capture integration.
- [ ] **PHASE 9: FastAPI + WebSocket Streaming Gateway**
  - Bi-directional WebSocket endpoint streaming live webcam frames and returning partial/final translations.
- [ ] **PHASE 10: Interactive React + TypeScript Frontend**
  - Production webcam UI, landmark overlay visualizer, real-time transcription history, accessibility controls.
- [ ] **PHASE 11: Evaluation & Benchmarking**
  - Formal evaluation across WER, CER, BLEU-1/2/3/4, ROUGE-L, METEOR, latency profiles on test splits.
- [ ] **PHASE 12: Hardware & Edge Optimization**
  - FP16 TensorRT / ONNX export, INT8 quantization for RTX 3050 and lightweight CPU execution.
- [ ] **PHASE 13: Optional Hardware & Audio Integration**
  - Offline text-to-speech (TTS), optional ESP32 micro-controller output over serial/Bluetooth to OLED display.
