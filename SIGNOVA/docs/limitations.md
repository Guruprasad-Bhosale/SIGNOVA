# SIGNOVA Limitations & Known Challenges

Documentation of hardware, algorithmic, and linguistic limitations identified during Phase 0 design.

---

## 1. Hardware & VRAM Constraints
- **GPU Capacity**: The local development machine has 6 GB VRAM (RTX 3050). Training very large end-to-end multi-modal models simultaneously with 1B+ parameter LLMs will exceed local memory.
- **Mitigation**: Distribute training into two discrete phases: (1) Pose-only continuous sign recognition (ST-GCN + Transformer + CTC), and (2) Lightweight gloss-to-English translation. Cloud compute (Kaggle / Colab / Cloud GPU) can be used for heavier training jobs.

---

## 2. Dataset Specifics
- **ISL Regional Dialects**: ISL displays regional variations across India. The ISLTranslate dataset captures specific signer domains.
- **Facial Non-Manual Markers**: Hand gestures carry primary lexical information, but grammatical nuances and questions rely heavily on facial expressions and head tilt (hence the retention of all 468 face landmarks).
- **Gold Standard Samples**: 291 samples in ISLTranslate have certified human expert gold validation. Model evaluation will benchmark against both transcribed text and human gold sets.

---

## 3. Real-Time Inference Challenges
- **Sliding Window Boundary Ambiguity**: In continuous sign language, the transition between signs (co-articulation) can blur segment boundaries. CTC decoding with prefix beam search is chosen specifically to mitigate alignment ambiguity.
- **Occlusion**: Fast hand movements or hand-over-face gestures can lead to temporary MediaPipe landmark tracking drops. Spatial-temporal interpolation and normalization routines will handle missing frames.
