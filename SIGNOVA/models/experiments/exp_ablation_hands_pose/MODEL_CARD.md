# Model Card: SIGNOVA BiGRU Isolated Dynamic Sign Recognizer (Hands + Pose)

## Model Overview
- **Model Name**: SIGNOVA Baseline A (Hands + Pose BiGRU)
- **Architecture**: 2-Stage Linear Projection ($225 \to 256 \to 256$) + 2-Layer Bidirectional GRU (Hidden Size: 256, Dropout: 0.2) + Masked Temporal Pooling + MLP Classifier Head.
- **Input Dimension**: $(T \times 75 \times 3)$ normalized coordinates + $(T \times 4)$ binary detection masks + $(B, T)$ boolean padding masks.
- **Output**: 10-Class Probability Distribution over ISL dynamic sign categories (`HELLO`, `THANK_YOU`, `PLEASE`, `YES`, `NO`, `NAME`, `HOW_ARE_YOU`, `HELP`, `GOODBYE`, `WELCOME`).
- **Total Trainable Parameters**: 2,230,794 (~8.51 MB FP32).

---

## Intended Use
- **Primary Use**: Word-level isolated dynamic sign gesture recognition from skeletal landmark sequences.
- **Evaluation Benchmark**: Serves as the validated temporal encoder baseline before continuous sentence translation (Phase 5+).

---

## Out-of-Scope & Non-Intended Uses
- **Continuous Sign Language Translation**: This model does NOT perform sentence-level translation, grammar alignment, or gloss-to-English text generation.
- **Direct Video Ingestion**: Expects pre-extracted, spatially normalized MediaPipe skeletal coordinates.

---

## Training & Evaluation Data
- **Dataset**: SIGNOVA Isolated Dynamic Sign Benchmark (280 samples, 10 categories, 8 recording sessions).
- **Split**: Track B Session-Independent Partition (Train: Sessions 1-6, Val: Session 7, Test: Session 8).
- **Session Leakage**: Zero session overlap across splits.

---

## Performance Summary
- **Test Accuracy**: 100.0% (30/30 on held-out session 8).
- **Test Macro F1**: 1.0000.
- **Inference Latency**: ~3.2 ms per sequence on CPU / < 1 ms on RTX 3050 CUDA.
