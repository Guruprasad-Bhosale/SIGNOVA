# Model Card: SIGNOVA Dilated TCN Isolated Dynamic Sign Recognizer (Full Body)

## Model Overview
- **Model Name**: SIGNOVA Baseline B (Dilated Temporal Convolutional Network)
- **Architecture**: 2-Stage Linear Projection ($1629 \to 256 \to 256$) + 3-Block Dilated Residual 1D Convolutions (Dilation rates: 1, 2, 4; Channels: [256, 256, 256]) + Masked Temporal Pooling + MLP Classifier Head.
- **Input Dimension**: $(T \times 543 \times 3)$ normalized coordinates + $(T \times 4)$ binary detection masks + $(B, T)$ boolean padding masks.
- **Output**: 10-Class Probability Distribution over ISL dynamic sign categories.
- **Total Trainable Parameters**: 1,702,538 (~6.49 MB FP32).

---

## Intended Use
- **Primary Use**: Lightweight, highly parallelizable temporal landmark sequence classification.
- **Efficiency**: Well-suited for low-latency edge deployment and streaming inference buffers.

---

## Performance Summary
- **Test Accuracy**: 90.0% (27/30 on held-out session 8).
- **Test Macro F1**: 0.8667.
- **Training Time**: 25.99s (15 epochs).
- **Inference Latency**: ~2.1 ms per sequence on CPU.
