# Streaming & Windowed Inference Interface: SIGNOVA Phase 4

## 1. Operational Inference Modes

SIGNOVA Phase 4 distinguishes between two inference modalities:

### Mode 1: Windowed Offline Inference (Primary Standard)
- Implemented in [`WindowedOfflineInference`](file:///g:/SingLang/SIGNOVA/src/signova/inference/streaming.py).
- Splits continuous video sequences into overlapping temporal windows ($W=64$, $S=32$).
- Computes per-frame representations across each window.
- Merges overlapping representations via weighted averaging across the video timeline.
- Fully compatible with bidirectional encoders (`ContinuousBiGRUEncoder`, dilated `ContinuousTCNEncoder`).

### Mode 2: Experimental Rolling Buffer Streaming (Optional)
- Implemented in [`ExperimentalRollingBufferStream`](file:///g:/SingLang/SIGNOVA/src/signova/inference/streaming.py).
- Maintains a FIFO feature buffer of length $W$ and triggers frame representations every $K$ frames.

---

## 2. Scientific Note on Latency & Causality

> [!NOTE]
> **Causality Constraint**:
> Bidirectional recurrent models (BiGRU) and non-causal TCNs observe both backward and forward frames within their temporal receptive field.
> Therefore, Phase 4 makes **no claim of zero-latency real-time causal streaming**.
> Windowed offline inference is the primary certified operational mode. True zero-latency streaming requires unidirectional causal backbones, planned for subsequent phases.
