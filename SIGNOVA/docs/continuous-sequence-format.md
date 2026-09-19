# Continuous Sequence Format Specification: SIGNOVA Phase 4

## 1. Sequence Tensor Representation

Continuous sign language videos vary in duration from short phrases ($2\text{--}3\,\text{s}$) to extended discourse ($10\text{--}30\,\text{s}$). SIGNOVA standardizes all continuous temporal representations into unified, variable-length skeletal arrays:

$$\mathbf{X} \in \mathbb{R}^{T \times K \times C}$$

where:
- $T$: Number of video frames in the continuous sequence.
- $K$: Number of anatomical joint landmarks according to selected `LandmarkGroup` ($K=543$ for `FULL`, $K=75$ for `HANDS_POSE`, $K=42$ for `HANDS`).
- $C=3$: Spatial coordinates $(x, y, z)$ normalized to the body bounding frame.

---

## 2. Associated Metadata Channels

Each continuous sequence is accompanied by synchronized per-frame telemetry:

1. **Binary Detection Masks** $\mathbf{M}_{\text{det}} \in \{0, 1\}^{T \times 4}$:
   - Column 0: Pose detection flag.
   - Column 1: Face detection flag.
   - Column 2: Left hand detection flag.
   - Column 3: Right hand detection flag.
2. **Timestamps** $\mathbf{t} \in \mathbb{R}^T$:
   - Real-world millisecond timestamps for each frame.
3. **Frame Indices** $\mathbf{i} \in \mathbb{Z}^T$:
   - Original video source frame indices.
4. **Padding Validity Mask** $\mathbf{M}_{\text{pad}} \in \{0, 1\}^{B \times T_{\max}}$:
   - Boolean mask indicating valid non-padded frames across dynamic batches.

---

## 3. Storage & Lazy Loading

Features are stored in compressed `.npz` format:
- `landmarks`: `float32` array $(T, 543, 3)$
- `detection_masks`: `float32` array $(T, 4)$
- `timestamps_ms`: `float32` array $(T,)$
- `frame_indices`: `int64` array $(T,)$
- `metadata_json`: Serialized string containing sample ID, split, FPS, and provenance.

Loaded lazily via [`ContinuousSignDataset`](file:///g:/SingLang/SIGNOVA/src/signova/data/continuous_dataset.py) and padded dynamically via [`ContinuousPadCollate`](file:///g:/SingLang/SIGNOVA/src/signova/data/continuous_dataset.py).
