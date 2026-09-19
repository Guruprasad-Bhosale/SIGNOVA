# Model Card: Phase 4 Continuous BiGRU Encoder

## Intended Use
Continuous per-frame temporal representation $(B, T, D)$ and windowed sequence encoding for Indian Sign Language.

## Architecture
- 2-Stage Spatial Projection Layer: Linear(225 $\rightarrow$ 256) + LayerNorm + GELU
- 2-Layer Bidirectional GRU: hidden_size=256, output_dim=512
- Preserves per-frame temporal representation $(B, T, 512)$ without global temporal pooling.

## Evaluation & Transfer
- Forward Latency: 20.038 ms (8303.6 fps)
- Verified non-leaking window extraction and deterministic split allocation.
