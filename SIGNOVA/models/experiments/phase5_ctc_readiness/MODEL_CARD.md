# Model Card: Phase 5 CTC Continuous Recognizer (Readiness Baseline)

## Intended Use
Continuous sign/gloss sequence recognition architecture for Indian Sign Language.

## Architecture
- Input: Landmark sequences $(B, T, 75, 3)$
- Spatial Projection: 2-Stage Linear + LayerNorm + GELU
- Temporal Backbone: 2-Layer Continuous BiGRU $(d_h=256)$
- Recognition Head: Linear Projection to 12 vocabulary classes with `<BLANK> = 0`.
- Loss: Native PyTorch CTCLoss.

## Status & Limitations
- **Synthetic Verification**: Loss optimized from 50.4207 to 5.3277.
- **Real Data Status**: Real-world CTC training is **BLOCKED** due to lack of aligned ISL sign/gloss sequence supervision in ISLTranslate.
- **Translation Disclaimer**: This model performs continuous sign sequence recognition and does NOT perform English translation.
