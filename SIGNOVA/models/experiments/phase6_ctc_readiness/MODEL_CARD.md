# Model Card: SIGNOVA Phase 6 CTC Continuous Sign Recognizer

## Model Details
- **Architecture**: Continuous Temporal Encoder (BiGRU / TCN) + Linear Projection + PyTorch CTC Loss
- **Input Representation**: MediaPipe 543-topology normalized landmarks (HANDS, HANDS_POSE, FULL)
- **Vocabulary Setup**: CTC Blank (<BLANK> = 0), Unknown (<UNK> = 1), Lexical tokens
- **Hardware Profile**: NVIDIA RTX 3050 Laptop GPU / CPU Fallback

## Intended Use
- **Primary Function**: Continuous Indian Sign Language (ISL) sign/gloss sequence recognition.
- **Explicit Limitation**: "This model performs continuous ISL sign/gloss recognition. It does not perform English translation."

## Data Gate & Supervision State
- **Status**: STATE C (Supervision Blocker Documented)
- **Rationale**: Real continuous ISL feature streams (72 streams, 11,980 frames) are profiled for temporal representation and latency. Real CTC training is formally blocked until native lexical gloss annotations are acquired.

## Quantitative Benchmark Summary (Synthetic CTC Optimization)
- **Best Backbone**: TCN_HANDS_POSE
- **Test Token Error Rate (TER)**: 0.8929
- **Test Sequence Exact Match**: 0.0
- **Test Macro F1**: 0.4914
