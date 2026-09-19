# Phase 6 Limitations & Scientific Boundary Document

## 1. Explicit Scientific Limitations

1. **Supervision Constraint**:
   - Continuous Indian Sign Language lexical sign gloss sequences are currently unavailable locally.
   - CTC models cannot be trained on real ISL video targets until native sequential gloss annotations are acquired.
2. **Recognition vs Translation Boundary**:
   - Phase 6 focuses strictly on Continuous Sign Language Recognition (CSLR): predicting sequences of ISL signs/glosses from continuous landmark motion.
   - Continuous sign recognition is **not** English translation. Spoken language translation belongs to Phase 7+.
3. **Signer Independence**:
   - In the absence of verified biological signer IDs across continuous sentences, signer-independent evaluation cannot be scientifically validated on `ISLTranslate`.
4. **Offline Bidirectional vs Causal Streaming**:
   - The BiGRU backbone utilizes bidirectional recurrence, requiring future temporal context within each window. It is suitable for windowed or offline inference, but is not a causal real-time streaming model.

## 2. Prohibition on Shortcuts

- No LLM synthesis of gloss targets.
- No direct mapping of English translation words to visual signs.
- No pseudo-labeling of unannotated video segments.
