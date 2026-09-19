# Model Card: SIGNOVA Phase 12 Continuous ISL CTC Recognizer

## Model Details
- **Architecture**: `ContinuousBiGRUEncoder` + CTC Loss Recognition Head
- **Input Representation**: Continuous MediaPipe Landmark Features (`HANDS_POSE`, 150-dimensional per frame)
- **Supervision Target**: Ordered Indian Sign Language (ISL) Gloss Token Sequences
- **Vocabulary Tokens**: Project ISL Glosses with `<BLANK> = 0`, `<UNK> = 1`
- **Execution Mode**: Offline and Windowed Continuous Sequence Recognition

## Intended Use
- Research baseline for continuous Indian Sign Language (ISL) sequence recognition.
- Strictly conditioned on genuine, verified human gloss annotations.

## Prohibited Claims & Limitations
- **No Signer Independence Claims**: Generalization across unseen signers cannot be claimed unless verified signer IDs are partitioned independently across train/val/test splits.
- **No Pseudo-Ground-Truth**: English translations, LLM outputs, and heuristic segment proposals must never be used as supervision labels.
- **Not Real-Time Commercial Translation**: Sub-second algorithmic latency does not equate to a deployment-ready end-to-end communication tool for Deaf users.

## Dataset & Annotation Governance
- **Collection Status**: Ingested via Phase 11/12 Human Annotation Platform.
- **Review Requirement**: Minimum `VERIFIED` or `LINGUIST_REVIEWED` quality grade with recorded annotator and reviewer separation.
- **Authorized Spend**: ₹0 (`ZERO_COST_MODE = DEFAULT`).
