# SIGNOVA Phase 23: First Real CTC Run, Compatibility Gating & Live Integration

## 1. Gated Execution Pipeline

Real CTC training is executed exclusively through `scripts/run_phase23_training.py` under two strict prerequisites:
1. **Canonical Readiness Authority**: `evaluate_phase19_readiness()["real_ctc_training_allowed"] == True`.
2. **Explicit User Confirmation**: Passing `--train` argument.
3. **Compatibility Gate**: Validating `dataset_sha256`, `vocabulary_size`, `split_fingerprint`, and model input specification.

Without `--train`, the system reports:
```
[!] TRAINING ACTION: NOT REQUESTED
```

---

## 2. Multi-Dimensional State Engine

Phase 23 decouples supervision state from training readiness and execution:
- `SUPERVISION_STATE`: `STATE_B` $\to$ `STATE_A_DATA_LIMITED` $\to$ `STATE_A`
- `TRAINING_READINESS`: `NOT_READY` $\to$ `READY_LIMITED` $\to$ `READY_RESEARCH_SCALE`
- `TRAINING_EXECUTION`: `NOT_REQUESTED` $\to$ `BLOCKED` $\to$ `RUNNING` $\to$ `COMPLETED` $\to$ `FAILED`

---

## 3. Dataset Freezing & Versioning

Datasets are frozen into immutable versioned directories (`phase23_dataset_v001`, `phase23_dataset_v002`, ...). Mid-training mutation invalidates the experiment run, preserving cryptographic lineage.
