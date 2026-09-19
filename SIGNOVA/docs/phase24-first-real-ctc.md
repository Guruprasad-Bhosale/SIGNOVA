# Phase 24: First Genuine Sequential ISL CTC Training Architecture

## Overview

Phase 24 establishes the complete orchestration, execution lock, and validation protocol for the first real CTC experiment in SIGNOVA.

### Architectural Invariants

1. **Phase 19 Gate Authority**: Real CTC training can never begin unless Phase 19 explicitly transitions from `STATE_B` to `AUTHORIZED`.
2. **Phase 21 Training Delegation**: Phase 24 acts strictly as the experiment orchestrator and does not duplicate CTC training or loss calculation logic. All training execution delegates to `Phase21Pipeline` and `CTCTrainer`.
3. **Explicit Confirmation**: Training requires both Phase 19 authorization and the explicit `--train` confirmation flag.
4. **Dynamic Input Specification**: Landmark topology (543 landmarks), feature grouping (`HANDS_POSE`), normalization (`v1.0.0`), and input dimensions are dynamically bound from `signova.live.model_registry.ModelInputSpec`.
5. **Phase 24 Experiment Lock**: Before any training commences, an immutable lock file (`phase24_experiment_lock.json`) binds:
   - `dataset_sha256`
   - `vocabulary_sha256`
   - `split_fingerprint`
   - `input_spec_fingerprint`
   - `feature_schema_version`
   - `normalization_version`
6. **Zero-Cost & Zero-Fabrication**: Strict adherence to zero compute/cloud costs and zero synthetic/pseudo-labeled data.

## Execution Flow

```
Phase 19 Gate Evaluation
        │
  ┌─────┴──────┐
  │            │
BLOCKED     AUTHORIZED
  │            │
  ▼            ▼
Fast Exit   Check --train Flag
(STATE_B)      │
         ┌─────┴──────┐
         │            │
       FALSE         TRUE
         │            │
         ▼            ▼
      Refuse       Verify Split & Input Spec
                      │
                      ▼
                   Generate Experiment Lock
                      │
                      ▼
                   Delegate to Phase 21 CTC Trainer
                      │
                      ▼
                   Produce Checkpoint & Manifest
```

## Current State

As genuine human sequential ISL annotations are currently pending acquisition (`0` qualified annotations), Phase 24 safely exits with `STATE_B`, keeping training cleanly blocked without synthetic artifacts.
