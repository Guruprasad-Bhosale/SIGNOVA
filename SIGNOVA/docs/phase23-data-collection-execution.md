# SIGNOVA Phase 23: Genuine ISL Data Collection Execution & Batch Qualification

## 1. Overview & Execution Architecture

Phase 23 connects human data acquisition in Phase 22 directly to the gated real CTC pipeline in Phase 21 without duplicating training code or creating competing authorization gates.

```
                    HUMAN SIGNERS
                         │
                         ↓
                ┌─────────────────┐
                │    PHASE 22     │
                │ Annotation/Data │
                └────────┬────────┘
                         │
                         ↓
                  FROZEN DATASET
                         │
                         ↓
                ┌─────────────────┐
                │    PHASE 19     │
                │ Canonical Gate  │
                └────────┬────────┘
                         │
                   AUTHORIZED?
                    /         \
                  NO           YES
                  │             │
                STOP            ↓
                         explicit --train
                               │
                               ↓
                     ┌─────────────────┐
                     │    PHASE 21     │
                     │  Real CTC       │
                     │ Train/Evaluate  │
                     └────────┬────────┘
                              │
                              ↓
                     CHECKPOINT VERIFIED
                              │
                              ↓
                       LIVE AUTHORIZED
                              │
                              ↓
                     ┌─────────────────┐
                     │    PHASE 20     │
                     │ Live Camera     │
                     │ CTC → Gloss     │
                     │ → English       │
                     └─────────────────┘
```

---

## 2. 14-Point Batch Data Accounting

Batch human annotation ingestion via `scripts/import_phase23_human_data.py` tracks:
1. `FILES_DISCOVERED`
2. `ANNOTATIONS_PARSED`
3. `ANNOTATIONS_VALID`
4. `ANNOTATIONS_INVALID`
5. `SOURCE_HASH_MATCH`
6. `SOURCE_HASH_MISMATCH`
7. `QUALIFIED_ANNOTATORS`
8. `UNQUALIFIED_ANNOTATORS`
9. `SUBMITTED`
10. `VERIFIED`
11. `REJECTED`
12. `REVISION_REQUIRED`
13. `TRAINING_ELIGIBLE`
14. `TRAINING_INELIGIBLE`

Every rejected or ineligible record is categorized by machine-readable rejection reasons (e.g. `PROHIBITED_SOURCE_*`, `ANNOTATOR_NOT_QUALIFIED_*`, `SOURCE_HASH_MISMATCH`, `NOT_TEMPORALLY_ALIGNED`).

---

## 3. Double Annotation & Agreement Status

When samples are assigned for double annotation:
- Tracks independent `annotation_A` and `annotation_B`.
- Reports `AGREEMENT_STATUS`: `NOT_COMPUTABLE` (when zero completed pairs exist) or `COMPUTABLE` with exact sequence and token-level agreement rates.
