# Phase 25: Dataset Freeze, Immutability & Cryptographic Fingerprinting

## Overview

The dataset freeze protocol guarantees that once a dataset partition is qualified and locked, no samples, labels, splits, or configurations can mutate without forcing the creation of a new dataset version identifier.

---

## Fingerprint Components

The Phase 25 dataset fingerprint binds the following elements cryptographically:

1. `sample_ids`: Alphabetically sorted list of all included sample IDs.
2. `source_sha256_map`: Mapping of sample ID to source video file SHA-256 hash.
3. `annotation_id_map`: Mapping of sample ID to human annotation ID.
4. `revision_id_map`: Mapping of sample ID to verified revision ID.
5. `ordered_gloss_ids_map`: Mapping of sample ID to integer gloss token IDs.
6. `split_assignments`: Partition assignment (`TRAIN`, `VALIDATION`, `TEST`).
7. `signer_id_map`: Signer identifier for signer-independence verification.
8. `vocabulary_sha256`: SHA-256 hash of canonical vocabulary mapping.
9. `feature_schema_version`: Feature schema identifier (`1.0.0`).
10. `normalization_version`: Feature normalization version (`1.0.0`).
11. `input_spec_fingerprint`: Model input geometry hash (543 landmarks, `HANDS_POSE`).
12. `dataset_manifest_sha256`: SHA-256 hash of dataset manifest file.
13. `composite_sha256`: Aggregate SHA-256 combining all components above.

---

## Freezing Commands

To build a candidate dataset:
```bash
python scripts/run_phase25_build_dataset.py --version phase25_v001
```

To freeze and generate lock file:
```bash
python scripts/run_phase25_freeze_dataset.py --version phase25_v001
```
