# Phase 5 Sequential Data Format Specification

## 1. Canonical Sequential Manifest Format

When continuous ISL video sequences are paired with authentic ordered sign/gloss annotations, the canonical manifest format is standardized as:

| Column Name | Type | Description |
|---|---|---|
| `sample_id` | `str` | Unique video sequence identifier (e.g. `isl_seq_001`). |
| `dataset` | `str` | Name of the source dataset. |
| `split` | `str` | Deterministic split assignment (`train`, `val`, `test`). |
| `session_id` | `str` | Unique recording session identifier for cross-session validation. |
| `signer_id` | `str` | Verified signer identity (or `unverified`). |
| `original_label` | `str` | Raw untouched source annotation string. |
| `canonical_tokens` | `List[str]` | Syntactically normalized sign/gloss token list (e.g. `['HELLO', 'WORLD']`). |
| `sequence_length` | `int` | Number of tokens in the canonical gloss target sequence ($U$). |
| `num_frames` | `int` | Number of frames in the continuous video sequence ($T$). |
| `feature_path` | `str` | Path to compressed `.npz` skeletal landmark features. |
| `is_synthetic` | `bool` | Explicit flag: `True` for synthetic fixtures, `False` for real datasets. |

---

## 2. Ingestion Adapter Architecture

Implemented in [`GenericSequentialAdapter`](file:///g:/SingLang/SIGNOVA/src/signova/data/adapters/sequential_adapter.py):

- **Normalizer**: Integrates [`LabelNormalizer`](file:///g:/SingLang/SIGNOVA/src/signova/preprocessing/label_normalizer.py) to standardize casing and whitespace without modifying semantics.
- **Verification Rule**: The adapter is marked `SYNTHETIC_FIXTURE_VALIDATED` unless external datasets with genuine sequential sign annotations are verified on disk.
