# ISLTranslate Duplicate & Repetition Analysis

Comprehensive evaluation of identifier collisions, raw UID duplicates, and cross-split textual repetitions.

---

## 1. Findings Summary

| Category | Count | Classification | Action / Impact |
|---|---|---|---|
| **Sample ID Collisions in SIGNOVA Manifest** | **0** | `SAFE` | All 31,222 keys are strictly unique |
| **Raw UID Duplications in Source CSV** | **1 UID (2 rows)** (`UbhONRGm-90--61`) | `SAFE` | Disambiguated to `_dup1` in canonical manifest |
| **Identical English Sentences across Splits** | **534 distinct texts** | `REVIEW` | Common phrases (e.g. "What is this?", "Page 10") signed by different signers |
| **Cross-Split Video Collision** | **0** | `SAFE` | Each video sample exists in exactly one split |

---

## 2. Classification Schema & Guidance

- **`SAFE`**: Expected structural phenomena that do not jeopardize scientific evaluation (e.g. unique sample IDs, distinct video clips).
- **`REVIEW`**: Common natural language phrases that appear across training, validation, and testing. In continuous sign language translation, different video performances of the same English sentence are standard multi-speaker instances.
- **`LEAKAGE_RISK`**: Identical video instances mapped to multiple splits (none detected).
