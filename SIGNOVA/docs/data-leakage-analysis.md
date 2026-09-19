# SIGNOVA Data Leakage & Split Overlap Analysis

Technical assessment of signer metadata, session continuity, and partition independence.

---

## 1. Signer Metadata Verification

- **Documented Status**: **`UNKNOWN`**
- **Finding**: Raw `ISLTranslate.csv` provides `uid` (e.g., `1782bea75c7d-10` or `aTB_lu2Im8Y--70`) and English translation text. The dataset documentation does NOT explicitly declare the UID prefix as an individual human signer ID.
- **True Nature of UID Prefix**: The prefix represents the **Source Video / Session ID** (e.g. `aTB_lu2Im8Y` is an 11-character YouTube video container ID).
- **Scientific Decision**: We avoid labeling this field as `signer_id` and explicitly denote it as `source_session_id`.

---

## 2. Partition Overlap in Sample-Level Random Hashing

In the official Phase 0 random sample split (Train: 25,004, Val: 3,116, Test: 3,102):
- **Total Unique Source Video Sessions**: 284
- **Train Sessions**: 284
- **Validation Sessions**: 280
- **Test Sessions**: 276
- **Session Overlap**: Sentences from the same long source video session appear in both training and test partitions (sentence-level split).

---

## 3. Proposed Session-Independent Benchmark Split

To establish a gold-standard leak-free benchmark for future research evaluation, SIGNOVA generated `data/manifests/proposed_signer_independent_split.csv`:

| Partition | Proposed Session-Independent Sample Count | Percentage (%) | Session Count |
|---|---|---|---|
| **Train** | 24,981 | 80.01% | 227 sessions |
| **Validation** | 3,370 | 10.79% | 28 sessions |
| **Test** | 2,871 | 9.20% | 29 sessions |
| **Session Intersections** | **0** ($Train \cap Test = \emptyset$) | **0.00%** | **0 overlap** |

*Note: The official Phase 0 split remains the primary baseline to preserve comparison with external published benchmarks, while the session-independent split provides an auxiliary evaluation track.*
