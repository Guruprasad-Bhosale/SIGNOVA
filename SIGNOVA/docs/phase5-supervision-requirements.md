# Sequential Sign Language Supervision Requirements: SIGNOVA Phase 5

## 1. The Linguistic Problem: English Text vs ISL Sign Sequences

Indian Sign Language (ISL) is a natural visual-spatial language with its own autonomous morphology, syntax, and grammatical structures.

- **English Grammar**: Subject-Verb-Object (SVO) with extensive function words (auxiliaries, prepositions, articles).
- **ISL Grammar**: Typically Subject-Object-Verb (SOV) / Topic-Comment, with non-manual facial markers, spatial referencing, and classifier predicates.

> [!CRITICAL]
> **Why English Sentences Cannot Be Tokenized into ISL Sign Labels:**
> An English translation such as:
> `English: "What is your name?"`
> corresponds to an ISL gloss sequence such as:
> `ISL Gloss: YOUR NAME WHAT`
> Tokenizing the English sentence into `"What" -> "is" -> "your" -> "name?"` produces invalid sign supervision because:
> 1. The copula `"is"` has no signed manual equivalent.
> 2. The word order differs fundamentally.
> 3. Visual morphological inflections are ignored.

---

## 2. Dataset Acquisition Specification for Continuous ISL Recognition

To train authentic continuous sign recognition models (CSR), any candidate dataset must satisfy the following specification:

| Field | Required Format | Description |
|---|---|---|
| **Video Stream** | MP4 / WebM ($\ge 25\text{ fps}$, $\ge 720\text{p}$) | Uninterrupted frontal view of continuous signing. |
| **Ordered Gloss Sequence** | List of canonical sign tokens | e.g. `["NAMASTE", "MEETING", "TODAY"]` |
| **Temporal Boundaries (Optional/Ideal)** | Frame / Millisecond intervals | `[(0, 35), (36, 80), (81, 120)]` |
| **Signer Metadata** | Verified Signer IDs | Enables signer-independent train/val/test splits. |
| **Annotation Provenance** | Certified Deaf / Expert annotators | Documentation of annotation conventions. |

---

## 3. Phase 5 Strategy Under Blocked Supervision

1. **Architecture & Pipeline Completion**:
   - Implement generic sequential data adapters.
   - Implement syntactic label normalizers and vocabulary engines.
   - Implement continuous CTC loss and greedy decoders with Levenshtein metrics.
2. **Controlled Verification**:
   - Unit-test all sequential modules against deterministic **SYNTHETIC** fixtures.
3. **Representation Diagnostics on Real Sequences**:
   - Run continuous representation profiling and transfer analysis on real ISLTranslate feature sequences.
4. **Honest Reporting**:
   - Report real-data CTC training as `BLOCKED` with no fabricated accuracy claims.
