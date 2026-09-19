# Research Limitations & Annotation Requirements: SIGNOVA Phase 4

## 1. Distinction Between Tasks

In the sign language research literature, three distinct problems must never be conflated:

1. **Isolated Sign Recognition (ISR)**:
   - Input: Pre-segmented video of one sign.
   - Output: Categorical sign class ID (e.g. 1-of-10 in Phase 3).
2. **Continuous Sign Recognition (CSR)**:
   - Input: Continuous video with multiple unsegmented signs.
   - Output: Sequential sign/gloss tokens (e.g., `[NAMASTE, THANK_YOU, YOU]`).
   - Requires: Token-level / gloss-level sequential supervision.
3. **Continuous Sign Translation (SLT)**:
   - Input: Continuous sign video.
   - Output: Natural spoken/written language sentences (e.g., English text).
   - Requires: Sequence-to-sequence translation architecture (Video $\rightarrow$ Text or Gloss $\rightarrow$ Text).

---

## 2. Specific Limitations of Phase 4

1. **Absence of Token-Level Gloss Supervision**:
   - `ISLTranslate` provides sentence-level English translations, **not** ISL sign gloss sequences.
   - Treating English words as visual ISL sign labels is scientifically invalid.
   - Therefore, CTC models are verified on synthetic fixtures, and real-data CTC training remains blocked.
2. **Absence of Ground-Truth Temporal Boundaries**:
   - No frame-level start/end timestamps exist for individual signs in `ISLTranslate`.
   - Heuristic candidate boundary proposals are generated for visualization only.
3. **Signer Identity Limitations**:
   - Signer validation must not be assumed from UID prefixes.
4. **Constructed Benchmark Scope**:
   - Phase 3's 100% accuracy was achieved on a controlled 10-class isolated dynamic benchmark and must not be presented as continuous translation accuracy.

---

## 3. Supervision Requirements for Phase 5

To unlock full continuous sequence recognition and end-to-end translation:
1. Direct End-to-End Video-to-Text translation modeling (e.g. encoder-decoder / cross-attention) trained against sentence-level English text.
2. Verified multi-signer datasets with aligned lexical glosses if CSR intermediate representations are desired.
