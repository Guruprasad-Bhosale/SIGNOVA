# Phase 4 Continuous Dataset & Label Audit

## Overview
This audit establishes the empirical ground-truth annotation availability across all local datasets in the SIGNOVA workspace.

## Audit Findings Matrix

| Dataset | Sample Count | Primary Label Type | Gloss Sequences | Frame Glosses | Boundaries | Signer Metadata | Session Metadata |
|---|---|---|---|---|---|---|---|
| **ISLTranslate** | 31,222 pairs | Sentence-level English Text | ❌ None | ❌ None | ❌ None | ❌ Unverified | ❌ None |
| **Phase 3 Benchmark** | 280 samples | Isolated Dynamic Class ID | ❌ N/A (Isolated) | ❌ None | ❌ None | ❌ Unverified | ✅ Yes (`session_id`) |
| **INCLUDE (Adapter)** | Isolated archive | Isolated Word Gloss ID | ❌ N/A (Isolated) | ❌ None | ❌ None | ⚠️ Metadata only | ⚠️ Folder level |

## Key Empirical Determinations

1. **Absence of Sequential Sign Supervision**:
   - `ISLTranslate.csv` contains `uid` and `text` (English translation sentences).
   - There are **no frame-level, token-level, or sequential ISL sign labels**.
2. **Strict Rule on English Translations**:
   - Treating English words like `"I"`, `"am"`, `"going"` as visual sign labels is scientifically invalid because ISL grammar, syntax, and morphology differ fundamentally from spoken English.
   - Fabricating glosses using heuristics, LLMs, or word tokenizers is **strictly forbidden**.
3. **Phase 4 Track Formalization**:
   - **Track B (No Valid Sequential Labels)** is active.
   - Continuous temporal modeling, sliding windowing, and heuristic candidate boundary analysis will proceed.
   - Real-data CTC training is recorded as:
     > `CTC training blocked: valid sequential sign targets unavailable.`
