# Phase 6 Dataset Discovery Report

## 1. Overview and Scope

Continuous Sign Language Recognition (CSLR) requires **ordered sequences of sign glosses** aligned with continuous signing videos, allowing Connectionist Temporal Classification (CTC) to learn monotonic temporal alignments between visual feature representations and lexical units.

During Phase 6 Step 2, a complete audit was performed across all local data resources in `ISLTranslate-main`, `isl-translator-main`, and `SIGNOVA/data/`.

## 2. Inventory of Audited Datasets

| Repository / Source | Format | Samples | Modality | Ground Truth Annotations | Verdict |
|---|---|---|---|---|---|
| `ISLTranslate-main` | CSV + Remote URLs | 31,222 | Continuous Video | English Sentences (Translation target) | **No ISL glosses** |
| `isl-translator-main` | PyTorch / Scripts | N/A | Codebase & Demos | Isolated signs (INCLUDE) & English text | **No ISL glosses** |
| `SIGNOVA/data/` | NPZ Landmark Tensors | 72 streams | Continuous Features | English translations (Phase 1) | **No ISL glosses** |
| `INCLUDE` (Isolated) | MP4 / Landmarks | 280 (subset) | Isolated Dynamic Clips | Isolated class names (1 sign/clip) | **Not sequential** |

## 3. Findings and Analysis

1. **Absence of Token-Level ISL Glosses**:
   `ISLTranslate` contains natural language spoken English translations (e.g. *"I am going to college"*). ISL does not follow English syntax (ISL uses Subject-Object-Verb / Topic-Comment order, inflected verbs, and non-manual facial markers). Tokenizing English sentences into pseudo-glosses (e.g. `I -> AM -> GOING -> COLLEGE`) produces corrupted ground truth that trains the visual model on non-existent visual movements.
2. **Absence of Frame-Level Boundaries**:
   No local dataset provides start/end timestamps for individual sign gestures within continuous sentences.
3. **Absence of Verified Signer Metadata**:
   UID prefixes in `ISLTranslate` denote upload channels or video batches, not unique signer identities.

## 4. Hard Data Gate Classification

- **Status**: **STATE C — NO VALID SEQUENTIAL DATA LOCALLY AVAILABLE**
- **Decision**:
  - Do NOT generate a fake `phase6_sequential_manifest.csv` or fake `phase6_vocabulary.csv`.
  - Formally record the supervision blocker in `outputs/reports/phase6_supervision_blocker.json`.
  - Perform synthetic CTC pipeline validation strictly on pure synthetic fixtures.
  - Profile the 72 real continuous feature streams strictly for temporal representation, latency, and tracking stability diagnostics without fake CTC training.
