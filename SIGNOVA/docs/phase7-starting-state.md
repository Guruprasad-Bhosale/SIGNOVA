# Phase 7 Starting-State Audit: Translation Supervision & Readiness

## 1. Executive Summary & Hard Data Gate Classification

**Date**: 2026-09-18  
**Gate Status**: **STATE C — REAL SEQUENTIAL SIGN $\rightarrow$ ENGLISH SUPERVISION BLOCKED**  

> [!CAUTION]
> **CRITICAL SCIENTIFIC GUARDRAIL**:
> **Phase 7 does not establish real ISL $\rightarrow$ English translation capability unless legitimate gloss $\rightarrow$ English paired supervision becomes available.**
> Phase 7 establishes the translation subsystem, data adapters, neural sequence-to-sequence architecture, decoding algorithms, diagnostics, and pipeline contracts. Because local datasets lack verified ordered ISL gloss annotations, Phase 7 operates under **STATE C** (Hard Data Gate: No Real Sequential Sign $\rightarrow$ English Pairs Available). All neural Seq2Seq experiments in Phase 7 are conducted on isolated synthetic fixtures strictly for implementation verification.

---

## 2. Dataset Inventory & Semantic Audit Findings

| Dataset Source | Formats / Fields Available | Source Representation | Target Representation | Genuine Ordered ISL Glosses? | Signer / Session Split? | Supervision State |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ISLTranslate` (`ISLTranslate.csv`) | `uid, text` (31,225 rows) | English sentences / phrases | English sentences | **NO** (Only English sentences) | UID prefix exists, but no verified metadata | **STATE C** |
| `ISLTranslate` (`ISL-signer_validation.csv`) | `uid, Transcribed Text, Gold Translation` (292 rows) | English textbook sentences | Signer-translated English text | **NO** (English text $\rightarrow$ English text) | None | **STATE C** |
| `INCLUDE` (`include.csv` / Zenodo) | `Word, Signer, Video_Path` (4,287 clips) | Isolated signs / single words | Single English word | **NO** (Isolated vocabulary, not sequential glosses) | Signer IDs 1–7 | **STATE C** |
| `isl-translator-main` (External Reference) | ST-GCN + Transformer + IndicTrans2 wrapper | MediaPipe keypoints $\rightarrow$ Isolated classes | Hindi / English text via IndicTrans2 | **NO** (No native gloss $\rightarrow$ text paired corpus) | Synthetic / isolated split | **STATE C** |

---

## 3. Supervision Blocker Analysis

1. **No Ordered Gloss Annotations**:
   - `ISLTranslate` contains natural language English sentences corresponding to educational videos. It does **not** contain discrete ISL gloss annotations (e.g. `["BOOK", "OPEN", "PAGE", "111"]`).
2. **Strict Guardrails Enforced**:
   - **No Gloss Fabrication**: We strictly refuse to tokenize or lemmatize English sentences into pseudo-glosses. ISL has its own distinct syntax (often Topic-Comment or SOV) and non-manual markers that differ fundamentally from English SVO syntax.
   - **No LLM Substitution**: We do not query an LLM to generate pseudo-ISL gloss sequences.
   - **No English-to-Sign Reverse Engineering**: English sentences are never used to fake sign gloss supervision.
3. **Execution Path for Phase 7**:
   - Establish the full neural Seq2Seq translation architecture, decoders, vocabulary abstractions, normalizers, evaluation metrics, leakage detection, and API wrappers.
   - Validate implementation correctness (forward/backward passes, masking, teacher forcing vs autoregressive decoding, loss convergence, save/load) on isolated synthetic fixtures.
   - Mark all synthetic experiment outputs as `SYNTHETIC FIXTURE — NOT REAL ISL TRANSLATION PERFORMANCE`.

---

## 4. Architectural Boundaries

- $\text{Recognition} \neq \text{Translation}$
- The translation subsystem accepts strictly a `List[str]` of recognized gloss tokens and outputs an English `str`.
- The translation module never accesses video frames or landmark coordinates.
