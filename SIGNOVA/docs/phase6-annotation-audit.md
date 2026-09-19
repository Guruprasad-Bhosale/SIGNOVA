# Phase 6 Annotation Semantics Audit

## 1. Purpose

The goal of this audit is to rigorously examine the semantic meaning of every metadata and annotation field present in the accessible datasets, ensuring that no target is mistakenly used for Connectionist Temporal Classification (CTC) sign recognition without verified visual provenance.

## 2. Field-by-Field Semantic Breakdown

### A. ISLTranslate (`data/ISLTranslate.csv`)

| Field Name | Type | Semantic Content | Usable for CTC Sign Recognition? | Scientific Rationale |
|---|---|---|---|---|
| `uid` | String | Clip / Batch Identifier | **No** | Identifier string (e.g. `youtube_123_4`); does not represent a visual sign or signer ID. |
| `text` | String | Spoken English Translation | **No** | Natural English sentence translation. ISL grammar differs from English grammar; using English words as CTC tokens forces visual alignment to phantom words (e.g., auxiliary verbs "am", "is", "the" which do not exist in ISL). |

### B. ISL Signer Validation (`data/ISL-signer_validation.csv`)

| Field Name | Type | Semantic Content | Usable for CTC Sign Recognition? | Scientific Rationale |
|---|---|---|---|---|
| `uid` | String | Sample Identifier | **No** | Reference index. |
| `Transcribed Text` | String | English text | **No** | Translation text. |
| `Gold Translation` | String | Signer-validated English text | **No** | Validated English translation of the video; not an ordered visual gloss sequence. |

### C. Isolated Sign Corpus (`INCLUDE`)

| Field Name | Type | Semantic Content | Usable for CTC Sign Recognition? | Scientific Rationale |
|---|---|---|---|---|
| `Word` / `ClassName` | String | Isolated sign concept | **No (for CSLR)** | Represents a single isolated sign gesture per video ($T_{\text{target}}=1$). Cannot be used for continuous multi-sign transition or coarticulation modeling. |

## 3. Supervision Blocker Conclusion

Under no circumstances should English sentences be mapped or tokenized into sign sequences for CTC training. The model must remain mathematically and experimentally honest: CTC readiness is verified through synthetic sequence tests, while continuous ISL features are analyzed for representation quality.
