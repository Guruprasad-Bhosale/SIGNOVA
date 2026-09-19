# SIGNOVA Deaf-ISL Human Annotation Protocol

**Version**: 1.0  
**Target Domain**: Continuous Indian Sign Language (ISL) $\to$ English  
**Status**: Ready for Human Ingestion (Phase 10)  

---

## 1. Protocol Purpose & Scope

The purpose of this protocol is to define a standardized linguistic annotation methodology for native Deaf Indian Sign Language signers, linguistic researchers, and trained annotators. It provides a formal schema to convert continuous ISL video into structured, machine-readable ordered gloss sequences and paired English translations.

---

## 2. Linguistic Annotation Unit

- **Unit**: Continuous sentence-level video segment or communicative turn.
- **Boundaries**: From the initial preparation movement (hands rising from resting position) through sign execution to return to rest position.

---

## 3. Gloss Naming & Orthographic Conventions

All glosses must represent the lexical sign produced, preserving ISL grammatical order (typically Subject-Object-Verb / Topic-Comment).

| Category | Convention | Example | Notes |
| :--- | :--- | :--- | :--- |
| **Standard Lexical Sign** | UPPERCASE ASCII | `SCHOOL`, `TODAY`, `EAT` | Base lemma in English transliteration |
| **Compound Sign** | Hyphen-separated | `ICE-CREAM`, `FATHER-MOTHER` (Parents) | Single conceptual sign formed by composition |
| **Fingerspelling** | Prefix `FS-` | `FS-DELHI`, `FS-GURU`, `FS-AI` | Used for names, acronyms, or unlexicalized terms |
| **Numbers & Quantifiers** | Prefix `NUM-` | `NUM-5`, `NUM-2024`, `NUM-100` | Cardinal or ordinal numerical signs |
| **Repeated / Plural Sign** | Suffix `++` | `WALK++`, `HOUSE++` | Denotes aspectual repetition or plurality |
| **Classifier Predicate** | Prefix `CL-<handshape>-` | `CL-V-WALK`, `CL-B-FLAT-SURFACE` | Spatial-predicate classifier representations |
| **Two-Handed Coordination** | Dominant / Non-Dominant `::` | `WRITE::PAPER` | Asymmetric two-handed productions |
| **Uncertain Sign** | `UNCERTAIN` | `UNCERTAIN` | Ambiguous sign requiring reviewer adjudication |
| **Unknown Sign** | `UNKNOWN` | `UNKNOWN` | Novel or uncataloged regional sign variation |

---

## 4. Temporal Alignment Tiers

1. **Sequence-Level Gloss (Tier 1 - CTC Minimum)**:
   - Ordered list of gloss tokens for the full video segment: e.g. `["I", "COLLEGE", "GO", "TODAY"]`.
   - Frame-level timestamps for individual signs are optional for CTC training.
2. **Segment-Level Gloss (Tier 2 - Recommended)**:
   - Start timestamp ($t_{\text{start}}$ in ms) and end timestamp ($t_{\text{end}}$ in ms) for each distinct sign gloss.
3. **Translation Tier (Tier 3 - NMT Supervision)**:
   - Natural English sentence translation: e.g., *"I am going to college today."*

---

## 5. Annotation Confidence Tiers

Every annotation record must include a confidence tag:
- **`HIGH`**: Unambiguous, clear handshape, location, movement, and orientation by native signer.
- **`MEDIUM`**: Minor motion blur or rapid transition, but lexical identity is clear from context.
- **`LOW`**: Partial occlusion, extreme co-articulation, or dialectal variation.
- **`UNCERTAIN`**: Ambiguity that cannot be resolved without secondary reviewer review.

---

## 6. Review & Adjudication Workflow

$$\text{Deaf Annotator 1} \longrightarrow \text{Deaf Annotator 2 (Double-Annotated Subset)} \longrightarrow \text{Adjudication / Lead Linguist Review}$$

1. **Primary Annotation**: First annotator produces temporal segments, glosses, and translation.
2. **Blind Dual-Annotation Subset**: A minimum 10% subset of videos is independently annotated by a second annotator.
3. **Adjudication**: Where annotator 1 and annotator 2 disagree, a lead Deaf ISL specialist reviews both annotations and determines the final ground truth.

---

## 7. Inter-Annotator Agreement Policy

> [!IMPORTANT]
> **Measurement Policy**:
> Inter-annotator agreement metrics (Token-level TER, Sequence Edit Distance, Cohen's Kappa) are calculated **only when multiple independent human annotations actually exist**. Synthetic or fabricated agreement metrics are strictly prohibited.
