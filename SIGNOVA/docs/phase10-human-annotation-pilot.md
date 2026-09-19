# SIGNOVA Phase 10 — Human Annotation Pilot Protocol & Implementation Guide

**Version**: 2.0  
**Supervision Gate**: **STATE C**  
**Pilot Objective**: Establish a reproducible, Deaf-led ISL annotation workflow with zero synthetic data fabrication.

---

## 1. Pilot Sample Selection & Reproducibility

- **Sampling Method**: Deterministic seeded random sampling (`seed = 42`) across diagnostic video segments.
- **Stratification**: Includes simple lexical signs, compound signs, fingerspelling, numbers, classifiers, and low-confidence/complex boundary segments.
- **Population**: Recorded in `outputs/reports/phase10_annotation_sample.json`.

---

## 2. Token Normalization & Orthographic Rules

All annotations adhere strictly to the Phase 9 canonical notation:
- **Simple Sign**: Uppercase lemma (`SCHOOL`, `EAT`, `TODAY`, `NAMASTE`).
- **Compound Sign**: Hyphen-separated (`ICE-CREAM`, `FATHER-MOTHER`).
- **Fingerspelling**: `FS-` prefix (`FS-DELHI`, `FS-RAHUL`, `FS-AI`).
- **Numbers**: `NUM-` prefix (`NUM-5`, `NUM-2024`, `NUM-100`).
- **Classifier Predicates**: `CL-<handshape>-` (`CL-V-WALK`, `CL-B-FLAT`).
- **Repeated Signs**: Suffix `++` (`WALK++`).
- **Two-Handed Asymmetric Signs**: `::` (`WRITE::PAPER`).

---

## 3. Human Availability & Safety Safeguard

> [!IMPORTANT]
> **Human Resource Constraint**:
> If native Deaf ISL annotators or linguists are unavailable during Phase 10 execution:
> - `annotation_pilot_status = "BLOCKED_HUMAN_RESOURCE"`
> - Synthetic annotations are **NEVER** created.
> - Model-generated pseudo-labels are **NEVER** treated as human annotations.
> - Inter-annotator agreement returns `NOT_COMPUTABLE` or `INSUFFICIENT_DATA`.

---

## 4. Disagreement Taxonomy & Adjudication

When double annotation is performed (`Annotator_A` and `Annotator_B`), disagreements are categorized under:
- `TOKENIZATION`: Different segmentation of continuous sign flow into distinct tokens.
- `COMPOUND_SIGN`: One annotator used hyphenated compound, other used single or separate tokens.
- `FINGERSPELLING`: Disagreement on fingerspelled prefix vs lexical loan sign.
- `NUMBER`: Disagreement on number notation vs word.
- `CLASSIFIER`: Disagreement on classifier handshape code.
- `TEMPORAL_BOUNDARY`: Disagreement on onset/offset timestamps $> 200\text{ ms}$.
- `GLOSS_SELECTION`: Synonymous English lemma selection for the same underlying ISL sign.
- `OTHER`: Regional dialect variation or camera occlusion ambiguity.

All disagreements are cataloged in `outputs/reports/phase10_annotation_disagreements.json` for expert Deaf-linguist adjudication.
