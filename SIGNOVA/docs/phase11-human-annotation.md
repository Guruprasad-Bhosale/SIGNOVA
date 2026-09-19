# SIGNOVA Phase 11 — Human Annotation Platform & Strategy Guide

**Supervision State**: **STATE_B (ANNOTATION INFRASTRUCTURE OPERATIONAL / REAL CTC BLOCKED)**  
**Platform Status**: `ANNOTATION_PLATFORM = OPERATIONAL`  
**Human Data Collection**: `HUMAN_DATA_COLLECTION = NOT_STARTED`  
**Pilot Status**: `PILOT_STATUS = BLOCKED_HUMAN_RESOURCE`  

---

## 1. Platform Purpose & Research Scope

Phase 11 establishes a specialized, zero-cost, offline human annotation infrastructure (`apps/annotation/`) designed specifically for native Deaf Indian Sign Language signers and linguists.

### Key Operational Capabilities
1. **Interactive Video & Frame Inspection**: Inspect temporal flow, seek frame-by-frame, and evaluate sign boundaries.
2. **Explicit Session Workflow**:
   $$\mathbf{OPEN} \longrightarrow \mathbf{DRAFT} \longrightarrow \mathbf{SAVE} \longrightarrow \mathbf{RESUME} \longrightarrow \mathbf{SUBMIT} \longrightarrow \mathbf{REVIEW} \longrightarrow \mathbf{VERIFY\ /\ REJECT}$$
3. **No Automated Ground Truth**: Model predictions, English translations, and heuristic segmentations are strictly prohibited from generating ground truth. Any optional assistance must be tagged `MODEL_SUGGESTION`.
4. **Canonical Ingestion**: All human submissions are serialized in canonical JSON (`data/annotations/phase11/v0.1.0/`).

---

## 2. Orthographic & Linguistic Conventions

- **Standard Lexical Signs**: Uppercase lemma (`SCHOOL`, `TODAY`, `NAMASTE`).
- **Compound Signs**: Hyphen-separated composition (`ICE-CREAM`, `FATHER-MOTHER`).
- **Fingerspelling**: `FS-` prefix (`FS-DELHI`, `FS-RAHUL`).
- **Numbers**: `NUM-` prefix (`NUM-5`, `NUM-2024`).
- **Classifier Predicates**: `CL-` prefix (`CL-V-WALK`, `CL-B-FLAT`).
- **Two-Handed Asymmetric Signs**: `::` coordination (`WRITE::PAPER`).

---

## 3. Reviewer Workflow & Adjudication

Annotations progress through distinct review states:
- `UNANNOTATED`: Video awaiting human inspection.
- `ANNOTATION_IN_PROGRESS`: In-progress draft saved locally.
- `REVIEW_PENDING`: Submitted by annotator; awaiting linguist review.
- `VERIFIED`: Confirmed by reviewer / linguist.
- `REJECTED`: Returned to annotator with explanatory notes.
