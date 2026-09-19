# SIGNOVA Phase 14 — Indian Sign Language (ISL) Annotation Protocol

## 1. Purpose & Guiding Principles
This protocol establishes the scientific guidelines for annotating continuous Indian Sign Language (ISL) video sequences for research in continuous sign language recognition (CSLR) and translation.

- **Deaf-Community Centered**: Sign language annotations must faithfully represent natural ISL signing syntax and lexical choices, avoiding forced mapping to English word order.
- **Evidence-Based**: No gloss sequence may be assumed, inferred, or auto-completed without visual signing evidence.
- **Uncertainty Transparency**: When handshape, orientation, or movement is ambiguous, annotators must record uncertainty rather than guessing.

---

## 2. Core Glossing Rules

### 2.1 Lexical Sign Representation
- Glosses must be written in uppercase alphanumeric tokens separated by hyphens (e.g., `NAMASTE`, `THANK-YOU`, `WATER`, `SCHOOL`).
- A gloss represents a discrete ISL lexical unit, not an English dictionary translation.

### 2.2 Sign Ordering
- Signs must be annotated strictly in the chronological order of signing execution.
- ISL grammatical order (often Topic-Comment or SOV) must be preserved as signed. Never reorganize glosses to match English syntax.

### 2.3 Repeated Signs
- If a sign is articulated repeatedly as distinct repetitions, annotate each repetition:
  - Example: A signer signing `AGAIN` twice $\rightarrow$ `["AGAIN", "AGAIN"]`.
  - In CTC sequence modeling, adjacent identical tokens require blank frame separation:
    $$T_{\text{required}} = L + \sum_{i=1}^{L-1} \mathbb{I}(y_i == y_{i+1})$$

### 2.4 Uncertain Signs
- When a sign is partially visible, occluded, or phonetically unclear:
  - Token: `UNCERTAIN_SIGN` or use the uncertainty tag `?GLOSS` (e.g., `?TEACHER`).
  - Flag confidence as `UNCERTAIN` in token metadata.

### 2.5 Non-Sign Movements, Pauses, and Transitions
- Transition movements (raising hands from rest or lowering to rest) are not glosses.
- Long rest pauses (> 1 second) should not be assigned lexical tokens. If temporal alignment is used, segment boundaries must omit non-signing rest periods.

### 2.6 Fingerspelling
- Fingerspelled words should be prefixed with `FS-` or hyphenated letter by letter:
  - Example: Fingerspelling "DELHI" $\rightarrow$ `FS-DELHI` or `D-E-L-H-I`.
  - *Decision Marker*: `REQUIRES_ISL_EXPERT_DECISION` for regional fingerspelling conventions.

### 2.7 Compound Signs
- Established compound signs formed by two fused signs are annotated with `+` or hyphenated compound (e.g., `SUN+RISE` or `PARENTS` as `MOTHER+FATHER`).
- *Decision Marker*: `REQUIRES_ISL_EXPERT_DECISION` based on ISL lexicalization norms.

### 2.8 Non-Manual Markers (Facial Gestures & Mouthing)
- Facial expressions (questions, negation, intensity) should be noted in metadata fields rather than replacing manual gloss tokens.

---

## 3. Workflow & Review Lifecycle

```
[ OPEN VIDEO ]
      ↓
[ DRAFT ANNOTATION ] (Seek frames, identify signs, mark segments)
      ↓
[ SAVE DRAFT ] (Persist in local draft storage)
      ↓
[ SUBMIT ANNOTATION ] (Review state: REVIEW_PENDING)
      ↓
[ INDEPENDENT LINGUISTIC REVIEW ]
      ↓
[ VERIFY ] or [ REJECT ] (With recorded reviewer ID, reason, and revision history)
```

- **Annotation Existence $\neq$ Training Eligibility**: An annotation file existing in the repository is not eligible for model training until verified by an independent reviewer and meeting quality gate criteria.
- **Immutable Revision History**: Reviewers must never overwrite historical annotations directly; corrections must generate a new versioned `AnnotationRevision`.

---

## 4. Participant & Metadata Separation Rules
1. `annotator_id` must strictly identify the person performing the annotation.
2. `reviewer_id` must identify the reviewer and must never equal `annotator_id`.
3. `signer_id` must identify the person appearing in the video. It must never be copied from the annotator ID or inferred from random filename prefixes.
4. `session_id` must represent the verified recording session.
5. All unknown identity fields must remain `"UNKNOWN"`.
