# SIGNOVA Phase 22: Genuine Human ISL Annotation Acquisition

## 1. Overview & Core Philosophy

Phase 22 establishes the authoritative human annotation acquisition infrastructure for SIGNOVA. Under the zero-tolerance scientific supervision framework:
- **Zero Synthetic Data / Zero Manufactured Labels**: No LLM-generated glosses, no pseudo-labels, no automatic transcription derivations.
- **Strict Provenance**: Every annotation record is immutably linked to its qualified annotator, source video SHA-256 hash, and full revision lineage.

---

## 2. Annotator Registration & Evidence-Based Qualification

Annotators are registered with configurable qualification metadata without collecting intrusive personal information:

```json
{
  "annotator_id": "ann_001",
  "qualification_status": "QUALIFIED",
  "qualification_method": "LINGUISTIC_EVALUATION",
  "qualification_evidence": "Certified ISL interpreter credential #ISL-2026-09",
  "verified_by": "Lead_Linguist_01",
  "verified_at": "2026-09-19T14:30:00Z"
}
```

### Qualification Invariant
The system never infers qualification from name, email, nationality, location, or unverified claims. Only annotators with `qualification_status == "QUALIFIED"` produce training-eligible annotations.

---

## 3. Non-Destructive Pilot Assignment System

Assignments are created deterministically via `scripts/create_phase22_assignments.py`:
- Existing assignments are immutable by default.
- Reassignment or regeneration requires explicit `--reassign` / `--regenerate` flags.
- Double annotation fraction defaults to 20% to support inter-annotator agreement evaluation.

---

## 4. Immutable Annotation Records & Revision Lineage

If an annotation is revised, the original record is preserved:
- `annotation_id`: Canonical sequence identifier (e.g., `annotation_001`)
- `revision_id`: Unique revision snapshot (e.g., `annotation_001_rev1`, `annotation_001_rev2`)
- `parent_revision_id`: Pointer to previous revision
- `revision_number`: Incrementing integer (1, 2, ...)

### Source Verification
Every annotation strictly enforces:
```yaml
annotation_source: HUMAN_DIRECT
```
Any record containing `LLM_GENERATED`, `PSEUDO_LABEL`, `SYNTHETIC`, `TRANSLATION_DERIVED`, or `UNKNOWN` is rejected by the schema and eligibility engine.

---

## 5. Review State Machine

Annotations transition through 6 formal lifecycle states:
1. `DRAFT`: Initial annotator work-in-progress.
2. `SUBMITTED`: Completed annotation awaiting review.
3. `REVIEW_PENDING`: In reviewer evaluation queue.
4. `VERIFIED`: Approved by qualified linguist reviewer.
5. `REJECTED`: Inaccurate boundaries, incorrect sign, or video corruption.
6. `REVISION_REQUIRED`: Sent back to annotator with structured feedback.
