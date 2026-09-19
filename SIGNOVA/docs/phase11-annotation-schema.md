# SIGNOVA Phase 11 — Canonical Human Annotation Schema Specification

**Version**: 1.0  
**Format**: Canonical JSON (with optional EAF export helper)  

---

## 1. Schema Architecture & Data Model

The canonical schema represents both **sequence-only** annotations and **temporally-aligned** annotations without conflating the two.

### Canonical VideoAnnotation JSON Specification

```json
{
  "annotation_id": "annot_1782bea75c7d-1_annotator_01",
  "sample_id": "1782bea75c7d-1",
  "annotator_id": "annotator_deaf_01",
  "is_temporally_aligned": true,
  "glosses": [
    "I",
    "GO",
    "COLLEGE",
    "TODAY"
  ],
  "segments": [
    {
      "gloss": "I",
      "start_frame": 12,
      "end_frame": 35,
      "start_time_ms": 400.0,
      "end_time_ms": 1166.7,
      "confidence": "HIGH",
      "notes": ""
    },
    {
      "gloss": "GO",
      "start_frame": 36,
      "end_frame": 60,
      "start_time_ms": 1200.0,
      "end_time_ms": 2000.0,
      "confidence": "HIGH",
      "notes": ""
    }
  ],
  "english_translation": "I am going to college today.",
  "review_status": "VERIFIED",
  "quality_grade": "LINGUIST_REVIEWED",
  "reviewer_id": "reviewer_linguist_01",
  "reviewer_notes": "Validated against Deaf native signing standards.",
  "dataset_split": "train",
  "training_eligible": true,
  "provenance_id": "prov_1782bea75c7d-1_v01",
  "version": "0.1.0",
  "created_at": "2026-09-18T23:56:00",
  "updated_at": "2026-09-18T23:56:00",
  "metadata": {}
}
```

---

## 2. Sequence-Only vs Temporally-Aligned Support

- **`is_temporally_aligned = false`**: `segments` is empty `[]`. Gloss sequence is ordered at the video level (sufficient for CTC sequence training).
- **`is_temporally_aligned = true`**: `segments` contains explicit start/end frame boundaries and millisecond timestamps for each sign.
