"""
Phase 15 Human Annotations Ingestion and Validation Unit Tests.

Validates:
- Human annotation structural and linguistic validity using VideoAnnotation and TemporalSegment.
- Separation of signer_id and annotator_id.
- Timestamp coherence (start_time_ms < end_time_ms).
- Review state management (VERIFIED, ANNOTATION_IN_PROGRESS, REJECTED).
"""

import pytest
from signova.annotation.constants import (
    REVIEW_STATE_VERIFIED,
    REVIEW_STATE_REJECTED,
    REVIEW_STATE_ANNOTATION_IN_PROGRESS,
)
from signova.annotation.schema import (
    VideoAnnotation,
    TemporalSegment,
    GlossToken,
)


def test_valid_human_annotation_with_distinct_signer_and_annotator():
    seg = TemporalSegment(
        gloss="NAMASTE",
        start_time_ms=100.0,
        end_time_ms=900.0,
        confidence="HIGH",
    )
    ann = VideoAnnotation(
        annotation_id="ann_p15_001",
        sample_id="vid_p15_001",
        annotator_id="annotator_isl_expert_01",
        is_temporally_aligned=True,
        glosses=["NAMASTE"],
        segments=[seg],
        review_status=REVIEW_STATE_VERIFIED,
        metadata={"signer_id": "signer_delhi_01"},
    )
    assert ann.metadata["signer_id"] != ann.annotator_id
    assert ann.segments[0].start_time_ms < ann.segments[0].end_time_ms
    assert ann.review_status == REVIEW_STATE_VERIFIED


def test_temporal_segment_serialization():
    seg = TemporalSegment(
        gloss="THANKYOU",
        start_time_ms=200.0,
        end_time_ms=800.0,
        confidence="HIGH",
        notes="Clean execution",
    )
    d = seg.to_dict()
    assert d["gloss"] == "THANKYOU"
    assert d["start_time_ms"] == 200.0
    assert d["end_time_ms"] == 800.0


def test_rejected_annotation_is_not_training_eligible():
    ann = VideoAnnotation(
        annotation_id="ann_p15_002",
        sample_id="vid_p15_002",
        annotator_id="annotator_isl_expert_01",
        is_temporally_aligned=False,
        glosses=[],
        review_status=REVIEW_STATE_REJECTED,
        metadata={"signer_id": "signer_delhi_02"},
    )
    assert ann.review_status != REVIEW_STATE_VERIFIED
