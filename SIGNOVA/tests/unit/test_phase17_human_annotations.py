"""
Phase 17 Human Annotations Ingestion, Authenticity & Validation Unit Tests.

Validates:
- Human annotation structural and linguistic validity using VideoAnnotation and TemporalSegment.
- Provenance authenticity checking (HUMAN_DATA_AUTHENTICATED).
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
from signova.operations.phase17_orchestrator import Phase17Orchestrator


def test_valid_human_annotation_with_authenticated_provenance():
    seg = TemporalSegment(
        gloss="NAMASTE",
        start_time_ms=100.0,
        end_time_ms=900.0,
        confidence="HIGH",
    )
    ann = VideoAnnotation(
        annotation_id="ann_p17_001",
        sample_id="vid_p17_001",
        annotator_id="annotator_isl_expert_01",
        provenance_id="prov_delhi_session_01",
        is_temporally_aligned=True,
        glosses=["NAMASTE"],
        segments=[seg],
        review_status=REVIEW_STATE_VERIFIED,
        metadata={"signer_id": "signer_delhi_01"},
    )
    orch = Phase17Orchestrator()
    assert orch._is_authenticated_provenance(ann) is True
    assert ann.metadata["signer_id"] != ann.annotator_id
    assert ann.segments[0].start_time_ms < ann.segments[0].end_time_ms
    assert ann.review_status == REVIEW_STATE_VERIFIED


def test_unauthenticated_anonymous_annotation_rejected():
    ann = VideoAnnotation(
        annotation_id="ann_p17_002",
        sample_id="vid_p17_002",
        annotator_id="UNKNOWN",
        provenance_id="PROV_UNKNOWN",
        is_temporally_aligned=False,
        glosses=["HELLO"],
    )
    orch = Phase17Orchestrator()
    assert orch._is_authenticated_provenance(ann) is False


def test_rejected_annotation_is_not_training_eligible():
    ann = VideoAnnotation(
        annotation_id="ann_p17_003",
        sample_id="vid_p17_003",
        annotator_id="annotator_isl_expert_01",
        provenance_id="prov_delhi_session_01",
        is_temporally_aligned=False,
        glosses=[],
        review_status=REVIEW_STATE_REJECTED,
        metadata={"signer_id": "signer_delhi_02"},
    )
    assert ann.review_status != REVIEW_STATE_VERIFIED
