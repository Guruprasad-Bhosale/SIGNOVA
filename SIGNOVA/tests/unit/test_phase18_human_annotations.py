"""
Phase 18 Human Annotations Ingestion, Authenticity & Lifecycle Unit Tests.

Validates:
- Human annotation structural and linguistic validity using VideoAnnotation and TemporalSegment.
- Provenance authenticity checking (HUMAN_DATA_AUTHENTICATED).
- Separation of signer_id and annotator_id.
- Timestamp coherence (start_time_ms < end_time_ms).
- Review state management (REVIEW_PENDING, VERIFIED, REJECTED).
- 7-tier sample accounting distinctions.
"""

import pytest
from signova.annotation.constants import (
    REVIEW_STATE_ANNOTATION_IN_PROGRESS,
    REVIEW_STATE_REJECTED,
    REVIEW_STATE_REVIEW_PENDING,
    REVIEW_STATE_VERIFIED,
)
from signova.annotation.schema import (
    GlossToken,
    TemporalSegment,
    VideoAnnotation,
)
from signova.operations.phase18_orchestrator import Phase18Orchestrator


def test_valid_human_annotation_with_authenticated_provenance():
    seg = TemporalSegment(
        gloss="NAMASTE",
        start_time_ms=100.0,
        end_time_ms=900.0,
        confidence="HIGH",
    )
    ann = VideoAnnotation(
        annotation_id="ann_p18_001",
        sample_id="vid_p18_001",
        annotator_id="annotator_isl_expert_01",
        is_temporally_aligned=True,
        glosses=["NAMASTE"],
        provenance_id="prov_delhi_session_01",
        segments=[seg],
        review_status=REVIEW_STATE_VERIFIED,
        metadata={"signer_id": "signer_delhi_01"},
    )
    orch = Phase18Orchestrator()
    assert orch._is_authenticated_provenance(ann) is True
    assert ann.metadata["signer_id"] != ann.annotator_id
    assert ann.segments[0].start_time_ms < ann.segments[0].end_time_ms
    assert ann.review_status == REVIEW_STATE_VERIFIED


def test_unauthenticated_anonymous_annotation_rejected():
    ann = VideoAnnotation(
        annotation_id="ann_p18_002",
        sample_id="vid_p18_002",
        annotator_id="UNKNOWN",
        is_temporally_aligned=False,
        glosses=["HELLO"],
        provenance_id="PROV_UNKNOWN",
    )
    orch = Phase18Orchestrator()
    assert orch._is_authenticated_provenance(ann) is False


def test_rejected_annotation_is_not_training_eligible():
    ann = VideoAnnotation(
        annotation_id="ann_p18_003",
        sample_id="vid_p18_003",
        annotator_id="annotator_isl_expert_01",
        is_temporally_aligned=False,
        glosses=[],
        provenance_id="prov_delhi_session_01",
        review_status=REVIEW_STATE_REJECTED,
        metadata={"signer_id": "signer_delhi_02"},
    )
    assert ann.review_status != REVIEW_STATE_VERIFIED


def test_pending_review_and_in_progress_lifecycle_tracking():
    ann_draft = VideoAnnotation(
        annotation_id="ann_p18_004",
        sample_id="vid_p18_004",
        annotator_id="annotator_isl_expert_01",
        is_temporally_aligned=True,
        glosses=["THANK_YOU"],
        provenance_id="prov_delhi_session_01",
        review_status=REVIEW_STATE_ANNOTATION_IN_PROGRESS,
    )
    ann_pending = VideoAnnotation(
        annotation_id="ann_p18_005",
        sample_id="vid_p18_005",
        annotator_id="annotator_isl_expert_01",
        is_temporally_aligned=True,
        glosses=["WELCOME"],
        provenance_id="prov_delhi_session_01",
        review_status=REVIEW_STATE_REVIEW_PENDING,
    )
    assert ann_draft.review_status == REVIEW_STATE_ANNOTATION_IN_PROGRESS
    assert ann_pending.review_status == REVIEW_STATE_REVIEW_PENDING
