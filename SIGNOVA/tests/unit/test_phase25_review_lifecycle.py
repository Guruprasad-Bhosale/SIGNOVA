"""
tests/unit/test_phase25_review_lifecycle.py
Unit tests verifying non-destructive review state transitions and rejection auditing.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from signova.operations.phase25_orchestrator import (
    ANNOTATION_SOURCE_HUMAN_DIRECT,
    HumanAnnotationRecord,
    Phase25Orchestrator,
    REVIEW_DRAFT,
    REVIEW_REJECTED,
    REVIEW_VERIFIED,
)


class TestPhase25ReviewLifecycle:
    """Tests verifying non-destructive review, verification, and rejection states."""

    def test_review_verification_transition(self, tmp_path: Path) -> None:
        """Verify DRAFT/SUBMITTED transitions to VERIFIED upon reviewer action."""
        orchestrator = Phase25Orchestrator(data_root=tmp_path / "data")
        rec = HumanAnnotationRecord(
            annotation_id="ann_review_001",
            source_video_id="vid_001",
            source_sha256="A" * 64,
            annotator_id="annotator_1",
            annotation_source=ANNOTATION_SOURCE_HUMAN_DIRECT,
            ordered_glosses=["<blank>"],
            review_state=REVIEW_DRAFT,
        )
        orchestrator.intake_annotation(rec)

        res = orchestrator.review_annotation(
            annotation_id="ann_review_001",
            action="VERIFY",
            reviewer_id="lead_reviewer_001",
        )
        assert res["status"] == "UPDATED"
        assert res["review_state"] == REVIEW_VERIFIED
        assert res["record"]["metadata"]["verified_by"] == "lead_reviewer_001"

    def test_review_rejection_transition(self, tmp_path: Path) -> None:
        """Verify review rejection transitions cleanly and records reason."""
        orchestrator = Phase25Orchestrator(data_root=tmp_path / "data")
        rec = HumanAnnotationRecord(
            annotation_id="ann_reject_001",
            source_video_id="vid_001",
            source_sha256="A" * 64,
            annotator_id="annotator_1",
            annotation_source=ANNOTATION_SOURCE_HUMAN_DIRECT,
            ordered_glosses=["<blank>"],
        )
        orchestrator.intake_annotation(rec)

        res = orchestrator.review_annotation(
            annotation_id="ann_reject_001",
            action="REJECT",
            reviewer_id="lead_reviewer_001",
            reason="Glosses do not match visual handshapes",
        )
        assert res["status"] == "UPDATED"
        assert res["review_state"] == REVIEW_REJECTED
        assert "Glosses do not match" in res["record"]["rejection_reason"]
