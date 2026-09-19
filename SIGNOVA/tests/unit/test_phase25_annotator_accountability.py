"""
tests/unit/test_phase25_annotator_accountability.py
Unit tests verifying orthogonal annotator status dimensions and missing identity rejection.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from signova.operations.phase25_orchestrator import (
    ANNOTATION_SOURCE_HUMAN_DIRECT,
    IDENTITY_MISSING,
    IDENTITY_PRESENT,
    AUTH_NOT_AUTHENTICATED,
    QUAL_UNVERIFIED,
    HumanAnnotationRecord,
    Phase25Orchestrator,
)


class TestPhase25AnnotatorAccountability:
    """Tests verifying annotator identity, authentication, and qualification dimensions."""

    def test_missing_annotator_identity_rejected(self, tmp_path: Path) -> None:
        """Verify annotations without annotator identity are rejected."""
        orchestrator = Phase25Orchestrator(data_root=tmp_path / "data")
        record = HumanAnnotationRecord(
            annotation_id="ann_no_id",
            source_video_id="vid_001",
            source_sha256="A" * 64,
            annotator_id="",  # Missing!
            annotation_source=ANNOTATION_SOURCE_HUMAN_DIRECT,
            ordered_glosses=["HELLO"],
        )
        res = orchestrator.intake_annotation(record)
        assert res["status"] == "REJECTED"
        assert res["reason"] == "MISSING_ANNOTATOR_IDENTITY"

    def test_orthogonal_annotator_status_dimensions(self, tmp_path: Path) -> None:
        """Verify identity, authentication, and qualification are reported as distinct orthogonal facts."""
        orchestrator = Phase25Orchestrator(data_root=tmp_path / "data")
        
        # Test unverified present annotator
        status = orchestrator.evaluate_annotator_status("ann_unverified_user")
        assert status["annotator_identity"] == IDENTITY_PRESENT
        assert status["annotator_authentication"] == AUTH_NOT_AUTHENTICATED
        assert status["annotator_qualification"] == QUAL_UNVERIFIED
        
        # Test empty annotator
        status_empty = orchestrator.evaluate_annotator_status(None)
        assert status_empty["annotator_identity"] == IDENTITY_MISSING
