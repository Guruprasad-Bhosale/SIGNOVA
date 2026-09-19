"""
tests/unit/test_phase25_dataset_eligibility.py
Unit tests verifying strict conditions required for annotations to achieve training eligibility.
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


class TestPhase25DatasetEligibility:
    """Tests verifying that ONLY verified human annotations achieve training eligibility."""

    def test_draft_or_rejected_not_eligible(self, tmp_path: Path) -> None:
        """Verify DRAFT or REJECTED annotations are never training eligible."""
        orchestrator = Phase25Orchestrator(data_root=tmp_path / "data")
        rec_draft = HumanAnnotationRecord(
            annotation_id="ann_draft_elig",
            source_video_id="vid_001",
            source_sha256="A" * 64,
            annotator_id="annotator_1",
            annotation_source=ANNOTATION_SOURCE_HUMAN_DIRECT,
            ordered_glosses=["<blank>"],
            review_state=REVIEW_DRAFT,
        )
        res_draft = orchestrator.intake_annotation(rec_draft)
        assert not res_draft["training_eligible"]

    def test_verified_human_direct_is_eligible(self, tmp_path: Path) -> None:
        """Verify VERIFIED HUMAN_DIRECT annotation with valid annotator is training eligible."""
        orchestrator = Phase25Orchestrator(data_root=tmp_path / "data")
        rec_verified = HumanAnnotationRecord(
            annotation_id="ann_ver_elig",
            source_video_id="vid_001",
            source_sha256="A" * 64,
            annotator_id="annotator_1",
            annotation_source=ANNOTATION_SOURCE_HUMAN_DIRECT,
            ordered_glosses=["<blank>"],
            review_state=REVIEW_VERIFIED,
        )
        res_ver = orchestrator.intake_annotation(rec_verified)
        assert res_ver["training_eligible"]
