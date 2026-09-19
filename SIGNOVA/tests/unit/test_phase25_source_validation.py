"""
tests/unit/test_phase25_source_validation.py
Unit tests verifying strict acceptance of HUMAN_DIRECT and rejection of all forbidden sources.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from signova.operations.phase25_orchestrator import (
    ANNOTATION_SOURCE_HUMAN_DIRECT,
    FORBIDDEN_SOURCES,
    HumanAnnotationRecord,
    Phase25Orchestrator,
)


class TestPhase25SourceValidation:
    """Tests verifying that ONLY HUMAN_DIRECT annotations are accepted."""

    def test_human_direct_accepted(self, tmp_path: Path) -> None:
        """Verify HUMAN_DIRECT annotation with valid annotator and glosses is accepted."""
        orchestrator = Phase25Orchestrator(data_root=tmp_path / "data")
        record = HumanAnnotationRecord(
            annotation_id="ann_valid_001",
            source_video_id="vid_001",
            source_sha256="A" * 64,
            annotator_id="annotator_real_001",
            annotation_source=ANNOTATION_SOURCE_HUMAN_DIRECT,
            ordered_glosses=["<blank>", "<unk>"],
        )
        res = orchestrator.intake_annotation(record)
        assert res["status"] == "ACCEPTED"

    @pytest.mark.parametrize("forbidden_source", sorted(list(FORBIDDEN_SOURCES)))
    def test_forbidden_sources_strictly_rejected(self, tmp_path: Path, forbidden_source: str) -> None:
        """Verify LLM, pseudo-label, synthetic, translation-derived, and unknown sources are rejected."""
        orchestrator = Phase25Orchestrator(data_root=tmp_path / "data")
        record = HumanAnnotationRecord(
            annotation_id=f"ann_forbidden_{forbidden_source}",
            source_video_id="vid_001",
            source_sha256="A" * 64,
            annotator_id="annotator_001",
            annotation_source=forbidden_source,
            ordered_glosses=["HELLO"],
        )
        res = orchestrator.intake_annotation(record)
        assert res["status"] == "REJECTED"
        assert "FORBIDDEN_SOURCE" in res["reason"]
        assert not res["training_eligible"]
