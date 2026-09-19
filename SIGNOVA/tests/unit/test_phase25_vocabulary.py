"""
tests/unit/test_phase25_vocabulary.py
Unit tests verifying vocabulary validation and unknown token handling in Phase 25.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from signova.operations.phase25_orchestrator import (
    ANNOTATION_SOURCE_HUMAN_DIRECT,
    HumanAnnotationRecord,
    Phase25Orchestrator,
)


class TestPhase25Vocabulary:
    """Tests verifying canonical vocabulary membership and unknown gloss routing."""

    def test_unknown_gloss_rejected(self, tmp_path: Path) -> None:
        """Verify annotations containing unknown/unregistered glosses are rejected."""
        orchestrator = Phase25Orchestrator(data_root=tmp_path / "data")
        record = HumanAnnotationRecord(
            annotation_id="ann_unknown_vocab",
            source_video_id="vid_001",
            source_sha256="A" * 64,
            annotator_id="annotator_real",
            annotation_source=ANNOTATION_SOURCE_HUMAN_DIRECT,
            ordered_glosses=["NON_EXISTENT_GLOSS_XYZ_999"],
        )
        res = orchestrator.intake_annotation(record)
        assert res["status"] == "REJECTED"
        assert "UNKNOWN_GLOSSES" in res["reason"]

    def test_known_gloss_accepted(self, tmp_path: Path) -> None:
        """Verify annotations containing known vocabulary tokens are accepted."""
        orchestrator = Phase25Orchestrator(data_root=tmp_path / "data")
        record = HumanAnnotationRecord(
            annotation_id="ann_known_vocab",
            source_video_id="vid_001",
            source_sha256="A" * 64,
            annotator_id="annotator_real",
            annotation_source=ANNOTATION_SOURCE_HUMAN_DIRECT,
            ordered_glosses=["<blank>", "<unk>"],
        )
        res = orchestrator.intake_annotation(record)
        assert res["status"] == "ACCEPTED"
