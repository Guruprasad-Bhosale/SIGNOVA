"""
tests/unit/test_phase25_double_annotation.py
Unit tests verifying double annotation requirements and non-computable agreement reporting.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from signova.operations.phase25_orchestrator import (
    ANNOTATION_SOURCE_HUMAN_DIRECT,
    HumanAnnotationRecord,
    Phase25Orchestrator,
)


class TestPhase25DoubleAnnotation:
    """Tests verifying independent double annotation and non-fabricated agreement calculation."""

    def test_agreement_not_computable_when_only_one_annotation_exists(self, tmp_path: Path) -> None:
        """Verify system explicitly reports AGREEMENT_NOT_COMPUTABLE when single annotation exists."""
        orchestrator = Phase25Orchestrator(data_root=tmp_path / "data")
        rec = HumanAnnotationRecord(
            annotation_id="ann_single_001",
            source_video_id="vid_001",
            source_sha256="A" * 64,
            annotator_id="annotator_1",
            annotation_source=ANNOTATION_SOURCE_HUMAN_DIRECT,
            ordered_glosses=["<blank>"],
        )
        orchestrator.intake_annotation(rec)

        res = orchestrator.evaluate_double_annotation_agreement("ann_single_001", None)
        assert res["agreement"] == "AGREEMENT_NOT_COMPUTABLE"

    def test_independent_double_annotation_agreement(self, tmp_path: Path) -> None:
        """Verify agreement is computed correctly when two distinct annotators submit."""
        orchestrator = Phase25Orchestrator(data_root=tmp_path / "data")
        rec1 = HumanAnnotationRecord(
            annotation_id="ann_pair_1",
            source_video_id="vid_001",
            source_sha256="A" * 64,
            annotator_id="annotator_1",
            annotation_source=ANNOTATION_SOURCE_HUMAN_DIRECT,
            ordered_glosses=["<blank>", "<unk>"],
            annotation_group_id="grp_001",
            independent_annotation_index=1,
        )
        rec2 = HumanAnnotationRecord(
            annotation_id="ann_pair_2",
            source_video_id="vid_001",
            source_sha256="A" * 64,
            annotator_id="annotator_2",
            annotation_source=ANNOTATION_SOURCE_HUMAN_DIRECT,
            ordered_glosses=["<blank>", "<unk>"],
            annotation_group_id="grp_001",
            independent_annotation_index=2,
        )
        orchestrator.intake_annotation(rec1)
        orchestrator.intake_annotation(rec2)

        res = orchestrator.evaluate_double_annotation_agreement("ann_pair_1", "ann_pair_2")
        assert res["status"] == "COMPUTED"
        assert res["agreement"] == "AGREEMENT"
        assert res["exact_match"]
