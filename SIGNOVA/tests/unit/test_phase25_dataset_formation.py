"""
tests/unit/test_phase25_dataset_formation.py
Unit tests verifying deterministic dataset construction from training-eligible human annotations.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from signova.operations.phase25_orchestrator import (
    ANNOTATION_SOURCE_HUMAN_DIRECT,
    HumanAnnotationRecord,
    Phase25Orchestrator,
    REVIEW_VERIFIED,
)


class TestPhase25DatasetFormation:
    """Tests verifying candidate dataset formation and partition generation."""

    def test_dataset_formation_from_verified_records(self, tmp_path: Path) -> None:
        """Verify build_dataset creates valid manifest with train/val/test splits."""
        orchestrator = Phase25Orchestrator(data_root=tmp_path / "data")
        
        # Populate 10 verified records
        for i in range(10):
            rec = HumanAnnotationRecord(
                annotation_id=f"ann_ver_{i:03d}",
                source_video_id=f"vid_{i:03d}",
                source_sha256=f"{i:02d}" * 32,
                annotator_id="annotator_real_01",
                annotation_source=ANNOTATION_SOURCE_HUMAN_DIRECT,
                ordered_glosses=["<blank>", "<unk>"],
                review_state=REVIEW_VERIFIED,
            )
            orchestrator.intake_annotation(rec)

        res = orchestrator.build_dataset(dataset_version="phase25_test_v001")
        assert res["dataset_status"] == "DATASET_READY"
        assert res["total_samples"] == 10
        assert res["train_count"] + res["val_count"] + res["test_count"] == 10
        assert res["leakage_status"] in ("PASSED", "SIGNER_INDEPENDENT_SPLIT_NOT_POSSIBLE")
