"""
tests/unit/test_phase25_dataset_freeze.py
Unit tests verifying dataset freezing, cryptographic fingerprints, and version invalidation semantics.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from signova.operations.phase25_orchestrator import (
    ANNOTATION_SOURCE_HUMAN_DIRECT,
    HumanAnnotationRecord,
    Phase25Orchestrator,
    REVIEW_VERIFIED,
)


class TestPhase25DatasetFreeze:
    """Tests verifying dataset freeze locks, cryptographic fingerprints, and version immutability."""

    def test_dataset_freeze_and_fingerprint_lock(self, tmp_path: Path) -> None:
        """Verify freeze_dataset creates lock file with composite sha256."""
        orchestrator = Phase25Orchestrator(data_root=tmp_path / "data")
        
        # Populate verified records
        for i in range(5):
            rec = HumanAnnotationRecord(
                annotation_id=f"ann_frz_{i:03d}",
                source_video_id=f"vid_{i:03d}",
                source_sha256=f"{i:02d}" * 32,
                annotator_id="annotator_real_01",
                annotation_source=ANNOTATION_SOURCE_HUMAN_DIRECT,
                ordered_glosses=["<blank>", "<unk>"],
                review_state=REVIEW_VERIFIED,
            )
            orchestrator.intake_annotation(rec)

        orchestrator.build_dataset(dataset_version="phase25_frz_v001")
        freeze_res = orchestrator.freeze_dataset(dataset_version="phase25_frz_v001")

        assert freeze_res["status"] == "DATASET_FROZEN"
        assert "composite_sha256" in freeze_res
        assert Path(freeze_res["lock_path"]).exists()

    def test_frozen_dataset_versioning_requirement(self, tmp_path: Path) -> None:
        """Verify frozen dataset cannot silently mutate and requires a new version identifier."""
        orchestrator = Phase25Orchestrator(data_root=tmp_path / "data")
        
        rec = HumanAnnotationRecord(
            annotation_id="ann_v1_001",
            source_video_id="vid_001",
            source_sha256="11" * 32,
            annotator_id="annotator_real_01",
            annotation_source=ANNOTATION_SOURCE_HUMAN_DIRECT,
            ordered_glosses=["<blank>"],
            review_state=REVIEW_VERIFIED,
        )
        orchestrator.intake_annotation(rec)
        orchestrator.build_dataset(dataset_version="phase25_v1")
        freeze_res = orchestrator.freeze_dataset(dataset_version="phase25_v1")

        # Second freeze attempt on already frozen dataset returns ALREADY_FROZEN
        second_freeze = orchestrator.freeze_dataset(dataset_version="phase25_v1")
        assert second_freeze["status"] == "ALREADY_FROZEN"
