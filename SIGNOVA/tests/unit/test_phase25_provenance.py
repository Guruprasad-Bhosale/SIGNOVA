"""
tests/unit/test_phase25_provenance.py
Unit tests verifying source video integrity, post-verification mutation detection, and invalidation.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import pytest

from signova.operations.phase25_orchestrator import (
    ANNOTATION_SOURCE_HUMAN_DIRECT,
    HumanAnnotationRecord,
    Phase25Orchestrator,
    REVIEW_VERIFIED,
)


class TestPhase25Provenance:
    """Tests verifying complete provenance and post-verification mutation detection chain."""

    def test_source_video_integrity_verification(self, tmp_path: Path) -> None:
        """Verify SHA-256 hash matches genuine video file bytes."""
        orchestrator = Phase25Orchestrator(data_root=tmp_path / "data")
        video_file = tmp_path / "sample_video.mp4"
        video_bytes = b"sample_video_content_bytes_12345"
        video_file.write_bytes(video_bytes)
        expected_sha = hashlib.sha256(video_bytes).hexdigest().upper()

        check = orchestrator.verify_source_video_integrity(video_file, expected_sha)
        assert check["source_exists"]
        assert check["hash_matches"]
        assert check["status"] == "PASSED"

    def test_post_verification_mutation_invalidates_annotation(self, tmp_path: Path) -> None:
        """Verify that if source video mutates after verification, the annotation is invalidated."""
        orchestrator = Phase25Orchestrator(data_root=tmp_path / "data")
        video_file = tmp_path / "video_mutate.mp4"
        initial_bytes = b"original_unmodified_content"
        video_file.write_bytes(initial_bytes)
        orig_sha = hashlib.sha256(initial_bytes).hexdigest().upper()

        record = HumanAnnotationRecord(
            annotation_id="ann_verified_to_mutate",
            source_video_id="vid_mutate",
            source_sha256=orig_sha,
            annotator_id="annotator_real",
            annotation_source=ANNOTATION_SOURCE_HUMAN_DIRECT,
            ordered_glosses=["<blank>"],
            review_state=REVIEW_VERIFIED,
        )
        orchestrator.intake_annotation(record)

        # Mutate the source video
        video_file.write_bytes(b"corrupted_or_replaced_content")

        # Invalidation check
        inv_res = orchestrator.invalidate_annotation_on_source_mutation(
            annotation_id="ann_verified_to_mutate",
            video_path=video_file,
        )

        assert inv_res["status"] == "INVALIDATED"
        assert not inv_res["training_eligible"]
        assert "SOURCE_MUTATION_DETECTED" in inv_res["reason"]
