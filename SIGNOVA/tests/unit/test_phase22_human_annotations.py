"""
Phase 22 Human Annotation Import, Provenance & Validation Tests.
"""

import pytest
from signova.operations.phase22_orchestrator import HumanAnnotationRecord, Phase22Orchestrator


def test_annotation_source_strict_rejection(tmp_path):
    orch = Phase22Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")

    # Invalid source should fail validation
    with pytest.raises(ValueError, match="Invalid annotation_source"):
        HumanAnnotationRecord(
            annotation_id="ann_rec_1",
            video_id="vid_001",
            annotator_id="ann_01",
            gloss_sequence=["HELLO", "WORLD"],
            start_frame=0,
            end_frame=100,
            annotation_source="LLM_GENERATED"
        )


def test_annotation_import_and_revision_lineage(tmp_path):
    orch = Phase22Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.register_annotator("ann_01", qualification_status="QUALIFIED")

    # Import revision 1
    rec1 = orch.import_human_annotation(
        annotation_id="ann_rec_001",
        video_id="vid_001",
        annotator_id="ann_01",
        gloss_sequence=["HELLO", "INDIA"],
        start_frame=0,
        end_frame=60,
        source_sha256="abc1234567890",
        source_path="/raw/isl_001.mp4",
        fps=30.0,
        file_size=102400,
        duration=2.0,
        frame_count=60
    )
    assert rec1.revision_number == 1
    assert rec1.review_status == "SUBMITTED"
    assert rec1.annotation_source == "HUMAN_DIRECT"

    # Reject revision 1
    rev1_status = orch.review_annotation("ann_rec_001", "REJECTED", reviewer_id="Lead_01", rejection_reason="Misaligned boundary")
    assert rev1_status["review_status"] == "REJECTED"

    # Submit revision 2 referencing parent
    rec2 = orch.import_human_annotation(
        annotation_id="ann_rec_001",
        video_id="vid_001",
        annotator_id="ann_01",
        gloss_sequence=["HELLO", "NAMASTE"],
        start_frame=0,
        end_frame=60,
        source_sha256="abc1234567890",
        parent_revision_id="ann_rec_001_rev1"
    )
    assert rec2.revision_number == 2
    assert rec2.parent_revision_id == "ann_rec_001_rev1"


def test_source_video_hash_mismatch_invalidation(tmp_path):
    orch = Phase22Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.register_annotator("ann_01", qualification_status="QUALIFIED")

    # Import with hash1
    rec = orch.import_human_annotation(
        annotation_id="ann_rec_002",
        video_id="vid_002",
        annotator_id="ann_01",
        gloss_sequence=["THANK", "YOU"],
        source_sha256="hash_version_1"
    )
    orch.review_annotation("ann_rec_002", "VERIFIED", reviewer_id="Lead_01")

    # Validate against expected different hash -> SOURCE_CHANGED
    val = orch.validate_annotation("ann_rec_002", expected_video_sha256="hash_version_2")
    assert val.get("valid") is False
    assert val.get("source_status") == "SOURCE_CHANGED"
