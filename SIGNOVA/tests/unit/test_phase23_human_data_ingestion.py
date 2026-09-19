"""
Phase 23 Batch Human Data Ingestion Tests.
"""

import json
from signova.operations.phase23_orchestrator import Phase23Orchestrator


def test_batch_human_data_ingestion_valid(tmp_path):
    orch = Phase23Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.phase22_orch.register_annotator("ann_isl_lead_01", qualification_status="QUALIFIED")

    # Create dummy raw video and compute sha
    raw_video = tmp_path / "videos" / "isl_001.mp4"
    raw_video.parent.mkdir(parents=True, exist_ok=True)
    raw_video.write_bytes(b"dummy_video_bytes_001")
    import hashlib
    sha = hashlib.sha256(b"dummy_video_bytes_001").hexdigest().upper()

    # Create batch annotation file
    batch_dir = tmp_path / "incoming_annotations"
    batch_dir.mkdir(parents=True, exist_ok=True)

    ann1 = {
        "annotation_id": "ann_b_001",
        "video_id": "isl_001.mp4",
        "annotator_id": "ann_isl_lead_01",
        "gloss_tokens": ["HELLO", "NAMASTE"],
        "annotation_source": "HUMAN_DIRECT",
        "source_sha256": sha,
        "source_uri": str(raw_video),
        "review_state": "VERIFIED",
    }
    (batch_dir / "ann_001.json").write_text(json.dumps(ann1), encoding="utf-8")

    res = orch.ingest_batch_human_data(source_dir_or_file=batch_dir, raw_videos_dir=raw_video.parent)
    assert res.files_discovered == 1
    assert res.annotations_parsed == 1
    assert res.annotations_valid == 1
    assert res.source_hash_match == 1
    assert res.training_eligible == 1
