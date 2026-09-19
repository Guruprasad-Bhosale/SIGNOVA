"""
Phase 23 Source Video Hash Provenance & Invalidation Tests.
"""

import hashlib
import json
from signova.operations.phase23_orchestrator import Phase23Orchestrator


def test_source_video_hash_mismatch_detection(tmp_path):
    orch = Phase23Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.phase22_orch.register_annotator("ann_01", qualification_status="QUALIFIED")

    # Write video
    raw_video = tmp_path / "videos" / "isl_vid_01.mp4"
    raw_video.parent.mkdir(parents=True, exist_ok=True)
    raw_video.write_bytes(b"actual_video_content")

    batch_dir = tmp_path / "incoming"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Annotation with wrong expected hash
    ann_data = {
        "annotation_id": "ann_mismatch",
        "video_id": "isl_vid_01.mp4",
        "annotator_id": "ann_01",
        "gloss_tokens": ["TEST"],
        "annotation_source": "HUMAN_DIRECT",
        "source_sha256": "DIFFERENT_HASH_VALUE",
        "source_uri": str(raw_video),
    }
    (batch_dir / "ann.json").write_text(json.dumps(ann_data), encoding="utf-8")

    res = orch.ingest_batch_human_data(source_dir_or_file=batch_dir, raw_videos_dir=raw_video.parent)
    assert res.source_hash_mismatch == 1
    assert "SOURCE_HASH_MISMATCH" in res.rejection_reasons
