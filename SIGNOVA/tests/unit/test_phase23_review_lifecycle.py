"""
Phase 23 Review Lifecycle & Batch Ingestion Accounting Tests.
"""

import json
from signova.operations.phase23_orchestrator import Phase23Orchestrator


def test_review_states_batch_accounting(tmp_path):
    orch = Phase23Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.phase22_orch.register_annotator("ann_01", qualification_status="QUALIFIED")

    batch_dir = tmp_path / "incoming"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # 1. Submitted
    (batch_dir / "ann_sub.json").write_text(json.dumps({
        "annotation_id": "ann_sub",
        "video_id": "v1.mp4",
        "annotator_id": "ann_01",
        "gloss_tokens": ["A"],
        "annotation_source": "HUMAN_DIRECT",
        "source_sha256": "SHA1",
        "review_state": "SUBMITTED",
    }), encoding="utf-8")

    # 2. Verified
    (batch_dir / "ann_ver.json").write_text(json.dumps({
        "annotation_id": "ann_ver",
        "video_id": "v2.mp4",
        "annotator_id": "ann_01",
        "gloss_tokens": ["B"],
        "annotation_source": "HUMAN_DIRECT",
        "source_sha256": "SHA2",
        "review_state": "VERIFIED",
    }), encoding="utf-8")

    # 3. Rejected
    (batch_dir / "ann_rej.json").write_text(json.dumps({
        "annotation_id": "ann_rej",
        "video_id": "v3.mp4",
        "annotator_id": "ann_01",
        "gloss_tokens": ["C"],
        "annotation_source": "HUMAN_DIRECT",
        "source_sha256": "SHA3",
        "review_state": "REJECTED",
    }), encoding="utf-8")

    # 4. Revision required
    (batch_dir / "ann_rev.json").write_text(json.dumps({
        "annotation_id": "ann_rev",
        "video_id": "v4.mp4",
        "annotator_id": "ann_01",
        "gloss_tokens": ["D"],
        "annotation_source": "HUMAN_DIRECT",
        "source_sha256": "SHA4",
        "review_state": "REVISION_REQUIRED",
    }), encoding="utf-8")

    res = orch.ingest_batch_human_data(source_dir_or_file=batch_dir)
    assert res.files_discovered == 4
    assert res.submitted == 1
    assert res.verified == 1
    assert res.rejected == 1
    assert res.revision_required == 1
