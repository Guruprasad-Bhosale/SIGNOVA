"""
Phase 23 Annotator Qualification & Batch Ingestion Accounting Tests.
"""

import json
from signova.operations.phase23_orchestrator import Phase23Orchestrator


def test_unqualified_annotator_batch_accounting(tmp_path):
    orch = Phase23Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.phase22_orch.register_annotator("ann_qual", qualification_status="QUALIFIED")
    orch.phase22_orch.register_annotator("ann_unqual", qualification_status="PENDING")

    batch_dir = tmp_path / "incoming"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # 1. Qualified annotator
    ann1 = {
        "annotation_id": "ann_q1",
        "video_id": "v1.mp4",
        "annotator_id": "ann_qual",
        "gloss_tokens": ["HELLO"],
        "annotation_source": "HUMAN_DIRECT",
        "source_sha256": "SHA1",
        "review_state": "VERIFIED",
    }
    (batch_dir / "ann1.json").write_text(json.dumps(ann1), encoding="utf-8")

    # 2. Unqualified annotator
    ann2 = {
        "annotation_id": "ann_uq1",
        "video_id": "v2.mp4",
        "annotator_id": "ann_unqual",
        "gloss_tokens": ["WORLD"],
        "annotation_source": "HUMAN_DIRECT",
        "source_sha256": "SHA2",
        "review_state": "VERIFIED",
    }
    (batch_dir / "ann2.json").write_text(json.dumps(ann2), encoding="utf-8")

    res = orch.ingest_batch_human_data(source_dir_or_file=batch_dir)
    assert res.qualified_annotators == 1
    assert res.unqualified_annotators == 1
    assert res.training_eligible == 1
    assert res.training_ineligible == 1
    assert any("ANNOTATOR_NOT_QUALIFIED" in k for k in res.rejection_reasons.keys())
