"""
Phase 23 Prohibited Source & Invalid Rejection Tests.
"""

import json
from signova.operations.phase23_orchestrator import Phase23Orchestrator


def test_batch_ingestion_rejects_prohibited_sources(tmp_path):
    orch = Phase23Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.phase22_orch.register_annotator("ann_01", qualification_status="QUALIFIED")

    batch_dir = tmp_path / "incoming"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prohibited source: LLM_GENERATED
    ann_llm = {
        "annotation_id": "ann_bad_1",
        "video_id": "v1.mp4",
        "annotator_id": "ann_01",
        "gloss_tokens": ["HELLO"],
        "annotation_source": "LLM_GENERATED",
    }
    (batch_dir / "ann_llm.json").write_text(json.dumps(ann_llm), encoding="utf-8")

    # 2. Prohibited source: PSEUDO_LABEL
    ann_pseudo = {
        "annotation_id": "ann_bad_2",
        "video_id": "v2.mp4",
        "annotator_id": "ann_01",
        "gloss_tokens": ["THANK_YOU"],
        "annotation_source": "PSEUDO_LABEL",
    }
    (batch_dir / "ann_pseudo.json").write_text(json.dumps(ann_pseudo), encoding="utf-8")

    res = orch.ingest_batch_human_data(source_dir_or_file=batch_dir)
    assert res.files_discovered == 2
    assert res.annotations_invalid == 2
    assert res.annotations_valid == 0
    assert "PROHIBITED_SOURCE_LLM_GENERATED" in res.rejection_reasons
    assert "PROHIBITED_SOURCE_PSEUDO_LABEL" in res.rejection_reasons
