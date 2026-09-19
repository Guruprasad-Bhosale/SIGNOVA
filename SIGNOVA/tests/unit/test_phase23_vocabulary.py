"""
Phase 23 Vocabulary Formation & Provenance Tests.
"""

from signova.operations.phase23_orchestrator import Phase23Orchestrator


def test_phase23_vocabulary_construction(tmp_path):
    orch = Phase23Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.phase22_orch.register_annotator("ann_01", qualification_status="QUALIFIED")

    orch.phase22_orch.import_human_annotation(
        annotation_id="ann_v1",
        video_id="v1.mp4",
        annotator_id="ann_01",
        gloss_sequence=["NAMASTE", "WELCOME"],
        review_status="VERIFIED"
    )

    vocab = orch.phase22_orch.build_vocabulary()
    assert "<BLANK>" in vocab["tokens"]
    assert "<UNK>" in vocab["tokens"]
    assert "NAMASTE" in vocab["tokens"]
    assert "WELCOME" in vocab["tokens"]
    assert vocab["vocabulary_size"] >= 4
