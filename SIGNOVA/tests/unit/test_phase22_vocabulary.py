"""
Phase 22 Vocabulary Formation Tests.
"""

from signova.operations.phase22_orchestrator import Phase22Orchestrator


def test_vocabulary_build_from_annotations(tmp_path):
    orch = Phase22Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.register_annotator("ann_01", qualification_status="QUALIFIED")

    orch.import_human_annotation(
        annotation_id="ann_v_1",
        video_id="vid_1",
        annotator_id="ann_01",
        gloss_sequence=["HELLO", "NAMASTE", "WELCOME"]
    )
    orch.review_annotation("ann_v_1", "VERIFIED", reviewer_id="Lead_01")

    orch.import_human_annotation(
        annotation_id="ann_v_2",
        video_id="vid_2",
        annotator_id="ann_01",
        gloss_sequence=["THANK_YOU", "WELCOME"]
    )
    orch.review_annotation("ann_v_2", "VERIFIED", reviewer_id="Lead_01")

    vocab = orch.build_vocabulary()
    assert vocab["vocabulary_size"] >= 6  # blank, unk, + 4 tokens
    assert "HELLO" in vocab["tokens"]
    assert "NAMASTE" in vocab["tokens"]
    assert "WELCOME" in vocab["tokens"]
    assert "THANK_YOU" in vocab["tokens"]
