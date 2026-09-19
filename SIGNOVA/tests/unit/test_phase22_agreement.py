"""
Phase 22 Inter-Annotator Agreement Tests.
"""

from signova.operations.phase22_orchestrator import Phase22Orchestrator


def test_inter_annotator_agreement_calculation(tmp_path):
    orch = Phase22Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.register_annotator("ann_01", qualification_status="QUALIFIED")
    orch.register_annotator("ann_02", qualification_status="QUALIFIED")

    # Matching annotations
    orch.import_human_annotation(
        annotation_id="ann_agree_1",
        video_id="vid_pair_1",
        annotator_id="ann_01",
        gloss_sequence=["HELLO", "INDIA"]
    )
    orch.import_human_annotation(
        annotation_id="ann_agree_2",
        video_id="vid_pair_1",
        annotator_id="ann_02",
        gloss_sequence=["HELLO", "INDIA"]
    )

    # Disagreeing annotations
    orch.import_human_annotation(
        annotation_id="ann_dis_1",
        video_id="vid_pair_2",
        annotator_id="ann_01",
        gloss_sequence=["THANK", "YOU"]
    )
    orch.import_human_annotation(
        annotation_id="ann_dis_2",
        video_id="vid_pair_2",
        annotator_id="ann_02",
        gloss_sequence=["SORRY", "YOU"]
    )

    res = orch.check_inter_annotator_agreement()
    assert res.get("pairs_evaluated") == 2
    assert res.get("exact_match_pairs") == 1
    assert res.get("exact_match_rate") == 0.5
