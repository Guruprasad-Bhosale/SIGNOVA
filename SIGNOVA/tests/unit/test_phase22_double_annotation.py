"""
Phase 22 Double Annotation & Dual Verification Tests.
"""

from signova.operations.phase22_orchestrator import Phase22Orchestrator


def test_double_annotation_management(tmp_path):
    orch = Phase22Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.register_annotator("ann_01", qualification_status="QUALIFIED")
    orch.register_annotator("ann_02", qualification_status="QUALIFIED")

    # Import primary and secondary annotations for same video
    orch.import_human_annotation(
        annotation_id="ann_dual_1",
        video_id="vid_dual",
        annotator_id="ann_01",
        gloss_sequence=["HELLO", "WORLD"]
    )
    orch.import_human_annotation(
        annotation_id="ann_dual_2",
        video_id="vid_dual",
        annotator_id="ann_02",
        gloss_sequence=["HELLO", "WORLD"]
    )

    anns = orch.list_annotations_for_video("vid_dual")
    assert len(anns) == 2
    annotator_ids = {a.annotator_id for a in anns}
    assert annotator_ids == {"ann_01", "ann_02"}
