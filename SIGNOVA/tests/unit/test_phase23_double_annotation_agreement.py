"""
Phase 23 Double Annotation Tracking & Agreement Status Tests.
"""

from signova.operations.phase23_orchestrator import Phase23Orchestrator


def test_double_annotation_tracking_and_agreement(tmp_path):
    orch = Phase23Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.phase22_orch.register_annotator("ann_01", qualification_status="QUALIFIED")
    orch.phase22_orch.register_annotator("ann_02", qualification_status="QUALIFIED")

    # Initially zero pairs -> NOT_COMPUTABLE
    acc1 = orch.get_data_accounting()
    assert acc1["double_annotation"]["agreement_status"] == "NOT_COMPUTABLE"
    assert acc1["double_annotation"]["double_completed"] == 0

    # Add dual annotations for same video
    orch.phase22_orch.import_human_annotation(
        annotation_id="ann_d1",
        video_id="vid_pair.mp4",
        annotator_id="ann_01",
        gloss_sequence=["HELLO", "INDIA"],
        review_status="VERIFIED"
    )
    orch.phase22_orch.import_human_annotation(
        annotation_id="ann_d2",
        video_id="vid_pair.mp4",
        annotator_id="ann_02",
        gloss_sequence=["HELLO", "INDIA"],
        review_status="VERIFIED"
    )

    acc2 = orch.get_data_accounting()
    assert acc2["double_annotation"]["agreement_status"] == "COMPUTABLE"
    assert acc2["double_annotation"]["double_completed"] == 1
    assert acc2["double_annotation"]["agreement_rate"] == 1.0
