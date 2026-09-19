"""
Phase 24 CTC Feasibility & Deficit Analysis Tests.
"""

from signova.operations.phase24_orchestrator import Phase24Orchestrator


def test_ctc_feasibility_evaluation_and_artifacts(tmp_path):
    orch = Phase24Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.phase22_orch.register_annotator("ann_01", qualification_status="QUALIFIED")

    orch.phase22_orch.import_human_annotation(
        annotation_id="ann_ctc_eval",
        video_id="vid_ctc.mp4",
        annotator_id="ann_01",
        gloss_sequence=["HELLO", "INDIA"],
        frame_count=64,
        review_status="VERIFIED"
    )

    res = orch.evaluate_ctc_feasibility()
    assert res["total_samples"] == 1
    assert res["feasible_samples"] == 1
    assert res["infeasible_samples"] == 0
    assert (tmp_path / "data" / "datasets" / "phase24_artifacts" / "phase24_ctc_feasibility.json").exists()
