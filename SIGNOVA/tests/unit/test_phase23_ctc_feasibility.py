"""
Phase 23 Repeated-Token CTC Feasibility & Deficit Tests.
"""

from signova.operations.phase23_orchestrator import evaluate_sample_ctc_feasibility, Phase23Orchestrator


def test_ctc_feasibility_deficit_reporting(tmp_path):
    orch = Phase23Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.phase22_orch.register_annotator("ann_01", qualification_status="QUALIFIED")

    # Infeasible sequence: ["A", "A", "A", "A"] -> L = 4, repeats = 3 -> T_req = 7. Feature frames = 4.
    orch.phase22_orch.import_human_annotation(
        annotation_id="ann_inf_1",
        video_id="v_inf.mp4",
        annotator_id="ann_01",
        gloss_sequence=["A", "A", "A", "A"],
        frame_count=4,
        review_status="VERIFIED"
    )

    acc = orch.get_data_accounting()
    deficits = acc.get("worst_ctc_deficits", [])
    assert len(deficits) == 1
    assert deficits[0]["required_timesteps"] == 7
    assert deficits[0]["available_timesteps"] == 4
    assert deficits[0]["deficit"] == 3
