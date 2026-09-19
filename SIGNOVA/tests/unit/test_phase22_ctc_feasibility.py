"""
Phase 22 Repeated-Token CTC Feasibility Calculation Tests.
"""

from signova.operations.phase22_orchestrator import evaluate_sample_ctc_feasibility, Phase22Orchestrator


def test_repeated_token_ctc_feasibility_formula():
    # Sequence with no adjacent duplicates: ["A", "B", "C"] -> L = 3, repeats = 0 -> T_req = 3
    res1 = evaluate_sample_ctc_feasibility("sample_1", ["A", "B", "C"], t_features=4)
    assert res1["l_tokens"] == 3
    assert res1["repeats"] == 0
    assert res1["t_required"] == 3
    assert res1["ctc_feasible"] is True

    # Sequence with adjacent duplicate: ["A", "A", "B"] -> L = 3, repeats = 1 -> T_req = 4
    # If t_features is 3: T_required (4) > T_features (3) -> Infeasible
    res2 = evaluate_sample_ctc_feasibility("sample_2", ["A", "A", "B"], t_features=3)
    assert res2["l_tokens"] == 3
    assert res2["repeats"] == 1
    assert res2["t_required"] == 4
    assert res2["ctc_feasible"] is False
    assert "exceeds" in res2["reason"]

    # If t_features is 4: T_required (4) <= T_features (4) -> Feasible
    res3 = evaluate_sample_ctc_feasibility("sample_3", ["A", "A", "B"], t_features=4)
    assert res3["ctc_feasible"] is True


def test_ctc_feasibility_dataset_evaluation(tmp_path):
    orch = Phase22Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.register_annotator("ann_01", qualification_status="QUALIFIED")

    # Sample 1: 60 frames -> T_features = 60. Tokens: 2. Feasible.
    orch.import_human_annotation(
        annotation_id="ann_ctc_1",
        video_id="vid_1",
        annotator_id="ann_01",
        gloss_sequence=["SIGN_A", "SIGN_B"],
        frame_count=60
    )
    orch.review_annotation("ann_ctc_1", "VERIFIED", reviewer_id="Lead_01")

    res = orch.check_ctc_feasibility()
    assert res["total_samples"] == 1
    assert res["feasible_samples"] == 1
    assert res["infeasible_samples"] == 0
    assert res["dataset_ctc_feasible"] is True
