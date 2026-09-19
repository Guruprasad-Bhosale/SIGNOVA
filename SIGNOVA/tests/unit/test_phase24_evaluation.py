"""
Phase 24 Held-Out Evaluation Refusal Tests.
"""

from signova.operations.phase24_orchestrator import Phase24Orchestrator


def test_held_out_evaluation_refusal_when_no_checkpoint(tmp_path):
    orch = Phase24Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    res = orch.run_held_out_evaluation()
    assert res["evaluation_status"] == "REFUSED_NO_CHECKPOINT"
    assert "refused" in res["message"].lower()
