"""
Phase 23 Gated Training Execution & Safety Guard Tests.
"""

from signova.operations.phase23_orchestrator import Phase23Orchestrator


def test_training_refusal_when_unauthorized(tmp_path):
    orch = Phase23Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    res = orch.run_gated_training(train_flag=True)

    assert res["training_execution"] == "BLOCKED"
    assert res["authorized"] is False
    assert res["checkpoint"] == "NONE"


def test_training_safety_guard_when_flag_omitted(tmp_path, monkeypatch):
    orch = Phase23Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")

    # Mock Phase 19 gate to simulate authorized state
    from signova.operations import phase23_orchestrator
    monkeypatch.setattr(
        phase23_orchestrator,
        "evaluate_phase19_readiness",
        lambda workspace_root: {"real_ctc_training_allowed": True, "supervision_state": "STATE_A_DATA_LIMITED"}
    )

    # Without --train flag -> NOT_REQUESTED
    res = orch.run_gated_training(train_flag=False)
    assert res["training_execution"] == "NOT_REQUESTED"
    assert res["authorized"] is True
    assert res["checkpoint"] == "NONE"
