"""
Phase 24 Explicit Training Guard & Confirmation Tests.
"""

from signova.operations.phase24_orchestrator import Phase24Orchestrator


def test_training_guard_unauthorized(tmp_path):
    orch = Phase24Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    res = orch.run_gated_training(train_flag=True)
    assert res["experiment_status"] == "BLOCKED"
    assert res["authorized"] is False
    assert res["checkpoint"] == "NONE"


def test_training_guard_flag_omitted(tmp_path, monkeypatch):
    orch = Phase24Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")

    from signova.operations import phase24_orchestrator
    monkeypatch.setattr(
        phase24_orchestrator,
        "evaluate_phase19_readiness",
        lambda workspace_root: {"real_ctc_training_allowed": True, "supervision_state": "STATE_A_DATA_LIMITED"}
    )

    res = orch.run_gated_training(train_flag=False)
    assert res["experiment_status"] == "NOT_STARTED"
    assert res["authorized"] is True
    assert res["checkpoint"] == "NONE"
