"""
Phase 23 Zero-Data Fast Exit & Default State Tests.
"""

from pathlib import Path
from signova.operations.phase23_orchestrator import evaluate_phase23_readiness, Phase23Orchestrator


def test_phase23_zero_data_readiness_under_current_repo_state():
    res = evaluate_phase23_readiness()
    assert res["phase"] == 23
    assert res["supervision_state"] == "STATE_B"
    assert res["final_state"] == "STATE_B"
    assert res["acquisition_status"] == "NOT_STARTED"
    assert res["training_readiness"] == "NOT_READY"
    assert res["training_execution"] == "BLOCKED"
    assert res["phase19"]["authorized"] is False
    assert res["phase21"]["training_status"] == "BLOCKED"
    assert res["phase21"]["checkpoint"] == "NONE"
    assert res["live_model"]["authorized"] is False
    assert res["next_physical_action"] == "Acquire genuine human sequential ISL annotations."
    assert res["reference_integrity"]["all_44_files_unchanged"] is True


def test_phase23_orchestrator_initialization(tmp_path):
    orch = Phase23Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    assert orch.workspace_root == tmp_path
    assert (tmp_path / "data" / "datasets" / "phase23_frozen").exists()
