"""
Phase 22 Readiness Summary & Gate Diagnostic Tests.
"""

from pathlib import Path
from signova.operations.phase22_orchestrator import evaluate_phase22_readiness, Phase22Orchestrator


def test_phase22_readiness_under_current_state_b():
    readiness = evaluate_phase22_readiness()
    assert readiness["phase"] == 22
    assert readiness["final_state"] == "STATE_B"
    assert readiness["phase19_authorized"] is False
    assert readiness["phase21_training_status"] == "BLOCKED"
    assert readiness["acquisition_mode"] == "MANUAL_HUMAN"
    assert readiness["acquisition_status"] == "NOT_STARTED"
    assert readiness["next_physical_action"] == "Acquire genuine human sequential ISL annotations."


def test_phase22_orchestrator_initialization(tmp_path):
    orch = Phase22Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    assert orch.workspace_root == tmp_path
    assert orch.data_root == tmp_path / "data"
    assert (tmp_path / "data" / "annotations" / "human").exists()
    assert (tmp_path / "data" / "annotators").exists()
    assert (tmp_path / "data" / "assignments").exists()
    assert (tmp_path / "data" / "datasets" / "phase22").exists()
