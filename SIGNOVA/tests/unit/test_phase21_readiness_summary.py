"""
Phase 21 Readiness Summary & Gate Diagnostic Tests.
"""

from pathlib import Path
from signova.operations.phase21_orchestrator import evaluate_phase21_readiness, Phase21Orchestrator


def test_phase21_readiness_under_current_state_b():
    readiness = evaluate_phase21_readiness()
    assert readiness["phase"] == 21
    assert readiness["supervision_state"] == "STATE_B"
    assert readiness["real_ctc_training_allowed"] is False
    assert readiness["real_ctc_status"] == "BLOCKED"
    assert "no_genuine_human_annotations_present" in readiness["training_authorization"]["reason"]
    assert readiness["reference_integrity"]["all_44_files_unchanged"] is True


def test_phase21_gate_snapshot_creation(tmp_path):
    orch = Phase21Orchestrator(workspace_root=tmp_path, experiments_dir=tmp_path / "experiments")
    snapshot = orch.create_gate_snapshot()

    assert (tmp_path / "experiments" / "phase19_gate_snapshot.json").exists()
    assert snapshot["supervision_state"] == "STATE_B"
    assert snapshot["fast_exit_required"] is True
