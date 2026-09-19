"""
Phase 21 Hard Training Gate and Fast-Exit Verification Tests.
"""

from pathlib import Path
from signova.operations.phase21_orchestrator import Phase21Orchestrator


def test_fast_exit_on_no_human_data(tmp_path):
    orch = Phase21Orchestrator(workspace_root=tmp_path, experiments_dir=tmp_path / "exp")
    res = orch.build_genuine_dataset()

    assert res["status"] == "FAST_EXIT_BLOCKED"
    assert res["supervision_state"] == "STATE_B"
    assert res["samples"] == []
    assert res["vocabulary"] is None


def test_training_refusal_under_state_b(tmp_path):
    orch = Phase21Orchestrator(workspace_root=tmp_path, experiments_dir=tmp_path / "exp")
    res = orch.train_real_ctc()

    assert res["status"] == "BLOCKED"
    assert res["supervision_state"] == "STATE_B"
    assert res["checkpoint_path"] is None
