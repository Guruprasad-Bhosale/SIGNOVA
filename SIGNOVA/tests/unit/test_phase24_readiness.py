"""
Phase 24 Readiness & Baseline Diagnostics Tests.
"""

from pathlib import Path
from signova.operations.phase24_orchestrator import evaluate_phase24_readiness, Phase24Orchestrator


def test_phase24_readiness_under_current_repo_state():
    res = evaluate_phase24_readiness()
    assert res["phase"] == 24
    assert res["supervision"]["state"] == "STATE_B"
    assert res["supervision"]["phase19_authorized"] is False
    assert res["training"]["status"] == "BLOCKED"
    assert res["training"]["checkpoint"] == "NONE"
    assert res["state_reporting"]["experiment_status"] == "BLOCKED"
    assert res["state_reporting"]["isl_recognition_validation"] == "NOT_PERFORMED"
    assert res["state_reporting"]["gloss_to_english_validation"] == "NOT_PERFORMED"
    assert res["live_model_authorized"] is False if "live_model_authorized" in res else res["model_readiness"]["live_authorized"] is False
    assert res["next_physical_action"] == "Acquire genuine human sequential ISL annotations."
    assert res["reference_integrity"]["all_44_files_unchanged"] is True


def test_phase24_orchestrator_initialization(tmp_path):
    orch = Phase24Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    assert orch.workspace_root == tmp_path
    assert (tmp_path / "data" / "datasets" / "phase24_artifacts").exists()
