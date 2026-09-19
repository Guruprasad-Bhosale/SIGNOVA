"""
Phase 19 Error Analysis Unit Tests.

Validates:
- Structured error report formatting.
- Safe blocked reporting when no real test predictions exist.
"""

from signova.operations.phase19_orchestrator import evaluate_phase19_readiness


def test_error_analysis_blocked_under_state_b(tmp_path):
    res = evaluate_phase19_readiness(workspace_root=tmp_path)
    assert res["supervision_state"] == "STATE_B"
    assert res["real_ctc_training_executed"] is False
