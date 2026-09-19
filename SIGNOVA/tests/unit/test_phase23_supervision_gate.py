"""
Phase 23 Canonical Phase 19 Gate Authority Tests.
"""

from signova.operations.phase23_orchestrator import evaluate_phase23_readiness


def test_phase23_supervision_gate_authority():
    res = evaluate_phase23_readiness()
    assert res["supervision_state"] == "STATE_B"
    assert res["phase19"]["state"] == "BLOCKED"
    assert res["phase19"]["authorized"] is False
    assert res["training_readiness"] == "NOT_READY"
    assert res["training_execution"] == "BLOCKED"
