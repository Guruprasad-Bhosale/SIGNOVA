"""
Phase 22 Canonical Phase 19 Gate & Phase 21 Unlock Tests.
"""

from signova.operations.phase22_orchestrator import evaluate_phase22_readiness


def test_phase22_supervision_gate_behavior():
    readiness = evaluate_phase22_readiness()

    # Under current repo conditions: zero qualified annotations
    assert readiness["final_state"] == "STATE_B"
    assert readiness["phase19_authorized"] is False
    assert readiness["phase21_training_status"] == "BLOCKED"
    assert readiness["training_eligible_count"] == 0
    assert "no_genuine_human_annotations_present" in readiness["phase19_reason"]
    assert readiness["next_physical_action"] == "Acquire genuine human sequential ISL annotations."
