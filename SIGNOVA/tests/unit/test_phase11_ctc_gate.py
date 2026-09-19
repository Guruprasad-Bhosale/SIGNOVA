"""
Unit tests for Phase 11 CTC Readiness Gate and 12 Exact STATE_A Conditions.
"""

import pytest
from signova.annotation.constants import (
    EXACT_12_STATE_A_CONDITIONS,
    SUPERVISION_STATE_A,
    SUPERVISION_STATE_B,
    SUPERVISION_STATE_C,
)
from signova.recognition.phase11_gate import (
    Phase11SupervisionGate,
    RealCTCTrainingBlockedError,
)


def test_12_state_a_conditions_count():
    assert len(EXACT_12_STATE_A_CONDITIONS) == 12
    assert "GENUINE_SEQUENTIAL_ANNOTATIONS_EXIST" in EXACT_12_STATE_A_CONDITIONS
    assert "MINIMUM_SAMPLE_THRESHOLD_SATISFIED" in EXACT_12_STATE_A_CONDITIONS


def test_gate_blocks_under_state_b():
    gate = Phase11SupervisionGate(supervision_state=SUPERVISION_STATE_B)
    assert gate.is_real_ctc_training_permitted is False
    assert gate.is_state_a_unlocked is False

    with pytest.raises(RealCTCTrainingBlockedError) as excinfo:
        gate.enforce_gate()

    assert "STATE_B" in str(excinfo.value)
    assert "Real CTC training is forbidden until all 12 STATE_A prerequisites are met" in str(excinfo.value)


def test_gate_unlocks_only_when_all_12_conditions_met():
    gate = Phase11SupervisionGate(supervision_state=SUPERVISION_STATE_A)

    # Mark all 12 conditions satisfied
    for c in gate.conditions.values():
        c.is_satisfied = True

    assert gate.is_state_a_unlocked is True
    assert gate.is_real_ctc_training_permitted is True

    # Should not raise error
    gate.enforce_gate()
