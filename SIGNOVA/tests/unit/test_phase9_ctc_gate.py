"""
Unit tests for Phase 9 Real CTC Training Gate blocking under STATE C.
"""

import pytest
from signova.recognition.phase9_gate import (
    Phase9GateStatus,
    RealCTCTrainingBlockedError,
    RealCTCTrainingGate,
)


def test_real_ctc_gate_blocks_under_state_c():
    gate = RealCTCTrainingGate()
    assert gate.can_train() is False

    with pytest.raises(RealCTCTrainingBlockedError) as excinfo:
        gate.enforce_gate()

    assert "Real CTC training is forbidden until all Phase 9 verification criteria are met" in str(excinfo.value)
    assert "STATE C" in str(excinfo.value)


def test_real_ctc_gate_passes_when_fully_verified():
    status = Phase9GateStatus(
        data_gate_state="STATE A",
        verified_sequential_glosses=True,
        verified_video_pairing=True,
        valid_license=True,
        leakage_audit_passed=True,
        pilot_validation_passed=True,
        budget_spend_inr=0.0,
    )
    gate = RealCTCTrainingGate(status)
    assert gate.can_train() is True

    # enforce_gate should not raise
    gate.enforce_gate()
