"""
Unit tests for Phase 10 Gate State Machine and Real CTC Training Lock.
"""

import json
from pathlib import Path
import pytest
from signova.recognition.phase9_gate import RealCTCTrainingBlockedError, RealCTCTrainingGate


def test_phase10_gate_report_structure():
    gate_path = Path("outputs/reports/phase10_gate.json")
    assert gate_path.is_file(), "outputs/reports/phase10_gate.json must exist"

    data = json.loads(gate_path.read_text(encoding="utf-8"))
    assert data["phase"] == 10
    assert data["starting_state"] == "STATE C"
    assert data["ending_state"] == "STATE C"
    assert data["real_ctc_training"] is False
    assert data["state_transition_allowed"] is False
    assert data["acquisition_outcome"] == "NO_ELIGIBLE_DATASET"


def test_phase10_ctc_gate_hard_blocked():
    gate = RealCTCTrainingGate()
    assert gate.can_train() is False

    with pytest.raises(RealCTCTrainingBlockedError):
        gate.enforce_gate()
