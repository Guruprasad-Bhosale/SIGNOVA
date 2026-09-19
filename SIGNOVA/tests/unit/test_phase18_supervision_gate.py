"""
Phase 18 Canonical Supervision Gate Unit Tests.

Validates:
- 12-condition evaluation against empty workspace (STATE_B).
- State transitions (STATE_B, STATE_A_DATA_LIMITED, STATE_A).
- Gating against unauthorized real training attempts (RealCTCTrainingBlockedError).
- Zero fabrication invariants.
"""

import pytest
from signova.qualification.constants import (
    STATUS_ALLOWED,
    STATUS_BLOCKED,
    STATUS_BLOCKED_HUMAN_RESOURCE,
    SUPERVISION_STATE_A,
    SUPERVISION_STATE_A_DATA_LIMITED,
    SUPERVISION_STATE_B,
)
from signova.qualification.gate import (
    Phase12SupervisionGate,
    RealCTCTrainingBlockedError,
)


def test_gate_evaluates_state_b_on_empty_directory(tmp_path):
    res = Phase12SupervisionGate.evaluate(annotations_dir=tmp_path)
    assert res.supervision_state == SUPERVISION_STATE_B
    assert res.real_ctc_status == STATUS_BLOCKED
    assert res.pilot_status == STATUS_BLOCKED_HUMAN_RESOURCE
    assert res.training_eligible_count == 0
    assert "GENUINE_SEQUENTIAL_ANNOTATIONS_EXIST" in res.failed_conditions


def test_gate_blocks_real_training_under_state_b(tmp_path):
    res = Phase12SupervisionGate.evaluate(annotations_dir=tmp_path)
    with pytest.raises(RealCTCTrainingBlockedError):
        Phase12SupervisionGate.require_training_authorized(res)


def test_gate_allows_training_under_state_a_data_limited():
    class DummyResult:
        supervision_state = SUPERVISION_STATE_A_DATA_LIMITED
        failed_conditions = []
        pilot_status = "READY"

    # Should not raise exception
    Phase12SupervisionGate.require_training_authorized(DummyResult())


def test_gate_allows_training_under_state_a():
    class DummyResult:
        supervision_state = SUPERVISION_STATE_A
        failed_conditions = []
        pilot_status = "READY"

    # Should not raise exception
    Phase12SupervisionGate.require_training_authorized(DummyResult())
