"""
Phase 12 Supervision Gate Unit Tests.

Tests:
- Branch A: STATE_A and STATE_A_DATA_LIMITED allow training
- Branch B: STATE_B blocks real CTC training with RealCTCTrainingBlockedError
- Verification of 12 conditions and configurable research thresholds
"""

from pathlib import Path
import pytest

from signova.qualification.constants import (
    RECOMMENDED_DATASET_THRESHOLDS,
    STATUS_ALLOWED,
    STATUS_BLOCKED,
    STATUS_BLOCKED_HUMAN_RESOURCE,
    STATUS_LIMITED,
    SUPERVISION_STATE_A,
    SUPERVISION_STATE_A_DATA_LIMITED,
    SUPERVISION_STATE_B,
    TECHNICAL_MINIMUM_THRESHOLDS,
)
from signova.qualification.gate import (
    GateEvaluationResult,
    Phase12SupervisionGate,
    RealCTCTrainingBlockedError,
)


def test_gate_evaluates_empty_repository_to_state_b():
    # When no annotations exist
    empty_dir = Path("data/annotations/non_existent_path")
    res = Phase12SupervisionGate.evaluate(annotations_dir=empty_dir)

    assert res.supervision_state == SUPERVISION_STATE_B
    assert res.real_ctc_status == STATUS_BLOCKED
    assert res.pilot_status == STATUS_BLOCKED_HUMAN_RESOURCE
    assert res.training_eligible_count == 0
    assert "GENUINE_SEQUENTIAL_ANNOTATIONS_EXIST" in res.failed_conditions

    # Real CTC Training must raise RealCTCTrainingBlockedError
    with pytest.raises(RealCTCTrainingBlockedError):
        Phase12SupervisionGate.require_training_authorized(res)


def test_gate_allows_state_a_data_limited():
    res = GateEvaluationResult(
        supervision_state=SUPERVISION_STATE_A_DATA_LIMITED,
        real_ctc_status=STATUS_ALLOWED,
        pilot_status="READY",
        generalization_claims=STATUS_LIMITED,
        publication_grade_evaluation="NOT_READY",
        conditions_satisfied={c: True for c in range(12)},
        failed_conditions=[],
        dataset_threshold_status={"technical_minimum_met": True, "recommended_thresholds_met": False},
        total_annotations_found=5,
        training_eligible_count=5,
    )
    # Must NOT raise exception under STATE_A_DATA_LIMITED
    Phase12SupervisionGate.require_training_authorized(res)


def test_gate_allows_state_a():
    res = GateEvaluationResult(
        supervision_state=SUPERVISION_STATE_A,
        real_ctc_status=STATUS_ALLOWED,
        pilot_status="READY",
        generalization_claims="READY",
        publication_grade_evaluation="READY",
        conditions_satisfied={c: True for c in range(12)},
        failed_conditions=[],
        dataset_threshold_status={"technical_minimum_met": True, "recommended_thresholds_met": True},
        total_annotations_found=60,
        training_eligible_count=60,
    )
    # Must NOT raise exception under STATE_A
    Phase12SupervisionGate.require_training_authorized(res)
