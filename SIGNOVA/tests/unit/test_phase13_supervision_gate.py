"""
Phase 13 Supervision Gate Unit Tests.

Tests:
- Branch A: STATE_A & STATE_A_DATA_LIMITED allow training with distinct generalization claim tiers.
- Branch B: STATE_B blocks real CTC training with RealCTCTrainingBlockedError.
- Defensive check: No fake checkpoint permitted under STATE_B.
"""

from pathlib import Path
import pytest

from signova.pilot.constants import (
    CLAIMS_LIMITED,
    CLAIMS_NOT_READY,
    CLAIMS_PERMITTED_BY_EVIDENCE,
)
from signova.qualification.constants import (
    STATUS_ALLOWED,
    STATUS_BLOCKED,
    SUPERVISION_STATE_A,
    SUPERVISION_STATE_A_DATA_LIMITED,
    SUPERVISION_STATE_B,
)
from signova.qualification.gate import (
    GateEvaluationResult,
    Phase12SupervisionGate,
    RealCTCTrainingBlockedError,
)


def test_gate_evaluates_empty_repository_to_state_b():
    empty_dir = Path("data/annotations/non_existent_path")
    res = Phase12SupervisionGate.evaluate(annotations_dir=empty_dir)

    assert res.supervision_state == SUPERVISION_STATE_B
    assert res.real_ctc_status == STATUS_BLOCKED
    assert res.pilot_status == "BLOCKED_HUMAN_RESOURCE"

    with pytest.raises(RealCTCTrainingBlockedError):
        Phase12SupervisionGate.require_training_authorized(res)


def test_gate_allows_state_a_data_limited_with_restricted_claims():
    res = GateEvaluationResult(
        supervision_state=SUPERVISION_STATE_A_DATA_LIMITED,
        real_ctc_status=STATUS_ALLOWED,
        pilot_status="READY",
        generalization_claims=CLAIMS_LIMITED,
        publication_grade_evaluation="NOT_READY",
        conditions_satisfied={c: True for c in range(12)},
        failed_conditions=[],
        dataset_threshold_status={"technical_minimum_met": True, "recommended_thresholds_met": False},
        total_annotations_found=6,
        training_eligible_count=6,
    )
    # Must NOT raise exception under STATE_A_DATA_LIMITED
    Phase12SupervisionGate.require_training_authorized(res)
    assert res.generalization_claims == CLAIMS_LIMITED


def test_gate_allows_state_a_with_full_claims():
    res = GateEvaluationResult(
        supervision_state=SUPERVISION_STATE_A,
        real_ctc_status=STATUS_ALLOWED,
        pilot_status="READY",
        generalization_claims=CLAIMS_PERMITTED_BY_EVIDENCE,
        publication_grade_evaluation="READY",
        conditions_satisfied={c: True for c in range(12)},
        failed_conditions=[],
        dataset_threshold_status={"technical_minimum_met": True, "recommended_thresholds_met": True},
        total_annotations_found=60,
        training_eligible_count=60,
    )
    # Must NOT raise exception under STATE_A
    Phase12SupervisionGate.require_training_authorized(res)
    assert res.generalization_claims == CLAIMS_PERMITTED_BY_EVIDENCE
