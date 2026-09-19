"""
Phase 14 Centralized Supervision Gate Unit Tests.
"""

from signova.pilot.constants import (
    CLAIMS_LIMITED,
    CLAIMS_NOT_READY,
    CLAIMS_PERMITTED_BY_EVIDENCE,
)
from signova.qualification.constants import (
    STATUS_ALLOWED,
    SUPERVISION_STATE_A,
    SUPERVISION_STATE_A_DATA_LIMITED,
    SUPERVISION_STATE_B,
)
from signova.qualification.gate import (
    GateEvaluationResult,
    Phase12SupervisionGate,
    RealCTCTrainingBlockedError,
)
import pytest


def test_centralized_gate_blocks_under_state_b():
    res = GateEvaluationResult(
        supervision_state=SUPERVISION_STATE_B,
        real_ctc_status="BLOCKED",
        pilot_status="BLOCKED_HUMAN_RESOURCE",
        generalization_claims=CLAIMS_NOT_READY,
        publication_grade_evaluation="NOT_READY",
        conditions_satisfied={},
        failed_conditions=["GENUINE_SEQUENTIAL_ANNOTATIONS_EXIST"],
        dataset_threshold_status={},
        total_annotations_found=0,
        training_eligible_count=0,
    )
    with pytest.raises(RealCTCTrainingBlockedError):
        Phase12SupervisionGate.require_training_authorized(res)


def test_centralized_gate_authorizes_state_a_data_limited():
    res = GateEvaluationResult(
        supervision_state=SUPERVISION_STATE_A_DATA_LIMITED,
        real_ctc_status=STATUS_ALLOWED,
        pilot_status="READY",
        generalization_claims=CLAIMS_LIMITED,
        publication_grade_evaluation="NOT_READY",
        conditions_satisfied={c: True for c in range(12)},
        failed_conditions=[],
        dataset_threshold_status={},
        total_annotations_found=8,
        training_eligible_count=8,
    )
    Phase12SupervisionGate.require_training_authorized(res)
