"""
Phase 15 Centralized Supervision Gate Unit Tests.

Validates the dynamic gate states: STATE_B, STATE_A_DATA_LIMITED, and STATE_A,
and verifies strict enforcement of training authorization and generalization claim distinctions.
"""

import pytest
from signova.qualification.constants import (
    SUPERVISION_STATE_B,
    SUPERVISION_STATE_A_DATA_LIMITED,
    SUPERVISION_STATE_A,
    STATUS_ALLOWED,
    STATUS_BLOCKED,
)
from signova.pilot.constants import (
    CLAIMS_NOT_READY,
    CLAIMS_LIMITED,
    CLAIMS_PERMITTED_BY_EVIDENCE,
)
from signova.qualification.gate import (
    Phase12SupervisionGate,
    GateEvaluationResult,
    RealCTCTrainingBlockedError,
)
from signova.operations.phase15_orchestrator import Phase15Orchestrator


def test_gate_state_b_defensively_blocks_training():
    """Under STATE_B, training is strictly blocked and raises RealCTCTrainingBlockedError."""
    res = GateEvaluationResult(
        supervision_state=SUPERVISION_STATE_B,
        real_ctc_status=STATUS_BLOCKED,
        pilot_status="BLOCKED_HUMAN_RESOURCE",
        generalization_claims=CLAIMS_NOT_READY,
        publication_grade_evaluation="NOT_READY",
        conditions_satisfied={},
        failed_conditions=["GENUINE_SEQUENTIAL_ANNOTATIONS_EXIST"],
        dataset_threshold_status={},
        total_annotations_found=0,
        training_eligible_count=0,
    )
    assert res.real_ctc_status == STATUS_BLOCKED
    with pytest.raises(RealCTCTrainingBlockedError):
        Phase12SupervisionGate.require_training_authorized(res)


def test_gate_state_a_data_limited_authorizes_training_with_limited_claims():
    """Under STATE_A_DATA_LIMITED, training is allowed but generalization claims are limited."""
    res = GateEvaluationResult(
        supervision_state=SUPERVISION_STATE_A_DATA_LIMITED,
        real_ctc_status=STATUS_ALLOWED,
        pilot_status="READY",
        generalization_claims=CLAIMS_LIMITED,
        publication_grade_evaluation="NOT_READY",
        conditions_satisfied={c: True for c in range(12)},
        failed_conditions=[],
        dataset_threshold_status={"scale": "DATA_LIMITED"},
        total_annotations_found=15,
        training_eligible_count=15,
    )
    assert res.real_ctc_status == STATUS_ALLOWED
    assert res.generalization_claims == CLAIMS_LIMITED
    assert res.publication_grade_evaluation == "NOT_READY"
    # Should not raise
    Phase12SupervisionGate.require_training_authorized(res)


def test_gate_state_a_authorizes_full_research_scale_training():
    """Under STATE_A, training is allowed and generalization claims are permitted by evidence."""
    res = GateEvaluationResult(
        supervision_state=SUPERVISION_STATE_A,
        real_ctc_status=STATUS_ALLOWED,
        pilot_status="READY",
        generalization_claims=CLAIMS_PERMITTED_BY_EVIDENCE,
        publication_grade_evaluation="READY",
        conditions_satisfied={c: True for c in range(12)},
        failed_conditions=[],
        dataset_threshold_status={"scale": "RESEARCH_SCALE"},
        total_annotations_found=60,
        training_eligible_count=60,
    )
    assert res.real_ctc_status == STATUS_ALLOWED
    assert res.generalization_claims == CLAIMS_PERMITTED_BY_EVIDENCE
    assert res.publication_grade_evaluation == "READY"
    # Should not raise
    Phase12SupervisionGate.require_training_authorized(res)


def test_phase15_orchestrator_dynamic_evaluation_without_hardcoding(tmp_path):
    """Dynamic evaluation with no annotations must naturally resolve to STATE_B."""
    empty_dir = tmp_path / "empty_pilot"
    empty_dir.mkdir()

    orchestrator = Phase15Orchestrator(annotations_dir=empty_dir)
    report = orchestrator.run_full_qualification()

    assert report["supervision_state"] == SUPERVISION_STATE_B
    assert report["real_ctc_status"] == STATUS_BLOCKED
    assert report["pilot_status"] == "BLOCKED_HUMAN_RESOURCE"
    assert report["generalization_claims"] == CLAIMS_NOT_READY
    assert report["total_annotations_found"] == 0
