"""
Phase 14 Human Resource & Blocker Reporting Unit Tests.
"""

from pathlib import Path
from signova.qualification.constants import STATUS_BLOCKED_HUMAN_RESOURCE, SUPERVISION_STATE_B
from signova.qualification.gate import Phase12SupervisionGate


def test_human_resource_unavailable_reported_as_state_b():
    empty_dir = Path("data/annotations/empty_test_path")
    res = Phase12SupervisionGate.evaluate(annotations_dir=empty_dir)

    assert res.supervision_state == SUPERVISION_STATE_B
    assert res.pilot_status == STATUS_BLOCKED_HUMAN_RESOURCE
    assert res.training_eligible_count == 0
