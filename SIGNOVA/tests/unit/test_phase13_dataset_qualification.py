"""
Phase 13 Dataset Qualification & Scale Classification Unit Tests.
"""

from signova.pilot.constants import (
    DATASET_SCALE_DATA_LIMITED,
    DATASET_SCALE_NO_DATA,
    DATASET_SCALE_PILOT_ONLY,
    DATASET_SCALE_RESEARCH_SCALE,
)
from signova.pilot.pilot_orchestrator import Phase13PilotOrchestrator


def test_classify_dataset_scale():
    orch = Phase13PilotOrchestrator()

    assert orch.classify_dataset_scale(0, 0) == DATASET_SCALE_NO_DATA
    assert orch.classify_dataset_scale(4, 2) == DATASET_SCALE_PILOT_ONLY
    assert orch.classify_dataset_scale(20, 10) == DATASET_SCALE_DATA_LIMITED
    assert orch.classify_dataset_scale(60, 25) == DATASET_SCALE_RESEARCH_SCALE
