"""
Phase 15 Preferred Split Hierarchy & Independence Constraint Unit Tests.

Validates:
- Preferred split hierarchy: SIGNER_INDEPENDENT -> SESSION_INDEPENDENT -> SOURCE_GROUP_INDEPENDENT -> RANDOM.
- Selection of the strongest available valid independence constraint supported by verified metadata.
- Explicit notation that SIGNER_INDEPENDENT cannot be claimed when signer metadata is unavailable.
"""

from signova.annotation.schema import VideoAnnotation
from signova.operations.phase15_orchestrator import Phase15Orchestrator
from signova.pilot.constants import (
    SPLIT_STRATEGY_SIGNER_INDEPENDENT,
    SPLIT_STRATEGY_SESSION_INDEPENDENT,
    SPLIT_STRATEGY_RANDOM,
)


def test_split_hierarchy_prefers_signer_independent_when_signers_present():
    orch = Phase15Orchestrator()
    annots = [
        VideoAnnotation("p15_a1", "s1", "u1", False, ["A"], metadata={"signer_id": "signer_delhi_01"}),
        VideoAnnotation("p15_a2", "s2", "u2", False, ["B"], metadata={"signer_id": "signer_mumbai_02"}),
    ]
    strat, explanation = orch._select_split_strategy(annots)
    assert strat == SPLIT_STRATEGY_SIGNER_INDEPENDENT
    assert "signer" in explanation.lower()


def test_split_hierarchy_prefers_session_independent_when_sessions_present():
    orch = Phase15Orchestrator()
    annots = [
        VideoAnnotation("p15_a1", "s1", "u1", False, ["A"], metadata={"session_id": "sess_recording_01"}),
        VideoAnnotation("p15_a2", "s2", "u2", False, ["B"], metadata={"session_id": "sess_recording_02"}),
    ]
    strat, explanation = orch._select_split_strategy(annots)
    assert strat == SPLIT_STRATEGY_SESSION_INDEPENDENT
    assert "session" in explanation.lower()


def test_split_hierarchy_falls_back_to_random_when_metadata_absent():
    orch = Phase15Orchestrator()
    annots = [
        VideoAnnotation("p15_a1", "s1", "u1", False, ["A"]),
        VideoAnnotation("p15_a2", "s2", "u2", False, ["B"]),
    ]
    strat, explanation = orch._select_split_strategy(annots)
    assert strat == SPLIT_STRATEGY_RANDOM
    assert "NOT supported" in explanation
