"""
Phase 13 Leakage and Split Strategy Unit Tests.
"""

from signova.annotation.schema import VideoAnnotation
from signova.pilot.constants import (
    SPLIT_STRATEGY_RANDOM,
    SPLIT_STRATEGY_SESSION_INDEPENDENT,
    SPLIT_STRATEGY_SIGNER_INDEPENDENT,
)
from signova.pilot.pilot_orchestrator import Phase13PilotOrchestrator
from signova.qualification.leakage import audit_phase12_leakage


def test_split_strategy_priority():
    orch = Phase13PilotOrchestrator()

    # When 2 distinct signers exist
    ann_signers = [
        VideoAnnotation("a1", "s1", "u1", False, ["A"], metadata={"signer_id": "signer_1"}),
        VideoAnnotation("a2", "s2", "u1", False, ["B"], metadata={"signer_id": "signer_2"}),
    ]
    strat, lim = orch.determine_split_strategy(ann_signers)
    assert strat == SPLIT_STRATEGY_SIGNER_INDEPENDENT

    # When only 2 sessions exist (signers unknown)
    ann_sessions = [
        VideoAnnotation("a1", "s1", "u1", False, ["A"], metadata={"session_id": "sess_1"}),
        VideoAnnotation("a2", "s2", "u1", False, ["B"], metadata={"session_id": "sess_2"}),
    ]
    strat, lim = orch.determine_split_strategy(ann_sessions)
    assert strat == SPLIT_STRATEGY_SESSION_INDEPENDENT

    # When metadata is unknown
    ann_unknown = [
        VideoAnnotation("a1", "s1", "u1", False, ["A"]),
        VideoAnnotation("a2", "s2", "u1", False, ["B"]),
    ]
    strat, lim = orch.determine_split_strategy(ann_unknown)
    assert strat == SPLIT_STRATEGY_RANDOM
