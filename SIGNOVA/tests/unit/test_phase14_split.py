"""
Phase 14 Split Strategy & Identity Separation Unit Tests.
"""

from signova.annotation.schema import VideoAnnotation
from signova.operations.constants import (
    SPLIT_STRATEGY_RANDOM,
    SPLIT_STRATEGY_SESSION_INDEPENDENT,
    SPLIT_STRATEGY_SIGNER_INDEPENDENT,
)
from signova.pilot.pilot_orchestrator import Phase13PilotOrchestrator


def test_split_strategy_hierarchy():
    orch = Phase13PilotOrchestrator()

    # When verified signers are available
    annots_signer = [
        VideoAnnotation("a1", "s1", "u1", False, ["A"], metadata={"signer_id": "signer_10"}),
        VideoAnnotation("a2", "s2", "u2", False, ["B"], metadata={"signer_id": "signer_20"}),
    ]
    strat, _ = orch.determine_split_strategy(annots_signer)
    assert strat == SPLIT_STRATEGY_SIGNER_INDEPENDENT

    # When only sessions are available
    annots_session = [
        VideoAnnotation("a1", "s1", "u1", False, ["A"], metadata={"session_id": "sess_A"}),
        VideoAnnotation("a2", "s2", "u2", False, ["B"], metadata={"session_id": "sess_B"}),
    ]
    strat, _ = orch.determine_split_strategy(annots_session)
    assert strat == SPLIT_STRATEGY_SESSION_INDEPENDENT

    # Fallback to random when metadata is unknown
    annots_unknown = [
        VideoAnnotation("a1", "s1", "u1", False, ["A"]),
        VideoAnnotation("a2", "s2", "u2", False, ["B"]),
    ]
    strat, _ = orch.determine_split_strategy(annots_unknown)
    assert strat == SPLIT_STRATEGY_RANDOM
