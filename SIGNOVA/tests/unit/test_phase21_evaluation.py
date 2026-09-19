"""
Phase 21 Held-Out Evaluation and Edit Distance Metric Tests.
"""

from signova.operations.phase21_orchestrator import Phase21Orchestrator


def test_edit_distance_and_error_breakdown():
    orch = Phase21Orchestrator()
    ref = ["HELLO", "WORLD", "ISL"]
    hyp = ["HELLO", "SIGN", "ISL", "EXTRA"]

    dist, ins, dels, subs = orch._compute_token_edit_distance(ref, hyp)
    assert dist == 2
    assert ins == 1
    assert subs == 1
    assert dels == 0


def test_exact_sequence_match():
    orch = Phase21Orchestrator()
    ref = ["NAMASTE", "THANK_YOU"]
    hyp = ["NAMASTE", "THANK_YOU"]

    dist, ins, dels, subs = orch._compute_token_edit_distance(ref, hyp)
    assert dist == 0
    assert ins == 0
    assert dels == 0
    assert subs == 0
