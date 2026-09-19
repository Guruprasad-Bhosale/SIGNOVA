"""
Phase 21 Error Analysis and Classification Tests.
"""

from signova.operations.phase21_orchestrator import Phase21Orchestrator


def test_insertion_and_deletion_counts():
    orch = Phase21Orchestrator()
    ref = ["A", "B", "C"]
    hyp = ["A", "C"]  # Deletion of B
    dist, ins, dels, subs = orch._compute_token_edit_distance(ref, hyp)
    assert dist == 1
    assert dels == 1
    assert ins == 0
    assert subs == 0

    hyp_ins = ["A", "X", "B", "C"]  # Insertion of X
    dist2, ins2, dels2, subs2 = orch._compute_token_edit_distance(ref, hyp_ins)
    assert dist2 == 1
    assert ins2 == 1
    assert dels2 == 0
    assert subs2 == 0
