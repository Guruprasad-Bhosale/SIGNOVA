"""
Phase 12 Sequence Metrics Unit Tests.
"""

from signova.experiments.metrics import (
    compute_levenshtein_breakdown,
    evaluate_sequence_predictions,
)


def test_levenshtein_breakdown_exact_match():
    ref = ["HELLO", "WORLD"]
    hyp = ["HELLO", "WORLD"]
    s, i, d, dist = compute_levenshtein_breakdown(ref, hyp)
    assert s == 0
    assert i == 0
    assert d == 0
    assert dist == 0


def test_levenshtein_breakdown_subs_ins_dels():
    ref = ["A", "B", "C"]
    hyp = ["A", "X", "C", "D"]  # 1 substitution (B->X), 1 insertion (D)
    s, i, d, dist = compute_levenshtein_breakdown(ref, hyp)
    assert s == 1
    assert i == 1
    assert d == 0
    assert dist == 2


def test_evaluate_sequence_predictions():
    refs = [["HELLO", "WORLD"], ["THANK", "YOU"]]
    hyps = [["HELLO", "WORLD"], ["THANK", "ME"]]

    res = evaluate_sequence_predictions(refs, hyps)
    assert res.sample_count == 2
    assert res.exact_sequence_match == 0.5  # 1 of 2 matches exactly
    assert res.substitutions == 1
    assert res.insertions == 0
    assert res.deletions == 0
    assert res.total_ref_tokens == 4
    assert res.ter == 1 / 4
