"""
Phase 14 Metrics Unit Tests.
"""

from signova.experiments.metrics import evaluate_sequence_predictions


def test_sequence_metrics_breakdown():
    refs = [["NAMASTE", "WELCOME"]]
    hyps = [["NAMASTE", "FRIEND"]]

    res = evaluate_sequence_predictions(refs, hyps)
    assert res.substitutions == 1
    assert res.insertions == 0
    assert res.deletions == 0
    assert res.exact_sequence_match == 0.0
    assert res.ter == 0.5
