"""
Phase 13 Sequence Evaluation Metrics Unit Tests.
"""

from signova.experiments.metrics import evaluate_sequence_predictions


def test_sequence_metrics_calculation():
    refs = [["A", "B", "C"], ["D", "E"]]
    hyps = [["A", "B", "C"], ["D", "F"]]

    res = evaluate_sequence_predictions(refs, hyps)
    assert res.sample_count == 2
    assert res.substitutions == 1
    assert res.insertions == 0
    assert res.deletions == 0
    assert res.exact_sequence_match == 0.5
    assert res.ter == 1 / 5
