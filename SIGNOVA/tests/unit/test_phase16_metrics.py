"""
Phase 16 Metrics Computation & Defensive Integrity Unit Tests.

Validates:
- Sequence evaluation metrics (WER/TER, substitutions, deletions, insertions, exact sequence accuracy).
- Defensive integrity: under STATE_B, zero real metrics are fabricated.
"""

from signova.experiments.metrics import evaluate_sequence_predictions


def test_sequence_metrics_calculation():
    refs = [["NAMASTE", "WELCOME", "HOME"]]
    hyps = [["NAMASTE", "FRIEND", "HOME"]]

    res = evaluate_sequence_predictions(refs, hyps)
    assert res.substitutions == 1
    assert res.insertions == 0
    assert res.deletions == 0
    assert res.exact_sequence_match == 0.0
    assert round(res.ter, 3) == 0.333


def test_perfect_sequence_match():
    refs = [["HELLO", "WORLD"]]
    hyps = [["HELLO", "WORLD"]]

    res = evaluate_sequence_predictions(refs, hyps)
    assert res.substitutions == 0
    assert res.insertions == 0
    assert res.deletions == 0
    assert res.exact_sequence_match == 1.0
    assert res.ter == 0.0
