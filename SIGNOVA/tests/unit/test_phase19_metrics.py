"""
Phase 19 Held-out Test Metrics Unit Tests.

Validates:
- Structured blocked reporting when no real supervised predictions exist.
- Calculation structures for Token Error Rate (TER), substitutions, insertions, deletions.
"""

from signova.experiments.metrics import evaluate_sequence_predictions


def test_token_error_rate_calculation():
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
