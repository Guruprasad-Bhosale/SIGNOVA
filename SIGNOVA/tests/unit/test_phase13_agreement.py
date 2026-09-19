"""
Phase 13 Agreement Unit Tests.
"""

from signova.annotation.agreement import compute_sequence_token_f1
from signova.annotation.schema import VideoAnnotation


def test_agreement_token_f1_exact():
    seq1 = ["NAMASTE", "HELP"]
    seq2 = ["NAMASTE", "HELP"]
    p, r, f1 = compute_sequence_token_f1(seq1, seq2)
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0


def test_agreement_token_f1_divergent():
    seq1 = ["NAMASTE", "HELP"]
    seq2 = ["NAMASTE", "WATER"]
    p, r, f1 = compute_sequence_token_f1(seq1, seq2)
    assert p == 0.5
    assert r == 0.5
    assert f1 == 0.5


def test_agreement_empty_safety():
    p, r, f1 = compute_sequence_token_f1([], [])
    assert f1 == 1.0
