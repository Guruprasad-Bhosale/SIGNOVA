"""
Phase 6 Sequence Evaluation & Levenshtein Metrics Tests for SIGNOVA.

Verifies:
1. Exact sequence match evaluation.
2. Edit distance breakdown (Substitutions, Deletions, Insertions).
3. Token Error Rate (TER) mathematical properties.
4. Token Macro F1 calculations.
"""

import pytest
from signova.evaluation.sequence_metrics import compute_levenshtein_alignment, compute_sequence_metrics


def test_levenshtein_exact_match():
    ref = ["HELLO", "WORLD", "PLEASE"]
    hyp = ["HELLO", "WORLD", "PLEASE"]
    dist, s, d, i = compute_levenshtein_alignment(ref, hyp)
    assert dist == 0
    assert s == 0 and d == 0 and i == 0


def test_levenshtein_substitutions_insertions_deletions():
    # 1 substitution
    dist_s, s, d, i = compute_levenshtein_alignment(["A", "B"], ["A", "C"])
    assert dist_s == 1
    assert s == 1 and d == 0 and i == 0

    # 1 deletion
    dist_d, s, d, i = compute_levenshtein_alignment(["A", "B"], ["A"])
    assert dist_d == 1
    assert s == 0 and d == 1 and i == 0

    # 1 insertion
    dist_i, s, d, i = compute_levenshtein_alignment(["A"], ["A", "B"])
    assert dist_i == 1
    assert s == 0 and d == 0 and i == 1


def test_compute_sequence_metrics_batch():
    refs = [
        ["NAMASTE", "WELCOME"],
        ["GOODBYE", "FRIEND"],
        ["PLEASE", "HELP"],
    ]
    hyps = [
        ["NAMASTE", "WELCOME"],  # Exact match
        ["GOODBYE"],             # 1 deletion
        ["PLEASE", "ASSIST"],    # 1 substitution
    ]
    metrics = compute_sequence_metrics(refs, hyps)

    assert metrics["total_sequences"] == 3
    assert metrics["exact_matches_count"] == 1
    assert metrics["exact_match_rate"] == round(1.0 / 3.0, 4)
    assert metrics["deletions"] == 1
    assert metrics["substitutions"] == 1
    assert metrics["insertions"] == 0
    assert metrics["total_edit_distance"] == 2
    assert metrics["token_error_rate"] == round(2.0 / 6.0, 4)
    assert 0.0 <= metrics["token_macro_f1"] <= 1.0
