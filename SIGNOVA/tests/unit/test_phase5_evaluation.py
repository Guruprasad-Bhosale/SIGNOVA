"""
Unit Tests for Phase 5 Sequence Evaluation Metrics (TER, Edit Distance, Precision/Recall/F1).
"""

import pytest

from signova.evaluation.sequence_metrics import compute_levenshtein_alignment, compute_sequence_metrics


def test_levenshtein_alignment_breakdown():
    ref = ["HELLO", "WORLD", "ISL"]
    hyp = ["HELLO", "SIGN", "ISL", "EXTRA"]

    dist, subs, dels, ins = compute_levenshtein_alignment(ref, hyp)
    assert subs == 1  # WORLD -> SIGN
    assert ins == 1   # EXTRA inserted
    assert dels == 0
    assert dist == 2


def test_sequence_metrics_batch_evaluation():
    refs = [
        ["HELLO", "WORLD"],
        ["NAME", "YOUR", "WHAT"],
    ]
    hyps = [
        ["HELLO", "WORLD"],          # Exact match
        ["NAME", "MY", "WHAT"],      # 1 Substitution
    ]

    metrics = compute_sequence_metrics(refs, hyps)
    assert metrics["total_sequences"] == 2
    assert metrics["exact_matches_count"] == 1
    assert metrics["exact_match_rate"] == 0.5
    assert metrics["substitutions"] == 1
    assert metrics["insertions"] == 0
    assert metrics["deletions"] == 0
    assert metrics["total_edit_distance"] == 1
    # TER = 1 / 5 = 0.20
    assert metrics["token_error_rate"] == 0.20
    assert "token_macro_f1" in metrics


def test_sequence_metrics_empty_safety():
    metrics = compute_sequence_metrics([], [])
    assert "error" in metrics
