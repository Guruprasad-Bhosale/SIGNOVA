"""
Unit tests for Phase 9 Human Annotation Agreement & Evaluation Module.
"""

import pytest
from signova.data.human_annotation import (
    compute_levenshtein_distance,
    evaluate_inter_annotator_agreement,
)


def test_levenshtein_distance_tokens():
    seq1 = ["I", "GO", "COLLEGE", "TODAY"]
    seq2 = ["I", "GO", "COLLEGE", "TODAY"]
    dist, s, ins, d = compute_levenshtein_distance(seq1, seq2)
    assert dist == 0
    assert s == 0 and ins == 0 and d == 0

    seq3 = ["I", "WALK", "COLLEGE"]
    dist, s, ins, d = compute_levenshtein_distance(seq1, seq3)
    assert dist == 2  # substitution of GO->WALK, deletion of TODAY
    assert s == 1
    assert d == 1


def test_empty_dual_annotations_safety():
    # When no dual human annotations exist, report status safely without fabricating data
    rep = evaluate_inter_annotator_agreement([])
    assert rep.dual_annotated_sample_count == 0
    assert rep.status == "NO_DUAL_ANNOTATIONS_AVAILABLE"
    assert rep.cohens_kappa is None


def test_dual_annotations_evaluation_with_data():
    paired_data = [
        (["I", "GO", "COLLEGE"], ["I", "GO", "COLLEGE"]),
        (["NAMASTE", "MY", "NAME"], ["NAMASTE", "NAME"]),
    ]
    rep = evaluate_inter_annotator_agreement(paired_data)
    assert rep.dual_annotated_sample_count == 2
    assert rep.status == "EVALUATED_FROM_HUMAN_DATA"
    assert rep.exact_sequence_match_rate == 0.5
    assert rep.mean_token_error_rate > 0.0
