"""
Unit tests for Phase 11 Inter-Annotator Agreement and Disagreement Taxonomy.
"""

import pytest
from signova.annotation.agreement import (
    classify_phase11_disagreement,
    compute_sequence_token_f1,
)
from signova.annotation.constants import (
    DISAGREEMENT_EXTRA_SIGN,
    DISAGREEMENT_OMITTED_SIGN,
    DISAGREEMENT_TOKEN_IDENTITY,
    DISAGREEMENT_TOKEN_ORDER,
)
from signova.annotation.schema import VideoAnnotation


def test_token_f1_computation():
    seq1 = ["NAMASTE", "COLLEGE", "GO"]
    seq2 = ["NAMASTE", "COLLEGE", "GO"]
    p, r, f1 = compute_sequence_token_f1(seq1, seq2)
    assert p == 1.0 and r == 1.0 and f1 == 1.0

    seq3 = ["NAMASTE", "GO"]
    p, r, f1 = compute_sequence_token_f1(seq1, seq3)
    assert p < 1.0 or r < 1.0


def test_disagreement_classification_types():
    annot_a = VideoAnnotation(annotation_id="a1", sample_id="s1", annotator_id="A", is_temporally_aligned=False, glosses=["I", "GO"])
    annot_b = VideoAnnotation(annotation_id="a2", sample_id="s1", annotator_id="B", is_temporally_aligned=False, glosses=["I", "COLLEGE", "GO"])

    disagree = classify_phase11_disagreement("s1", annot_a, annot_b)
    assert disagree.disagreement_type == DISAGREEMENT_OMITTED_SIGN

    annot_c = VideoAnnotation(annotation_id="a3", sample_id="s1", annotator_id="C", is_temporally_aligned=False, glosses=["GO", "I"])
    disagree_order = classify_phase11_disagreement("s1", annot_a, annot_c)
    assert disagree_order.disagreement_type == DISAGREEMENT_TOKEN_ORDER
