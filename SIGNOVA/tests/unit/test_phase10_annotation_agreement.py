"""
Unit tests for Phase 10 Inter-Annotator Agreement and Disagreement Categorization.
"""

import json
from pathlib import Path
import pytest
from signova.data.human_annotation import (
    ALLOWED_DISAGREEMENT_TYPES,
    classify_annotation_disagreement,
    evaluate_inter_annotator_agreement,
)


def test_disagreement_classification_types():
    # Compound sign difference
    d_comp = classify_annotation_disagreement(
        "s1", "A", "B", ["ICE-CREAM", "GOOD"], ["ICE", "CREAM", "GOOD"]
    )
    assert d_comp.disagreement_type == "COMPOUND_SIGN"
    assert d_comp.disagreement_type in ALLOWED_DISAGREEMENT_TYPES

    # Fingerspelling difference
    d_fs = classify_annotation_disagreement(
        "s2", "A", "B", ["FS-DELHI", "VISIT"], ["CITY", "VISIT"]
    )
    assert d_fs.disagreement_type == "FINGERSPELLING"

    # Number difference
    d_num = classify_annotation_disagreement(
        "s3", "A", "B", ["NUM-5", "BOOK"], ["FIVE", "BOOK"]
    )
    assert d_num.disagreement_type == "NUMBER"


def test_interannotator_agreement_single_annotator_returns_not_computable():
    rep = evaluate_inter_annotator_agreement([])
    assert rep.status in {"NOT_COMPUTABLE", "NO_DUAL_ANNOTATIONS_AVAILABLE"}
    assert rep.dual_annotated_sample_count == 0
    assert rep.cohens_kappa is None


def test_interannotator_agreement_disagreement_tracking():
    pairs = [
        (["I", "GO", "COLLEGE"], ["I", "GO", "COLLEGE"]),
        (["ICE-CREAM", "LIKE"], ["ICE", "CREAM", "LIKE"]),
    ]
    rep = evaluate_inter_annotator_agreement(pairs)
    assert rep.status == "EVALUATED_FROM_HUMAN_DATA"
    assert rep.dual_annotated_sample_count == 2
    assert len(rep.disagreements) == 1
    assert rep.disagreements[0].disagreement_type == "COMPOUND_SIGN"
