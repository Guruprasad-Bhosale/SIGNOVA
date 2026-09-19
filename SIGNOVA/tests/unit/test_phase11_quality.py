"""
Unit tests for Phase 11 Quality Grade Calculation and Human Review Requirements.
"""

import pytest
from signova.annotation.constants import (
    QUALITY_LINGUIST_REVIEWED,
    QUALITY_PARTIAL,
    QUALITY_VERIFIED,
    QUALITY_WEAK,
    REVIEW_STATE_REVIEW_PENDING,
    REVIEW_STATE_VERIFIED,
)
from signova.annotation.quality import calculate_annotation_quality_grade
from signova.annotation.schema import VideoAnnotation
from signova.annotation.validation import evaluate_training_eligibility


def test_quality_unreviewed_is_partial_not_verified():
    # Structurally valid annotation without reviewer evidence
    annot = VideoAnnotation(
        annotation_id="a_struct",
        sample_id="s_struct",
        annotator_id="u_01",
        is_temporally_aligned=False,
        glosses=["NAMASTE", "COLLEGE"],
        review_status=REVIEW_STATE_REVIEW_PENDING,
        reviewer_id=None,
    )
    grade = calculate_annotation_quality_grade(annot)
    assert grade == QUALITY_PARTIAL, "Structurally valid unreviewed annotation must be PARTIAL, not VERIFIED"

    eligible, reasons = evaluate_training_eligibility(annot)
    assert eligible is False
    assert "REVIEW_STATUS_NOT_VERIFIED" in reasons[0]


def test_quality_reviewed_becomes_verified():
    annot = VideoAnnotation(
        annotation_id="a_rev",
        sample_id="s_rev",
        annotator_id="u_01",
        is_temporally_aligned=False,
        glosses=["NAMASTE", "COLLEGE"],
        review_status=REVIEW_STATE_VERIFIED,
        reviewer_id="linguist_01",
        dataset_split="train",
    )
    grade = calculate_annotation_quality_grade(annot, is_linguist_review=True)
    assert grade == QUALITY_LINGUIST_REVIEWED
    annot.quality_grade = grade

    eligible, reasons = evaluate_training_eligibility(annot)
    assert eligible is True
    assert len(reasons) == 0
