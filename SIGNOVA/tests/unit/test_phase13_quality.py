"""
Phase 13 Quality Grading & Training Eligibility Unit Tests.
"""

from signova.annotation.constants import (
    QUALITY_LINGUIST_REVIEWED,
    QUALITY_PARTIAL,
    QUALITY_UNVERIFIED,
    QUALITY_VERIFIED,
    QUALITY_WEAK,
    REVIEW_STATE_ANNOTATION_IN_PROGRESS,
    REVIEW_STATE_VERIFIED,
)
from signova.annotation.quality import calculate_annotation_quality_grade
from signova.annotation.schema import VideoAnnotation
from signova.annotation.validation import evaluate_training_eligibility


def test_quality_grade_tiers():
    a1 = VideoAnnotation(
        annotation_id="a1", sample_id="s1", annotator_id="u1", is_temporally_aligned=False,
        glosses=[], review_status=REVIEW_STATE_ANNOTATION_IN_PROGRESS
    )
    assert calculate_annotation_quality_grade(a1) == QUALITY_WEAK

    a2 = VideoAnnotation(
        annotation_id="a2", sample_id="s2", annotator_id="u1", is_temporally_aligned=False,
        glosses=["NAMASTE"], review_status=REVIEW_STATE_ANNOTATION_IN_PROGRESS
    )
    assert calculate_annotation_quality_grade(a2) == QUALITY_PARTIAL

    a3 = VideoAnnotation(
        annotation_id="a3", sample_id="s3", annotator_id="u1", reviewer_id="r1", is_temporally_aligned=False,
        glosses=["NAMASTE"], review_status=REVIEW_STATE_VERIFIED
    )
    assert calculate_annotation_quality_grade(a3) == QUALITY_VERIFIED
    assert calculate_annotation_quality_grade(a3, is_linguist_review=True) == QUALITY_LINGUIST_REVIEWED


def test_training_eligibility_evaluation():
    a = VideoAnnotation(
        annotation_id="a1", sample_id="s1", annotator_id="u1", reviewer_id="r1",
        is_temporally_aligned=False, glosses=["NAMASTE"], review_status=REVIEW_STATE_VERIFIED,
        quality_grade=QUALITY_VERIFIED, dataset_split="train"
    )
    eligible, reasons = evaluate_training_eligibility(a)
    assert eligible is True
    assert len(reasons) == 0
