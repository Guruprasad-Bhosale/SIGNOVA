"""
Phase 14 Quality Validation Unit Tests.
"""

from signova.annotation.constants import (
    QUALITY_LINGUIST_REVIEWED,
    QUALITY_VERIFIED,
    REVIEW_STATE_VERIFIED,
)
from signova.annotation.quality import calculate_annotation_quality_grade
from signova.annotation.schema import VideoAnnotation


def test_quality_requires_human_reviewer_evidence():
    a = VideoAnnotation(
        annotation_id="a1",
        sample_id="s1",
        annotator_id="u1",
        reviewer_id="reviewer_expert_01",
        is_temporally_aligned=False,
        glosses=["NAMASTE"],
        review_status=REVIEW_STATE_VERIFIED,
    )
    grade = calculate_annotation_quality_grade(a, is_linguist_review=True)
    assert grade == QUALITY_LINGUIST_REVIEWED
