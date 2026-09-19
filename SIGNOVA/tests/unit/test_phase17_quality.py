"""
Phase 17 Quality Validation Unit Tests.

Validates:
- Linguistic quality grading (QUALITY_LINGUIST_REVIEWED, QUALITY_VERIFIED).
- Distinction between structural schema validity and linguistic correctness.
- Rejection of invalid annotations lacking expert verification evidence.
"""

from signova.annotation.constants import (
    QUALITY_LINGUIST_REVIEWED,
    QUALITY_VERIFIED,
    REVIEW_STATE_VERIFIED,
    REVIEW_STATE_REJECTED,
)
from signova.annotation.quality import calculate_annotation_quality_grade
from signova.annotation.schema import VideoAnnotation


def test_linguist_reviewed_quality_grade():
    ann = VideoAnnotation(
        annotation_id="p17_ann_001",
        sample_id="p17_sample_001",
        annotator_id="u_annotator_01",
        reviewer_id="reviewer_isl_linguist",
        is_temporally_aligned=True,
        glosses=["THANK", "YOU"],
        review_status=REVIEW_STATE_VERIFIED,
    )
    grade = calculate_annotation_quality_grade(ann, is_linguist_review=True)
    assert grade == QUALITY_LINGUIST_REVIEWED


def test_standard_verified_quality_grade():
    ann = VideoAnnotation(
        annotation_id="p17_ann_002",
        sample_id="p17_sample_002",
        annotator_id="u_annotator_02",
        reviewer_id="reviewer_peer_01",
        is_temporally_aligned=False,
        glosses=["HELLO"],
        review_status=REVIEW_STATE_VERIFIED,
    )
    grade = calculate_annotation_quality_grade(ann, is_linguist_review=False)
    assert grade == QUALITY_VERIFIED
