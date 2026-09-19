"""
Phase 14 Annotation Collection Lifecycle & Existence vs Eligibility Tests.
"""

from signova.annotation.constants import (
    QUALITY_PARTIAL,
    QUALITY_VERIFIED,
    REVIEW_STATE_ANNOTATION_IN_PROGRESS,
    REVIEW_STATE_REVIEW_PENDING,
    REVIEW_STATE_VERIFIED,
)
from signova.annotation.schema import VideoAnnotation
from signova.annotation.validation import evaluate_training_eligibility


def test_annotation_existence_does_not_imply_training_eligibility():
    # Structurally valid draft exists
    draft = VideoAnnotation(
        annotation_id="draft_01",
        sample_id="sample_01",
        annotator_id="annot_01",
        is_temporally_aligned=False,
        glosses=["NAMASTE", "HELP"],
        review_status=REVIEW_STATE_ANNOTATION_IN_PROGRESS,
        quality_grade=QUALITY_PARTIAL,
        dataset_split="train",
    )
    eligible, reasons = evaluate_training_eligibility(draft)
    assert eligible is False
    assert any("REVIEW_STATUS_NOT_VERIFIED" in r for r in reasons)


def test_annotation_becomes_eligible_only_after_linguistic_review():
    verified_annot = VideoAnnotation(
        annotation_id="annot_01",
        sample_id="sample_01",
        annotator_id="annot_01",
        reviewer_id="reviewer_lead_01",
        is_temporally_aligned=False,
        glosses=["NAMASTE", "HELP"],
        review_status=REVIEW_STATE_VERIFIED,
        quality_grade=QUALITY_VERIFIED,
        dataset_split="train",
    )
    eligible, reasons = evaluate_training_eligibility(verified_annot)
    assert eligible is True
    assert len(reasons) == 0
