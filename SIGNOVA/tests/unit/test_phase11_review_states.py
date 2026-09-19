"""
Unit tests for Phase 11 Review State Machine and Full Session Workflow.
Workflow: OPEN -> DRAFT -> SAVE -> RESUME -> SUBMIT -> REVIEW -> VERIFY/REJECT
"""

import pytest
from signova.annotation.constants import (
    REVIEW_STATE_ANNOTATION_IN_PROGRESS,
    REVIEW_STATE_REJECTED,
    REVIEW_STATE_REVIEW_PENDING,
    REVIEW_STATE_UNANNOTATED,
    REVIEW_STATE_VERIFIED,
)
from signova.annotation.review import (
    AnnotationStateMachine,
    AnnotationWorkflowError,
)
from signova.annotation.schema import VideoAnnotation


def test_session_workflow_full_lifecycle():
    annot = VideoAnnotation(
        annotation_id="a_01",
        sample_id="sample_01",
        annotator_id="deaf_ann_01",
        is_temporally_aligned=False,
        glosses=[],
        review_status=REVIEW_STATE_UNANNOTATED,
    )

    # 1. Start Draft (OPEN -> DRAFT)
    AnnotationStateMachine.start_draft(annot)
    assert annot.review_status == REVIEW_STATE_ANNOTATION_IN_PROGRESS

    # Add glosses
    annot.glosses = ["NAMASTE", "COLLEGE", "GO"]

    # 2. Save Draft (DRAFT -> SAVE)
    AnnotationStateMachine.save_draft(annot)
    assert annot.review_status == REVIEW_STATE_ANNOTATION_IN_PROGRESS

    # 3. Resume Draft (RESUME)
    AnnotationStateMachine.resume_draft(annot)
    assert annot.review_status == REVIEW_STATE_ANNOTATION_IN_PROGRESS

    # 4. Submit for Review (SUBMIT)
    AnnotationStateMachine.submit_for_review(annot)
    assert annot.review_status == REVIEW_STATE_REVIEW_PENDING
    assert annot.training_eligible is False

    # 5. Reviewer Verifies (REVIEW -> VERIFY)
    AnnotationStateMachine.verify_by_reviewer(
        annot, reviewer_id="linguist_rev_01", reviewer_notes="Approved."
    )
    assert annot.review_status == REVIEW_STATE_VERIFIED
    assert annot.reviewer_id == "linguist_rev_01"


def test_submit_empty_annotation_raises_error():
    annot = VideoAnnotation(
        annotation_id="a_empty",
        sample_id="s_empty",
        annotator_id="u_01",
        is_temporally_aligned=False,
        glosses=[],
        review_status=REVIEW_STATE_ANNOTATION_IN_PROGRESS,
    )
    with pytest.raises(AnnotationWorkflowError) as excinfo:
        AnnotationStateMachine.submit_for_review(annot)
    assert "Cannot submit an empty gloss sequence" in str(excinfo.value)


def test_reviewer_rejection_workflow():
    annot = VideoAnnotation(
        annotation_id="a_rej",
        sample_id="s_rej",
        annotator_id="u_01",
        is_temporally_aligned=False,
        glosses=["WRONG_SIGN"],
        review_status=REVIEW_STATE_REVIEW_PENDING,
    )
    AnnotationStateMachine.reject_by_reviewer(
        annot, reviewer_id="linguist_01", rejection_reason="Sign gloss inaccurate."
    )
    assert annot.review_status == REVIEW_STATE_REJECTED
    assert annot.training_eligible is False
