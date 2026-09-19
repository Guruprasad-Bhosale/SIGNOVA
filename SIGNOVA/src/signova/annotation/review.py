"""
Phase 11 Annotation Session Workflow and Review State Machine for SIGNOVA.

Enforces:
OPEN -> DRAFT -> SAVE -> RESUME -> SUBMIT -> REVIEW -> VERIFY / REJECT

State Transitions:
- UNANNOTATED -> (open/draft) -> ANNOTATION_IN_PROGRESS
- ANNOTATION_IN_PROGRESS -> (save) -> ANNOTATION_IN_PROGRESS
- ANNOTATION_IN_PROGRESS -> (submit) -> REVIEW_PENDING
- REVIEW_PENDING -> (linguist approval) -> VERIFIED
- REVIEW_PENDING -> (rejection) -> REJECTED
- SECOND_ANNOTATION_PENDING -> (independent annotation added) -> DISAGREEMENT or VERIFIED
"""

from typing import Optional
from signova.annotation.constants import (
    REVIEW_STATE_ANNOTATED,
    REVIEW_STATE_ANNOTATION_IN_PROGRESS,
    REVIEW_STATE_DISAGREEMENT,
    REVIEW_STATE_REJECTED,
    REVIEW_STATE_REVIEW_PENDING,
    REVIEW_STATE_SECOND_ANNOTATION_PENDING,
    REVIEW_STATE_UNANNOTATED,
    REVIEW_STATE_VERIFIED,
)
from signova.annotation.schema import VideoAnnotation


class AnnotationWorkflowError(Exception):
    """Raised when an invalid state transition is attempted in the annotation workflow."""
    pass


class AnnotationStateMachine:
    """
    Manages explicit state transitions for human annotation sessions.
    """

    @staticmethod
    def start_draft(annotation: VideoAnnotation) -> VideoAnnotation:
        if annotation.review_status not in {REVIEW_STATE_UNANNOTATED, REVIEW_STATE_REJECTED}:
            raise AnnotationWorkflowError(
                f"Cannot start draft from state '{annotation.review_status}'. Must be UNANNOTATED or REJECTED."
            )
        annotation.review_status = REVIEW_STATE_ANNOTATION_IN_PROGRESS
        annotation.training_eligible = False
        return annotation

    @staticmethod
    def save_draft(annotation: VideoAnnotation) -> VideoAnnotation:
        if annotation.review_status != REVIEW_STATE_ANNOTATION_IN_PROGRESS:
            raise AnnotationWorkflowError(
                f"Cannot save draft in state '{annotation.review_status}'. Draft must be IN_PROGRESS."
            )
        # Keeps status in ANNOTATION_IN_PROGRESS
        annotation.training_eligible = False
        return annotation

    @staticmethod
    def resume_draft(annotation: VideoAnnotation) -> VideoAnnotation:
        if annotation.review_status != REVIEW_STATE_ANNOTATION_IN_PROGRESS:
            raise AnnotationWorkflowError(
                f"Cannot resume draft in state '{annotation.review_status}'."
            )
        return annotation

    @staticmethod
    def submit_for_review(annotation: VideoAnnotation) -> VideoAnnotation:
        if annotation.review_status != REVIEW_STATE_ANNOTATION_IN_PROGRESS:
            raise AnnotationWorkflowError(
                f"Cannot submit for review from state '{annotation.review_status}'. Must be IN_PROGRESS."
            )
        if not annotation.glosses or len(annotation.glosses) == 0:
            raise AnnotationWorkflowError("Cannot submit an empty gloss sequence for review.")

        annotation.review_status = REVIEW_STATE_REVIEW_PENDING
        annotation.training_eligible = False  # Not eligible until reviewed
        return annotation

    @staticmethod
    def verify_by_reviewer(
        annotation: VideoAnnotation,
        reviewer_id: str,
        reviewer_notes: str = "Verified by Deaf ISL Linguist.",
        is_linguist: bool = True,
    ) -> VideoAnnotation:
        if annotation.review_status != REVIEW_STATE_REVIEW_PENDING:
            raise AnnotationWorkflowError(
                f"Cannot verify annotation from state '{annotation.review_status}'. Must be REVIEW_PENDING."
            )
        annotation.review_status = REVIEW_STATE_VERIFIED
        annotation.reviewer_id = reviewer_id
        annotation.reviewer_notes = reviewer_notes
        # Note: training eligibility is calculated by validation module, not set arbitrarily
        return annotation

    @staticmethod
    def reject_by_reviewer(
        annotation: VideoAnnotation,
        reviewer_id: str,
        rejection_reason: str,
    ) -> VideoAnnotation:
        if annotation.review_status != REVIEW_STATE_REVIEW_PENDING:
            raise AnnotationWorkflowError(
                f"Cannot reject annotation from state '{annotation.review_status}'. Must be REVIEW_PENDING."
            )
        annotation.review_status = REVIEW_STATE_REJECTED
        annotation.reviewer_id = reviewer_id
        annotation.reviewer_notes = rejection_reason
        annotation.training_eligible = False
        return annotation
