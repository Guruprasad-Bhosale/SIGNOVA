"""
Phase 11 Deterministic Training Eligibility Validator for SIGNOVA.

Calculates training eligibility from evidence rather than manual flags.
"""

from typing import Dict, List, Tuple
from signova.annotation.constants import (
    QUALITY_LINGUIST_REVIEWED,
    QUALITY_VERIFIED,
    REVIEW_STATE_VERIFIED,
)
from signova.annotation.schema import VideoAnnotation
from signova.data.quality_audit import validate_gloss_token


def evaluate_training_eligibility(annotation: VideoAnnotation) -> Tuple[bool, List[str]]:
    """
    Evaluates whether an annotation record satisfies all scientific and structural requirements
    to become eligible for model training.
    """
    reasons = []

    # 1. Non-empty gloss sequence
    if not annotation.glosses or len(annotation.glosses) == 0:
        reasons.append("EMPTY_GLOSS_SEQUENCE")

    # 2. Token validity
    malformed = [g for g in annotation.glosses if not validate_gloss_token(str(g))]
    if malformed:
        reasons.append(f"MALFORMED_TOKENS: {malformed[:3]}")

    # 3. Review state must be VERIFIED
    if annotation.review_status != REVIEW_STATE_VERIFIED:
        reasons.append(f"REVIEW_STATUS_NOT_VERIFIED (current: {annotation.review_status})")

    # 4. Reviewer must be recorded
    if not annotation.reviewer_id:
        reasons.append("MISSING_REVIEWER_ID")

    # 5. Quality grade must be VERIFIED or LINGUIST_REVIEWED
    if annotation.quality_grade not in {QUALITY_VERIFIED, QUALITY_LINGUIST_REVIEWED}:
        reasons.append(f"INSUFFICIENT_QUALITY_GRADE (current: {annotation.quality_grade})")

    # 6. Valid dataset split
    if annotation.dataset_split not in {"train", "val", "test"}:
        reasons.append(f"INVALID_DATASET_SPLIT (current: {annotation.dataset_split})")

    # 7. Temporal consistency if temporally aligned
    if annotation.is_temporally_aligned and annotation.segments:
        for seg in annotation.segments:
            if seg.start_frame is not None and seg.end_frame is not None:
                if seg.start_frame >= seg.end_frame:
                    reasons.append(f"INVALID_TEMPORAL_BOUNDARIES: start {seg.start_frame} >= end {seg.end_frame}")
                    break

    eligible = (len(reasons) == 0)
    annotation.training_eligible = eligible
    return eligible, reasons
