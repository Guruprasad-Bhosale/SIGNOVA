"""
Phase 11 Quality Grade Calculation Engine for SIGNOVA.

Enforces:
Structural validity != Linguistic Verification.
An annotation receives VERIFIED / LINGUIST_REVIEWED only when explicit human reviewer evidence exists.
"""

from typing import Dict, List, Optional
from signova.annotation.constants import (
    QUALITY_LINGUIST_REVIEWED,
    QUALITY_PARTIAL,
    QUALITY_UNVERIFIED,
    QUALITY_VERIFIED,
    QUALITY_WEAK,
    REVIEW_STATE_VERIFIED,
)
from signova.annotation.schema import VideoAnnotation
from signova.data.quality_audit import validate_gloss_token


def calculate_annotation_quality_grade(
    annotation: VideoAnnotation,
    is_linguist_review: bool = False,
) -> str:
    """
    Computes deterministic quality grade based on structural and review evidence.
    """
    glosses = annotation.glosses
    if not glosses or len(glosses) == 0:
        return QUALITY_WEAK

    # Check token validity
    malformed = [g for g in glosses if not validate_gloss_token(str(g))]
    if malformed:
        return QUALITY_WEAK

    # Check temporal segment consistency if temporally aligned
    if annotation.is_temporally_aligned and annotation.segments:
        for seg in annotation.segments:
            if seg.start_frame is not None and seg.end_frame is not None:
                if seg.start_frame >= seg.end_frame:
                    return QUALITY_WEAK

    # Check review evidence
    has_reviewer = bool(annotation.reviewer_id and annotation.review_status == REVIEW_STATE_VERIFIED)

    if has_reviewer:
        return QUALITY_LINGUIST_REVIEWED if is_linguist_review else QUALITY_VERIFIED

    # Structurally valid but unreviewed
    return QUALITY_PARTIAL
