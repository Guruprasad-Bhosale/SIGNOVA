"""
Phase 15 Annotator Qualification & Linguistic Expertise Validator for SIGNOVA.

Enforces:
- annotator qualification must be explicitly documented
- qualification is NEVER inferred from name, email, UID prefix, or project membership
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

QUALIFICATION_DOCUMENTED = "QUALIFICATION_DOCUMENTED"
QUALIFICATION_NOT_DOCUMENTED = "QUALIFICATION_NOT_DOCUMENTED"
REQUIRES_ISL_EXPERT_REVIEW = "REQUIRES_ISL_EXPERT_REVIEW"


@dataclass
class AnnotatorQualificationProfile:
    annotator_id: str
    qualification_status: str  # QUALIFICATION_DOCUMENTED, QUALIFICATION_NOT_DOCUMENTED, REQUIRES_ISL_EXPERT_REVIEW
    is_native_or_fluent_signer: bool
    formal_linguistics_training: bool
    verified_by_organization: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotator_id": self.annotator_id,
            "qualification_status": self.qualification_status,
            "is_native_or_fluent_signer": self.is_native_or_fluent_signer,
            "formal_linguistics_training": self.formal_linguistics_training,
            "verified_by_organization": self.verified_by_organization,
            "notes": self.notes,
        }


def validate_annotator_qualification(
    profile_data: Optional[Dict[str, Any]],
) -> AnnotatorQualificationProfile:
    """
    Validates that annotator qualification is explicitly recorded rather than inferred.
    """
    if not profile_data or not profile_data.get("annotator_id"):
        return AnnotatorQualificationProfile(
            annotator_id="UNKNOWN",
            qualification_status=QUALIFICATION_NOT_DOCUMENTED,
            is_native_or_fluent_signer=False,
            formal_linguistics_training=False,
            notes="No explicit annotator qualification metadata provided.",
        )

    ann_id = profile_data["annotator_id"]
    is_fluent = bool(profile_data.get("is_native_or_fluent_signer", False))
    has_training = bool(profile_data.get("formal_linguistics_training", False))
    org = profile_data.get("verified_by_organization")

    if is_fluent and (has_training or org):
        status = QUALIFICATION_DOCUMENTED
    elif is_fluent:
        status = REQUIRES_ISL_EXPERT_REVIEW
    else:
        status = QUALIFICATION_NOT_DOCUMENTED

    return AnnotatorQualificationProfile(
        annotator_id=ann_id,
        qualification_status=status,
        is_native_or_fluent_signer=is_fluent,
        formal_linguistics_training=has_training,
        verified_by_organization=org,
        notes=profile_data.get("notes", ""),
    )
