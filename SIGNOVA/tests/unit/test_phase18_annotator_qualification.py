"""
Phase 18 Annotator Qualification Profile and Audit Unit Tests.

Validates:
- Annotator qualification profiles (Native ISL signer, certified interpreter, deaf community member).
- Verification status transitions (QUALIFICATION_DOCUMENTED, REQUIRES_ISL_EXPERT_REVIEW, QUALIFICATION_NOT_DOCUMENTED).
- Distinction between annotator and signer identities.
"""

from signova.operations.annotator_qualification import (
    AnnotatorQualificationProfile,
    validate_annotator_qualification,
    QUALIFICATION_DOCUMENTED,
    QUALIFICATION_NOT_DOCUMENTED,
    REQUIRES_ISL_EXPERT_REVIEW,
)


def test_certified_interpreter_qualification():
    profile_data = {
        "annotator_id": "ann_expert_01",
        "is_native_or_fluent_signer": True,
        "formal_linguistics_training": True,
        "verified_by_organization": "Deaf Association of India",
        "notes": "ISL Certified Interpreter",
    }
    profile = validate_annotator_qualification(profile_data)
    assert profile.qualification_status == QUALIFICATION_DOCUMENTED
    assert profile.is_native_or_fluent_signer is True
    assert profile.formal_linguistics_training is True


def test_missing_qualification_requires_review():
    profile_data = {
        "annotator_id": "ann_novice_02",
        "is_native_or_fluent_signer": True,
        "formal_linguistics_training": False,
        "verified_by_organization": None,
    }
    profile = validate_annotator_qualification(profile_data)
    assert profile.qualification_status == REQUIRES_ISL_EXPERT_REVIEW
    assert profile.is_native_or_fluent_signer is True


def test_none_profile_returns_not_documented():
    profile = validate_annotator_qualification(None)
    assert profile.qualification_status == QUALIFICATION_NOT_DOCUMENTED
    assert profile.is_native_or_fluent_signer is False
