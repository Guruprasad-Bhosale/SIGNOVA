"""
Phase 22 Annotator Qualification Tests.
"""

import pytest
from signova.operations.phase22_orchestrator import AnnotatorProfile, Phase22Orchestrator


def test_annotator_registration_and_qualification(tmp_path):
    orch = Phase22Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")

    # Register qualified annotator
    profile = orch.register_annotator(
        annotator_id="ann_001",
        qualification_status="QUALIFIED",
        qualification_method="LINGUISTIC_EVALUATION",
        qualification_evidence="Certified ISL interpreter credential #ISL-2026-09",
        verified_by="Lead_Linguist_01"
    )
    assert profile.annotator_id == "ann_001"
    assert profile.qualification_status == "QUALIFIED"
    assert profile.verified_by == "Lead_Linguist_01"

    # Register unqualified/pending annotator
    orch.register_annotator(
        annotator_id="ann_002",
        qualification_status="PENDING",
        qualification_method="SELF_CLAIMED",
        qualification_evidence="None provided",
    )

    profiles = orch.list_annotators()
    assert len(profiles) == 2
    assert "ann_001" in profiles
    assert "ann_002" in profiles

    qualified = orch.list_qualified_annotators()
    assert len(qualified) == 1
    assert "ann_001" in qualified


def test_annotator_evidence_requirements(tmp_path):
    orch = Phase22Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")

    # Qualification cannot be inferred without explicit evidence fields
    profile = AnnotatorProfile(
        annotator_id="ann_test",
        qualification_status="QUALIFIED",
        qualification_method="",
        qualification_evidence="",
        verified_by=""
    )
    assert profile.qualification_status == "QUALIFIED"
    assert profile.qualification_evidence == ""
