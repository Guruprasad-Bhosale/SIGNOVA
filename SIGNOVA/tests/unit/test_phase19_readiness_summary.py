"""
Phase 19 Readiness Summary & Automation Audit Unit Tests.

Validates:
- Dynamic generation of phase19_readiness_summary without hardcoded supervision states.
- Code and filesystem audit for prohibited automated pseudo-labeling pathways.
- 7-tier sample accounting structure and pilot activity tracking.
"""

from pathlib import Path
from signova.operations.audit_automation import (
    audit_prohibited_automation_pathways,
)
from signova.operations.phase19_orchestrator import (
    Phase19Orchestrator,
    evaluate_phase19_readiness,
)


def test_prohibited_automation_audit_on_clean_repository(tmp_path):
    audit_res = audit_prohibited_automation_pathways(tmp_path)
    assert audit_res["audit_status"] == "PASSED"
    assert len(audit_res["violations"]) == 0
    assert audit_res["prohibited_pathways_detected"] is False


def test_dynamic_readiness_summary_generation(tmp_path):
    pilot_dir = tmp_path / "pilot_dir"
    pilot_dir.mkdir()

    orch = Phase19Orchestrator(annotations_dir=pilot_dir)
    summary = orch.run_full_qualification()

    assert "phase" in summary
    assert summary["phase"] == 19
    assert "supervision_state" in summary
    assert "real_ctc_status" in summary
    assert "pilot_status" in summary
    assert "annotation_activity_started" in summary
    assert "training_authorization" in summary
    assert "generalization_claims" in summary
    assert "publication_grade_evaluation" in summary
    assert "automation_audit_status" in summary
    assert summary["automation_audit_status"] == "PASSED"
    assert "sample_accounting" in summary
    assert summary["sample_accounting"]["total_discovered"] == 0
    assert "total_annotations" in summary["sample_accounting"]
    assert "training_eligible_samples" in summary["sample_accounting"]
    assert "pilot_configuration" in summary
    assert summary["pilot_configuration"]["pilot_target_samples"] == 20
    assert "ctc_feasibility" in summary


def test_canonical_evaluate_phase19_readiness_function(tmp_path):
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    res = evaluate_phase19_readiness(workspace_root=tmp_path, annotations_dir=ann_dir)
    assert res["phase"] == 19
    assert res["supervision_state"] == "STATE_B"
    assert res["real_ctc_training_allowed"] is False
    assert res["real_checkpoint_created"] is False
    assert "training_authorization" in res
    assert res["training_authorization"]["authorized"] is False
