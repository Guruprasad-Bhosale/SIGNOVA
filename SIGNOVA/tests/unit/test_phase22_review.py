"""
Phase 22 Annotation Review State Machine Tests.
"""

import pytest
from signova.operations.phase22_orchestrator import Phase22Orchestrator


def test_review_state_transitions(tmp_path):
    orch = Phase22Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.register_annotator("ann_01", qualification_status="QUALIFIED")

    orch.import_human_annotation(
        annotation_id="ann_review_1",
        video_id="vid_01",
        annotator_id="ann_01",
        gloss_sequence=["SIGN", "LANGUAGE"]
    )

    # Transition from SUBMITTED to VERIFIED
    res = orch.review_annotation("ann_review_1", "VERIFIED", reviewer_id="Supervisor_1")
    assert res["review_status"] == "VERIFIED"
    assert res["reviewed_by"] == "Supervisor_1"

    # Transition to REJECTED with reason
    res_rej = orch.review_annotation("ann_review_1", "REJECTED", reviewer_id="Supervisor_1", rejection_reason="Blurry sequence")
    assert res_rej["review_status"] == "REJECTED"
    assert res_rej["rejection_reason"] == "Blurry sequence"
