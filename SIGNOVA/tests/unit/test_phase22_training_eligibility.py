"""
Phase 22 Training Eligibility Verification Tests.
"""

from signova.operations.phase22_orchestrator import (
    HumanAnnotationRecord,
    AnnotatorProfile,
    evaluate_training_eligibility,
    Phase22Orchestrator
)


def test_training_eligibility_criteria(tmp_path):
    orch = Phase22Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.register_annotator("ann_01", qualification_status="QUALIFIED")
    orch.register_annotator("ann_02", qualification_status="PENDING")

    # Annotation 1: Qualified annotator, verified review -> Eligible
    orch.import_human_annotation(
        annotation_id="ann_el_1",
        video_id="vid_1",
        annotator_id="ann_01",
        gloss_sequence=["SIGN", "ONE"]
    )
    orch.review_annotation("ann_el_1", "VERIFIED", reviewer_id="Rev_1")

    # Annotation 2: Unqualified annotator -> Ineligible
    orch.import_human_annotation(
        annotation_id="ann_el_2",
        video_id="vid_2",
        annotator_id="ann_02",
        gloss_sequence=["SIGN", "TWO"]
    )
    orch.review_annotation("ann_el_2", "VERIFIED", reviewer_id="Rev_1")

    # Annotation 3: Qualified annotator, but review rejected -> Ineligible
    orch.import_human_annotation(
        annotation_id="ann_el_3",
        video_id="vid_3",
        annotator_id="ann_01",
        gloss_sequence=["SIGN", "THREE"]
    )
    orch.review_annotation("ann_el_3", "REJECTED", reviewer_id="Rev_1")

    eligible = orch.list_training_eligible_annotations()
    assert len(eligible) == 1
    assert eligible[0].annotation_id == "ann_el_1"
