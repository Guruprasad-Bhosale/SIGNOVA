"""
Phase 23 Dataset Freezing, Versioning & Compatibility Gate Tests.
"""

from signova.operations.phase23_orchestrator import Phase23Orchestrator


def test_dataset_freezing_and_versioning(tmp_path):
    orch = Phase23Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.phase22_orch.register_annotator("ann_01", qualification_status="QUALIFIED")

    # Add verified sample
    orch.phase22_orch.import_human_annotation(
        annotation_id="ann_f1",
        video_id="vid_f1.mp4",
        annotator_id="ann_01",
        gloss_sequence=["SIGN_A", "SIGN_B"],
        review_status="VERIFIED"
    )

    # Freeze version 1
    ds1 = orch.freeze_phase23_dataset(split_strategy="RANDOM", allow_random=True)
    assert ds1["dataset_version"] == "phase23_dataset_v001"
    assert ds1["is_frozen"] is True
    assert "dataset_sha256" in ds1
    assert "vocabulary_sha256" in ds1

    # Compatibility gate
    comp = orch.verify_dataset_model_compatibility(ds1)
    assert comp["compatible"] is True
    assert comp["reasons"] == ["PASSED"]

    # Freeze version 2
    ds2 = orch.freeze_phase23_dataset(split_strategy="RANDOM", allow_random=True)
    assert ds2["dataset_version"] == "phase23_dataset_v002"
