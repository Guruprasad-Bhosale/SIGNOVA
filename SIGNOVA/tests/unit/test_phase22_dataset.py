"""
Phase 22 Dataset Formation & Freeze Semantics Tests.
"""

import pytest
from signova.operations.phase22_orchestrator import Phase22Orchestrator


def test_dataset_freeze_and_invalidation(tmp_path):
    orch = Phase22Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.register_annotator("ann_01", qualification_status="QUALIFIED")

    # Import and verify annotation
    orch.import_human_annotation(
        annotation_id="ann_d_1",
        video_id="vid_1",
        annotator_id="ann_01",
        gloss_sequence=["HELLO", "WORLD"]
    )
    orch.review_annotation("ann_d_1", "VERIFIED", reviewer_id="Lead_01")

    # Build dataset
    ds = orch.build_dataset(split_strategy="RANDOM", allow_random=True, freeze=True)
    assert ds["dataset_status"] == "DATASET_FROZEN"
    assert ds["is_frozen"] is True
    assert "dataset_fingerprint" in ds

    # Attempting to build or modify frozen dataset without unfreezing should be rejected
    with pytest.raises(ValueError, match="Dataset is currently frozen"):
        orch.build_dataset(split_strategy="RANDOM", allow_random=True, freeze=True)

    # Unfreeze succeeds
    unf = orch.unfreeze_dataset(reason="Adding new verified annotations")
    assert unf["dataset_status"] == "DATASET_DRAFT"
