"""
Phase 23 Strict Split Hierarchy Tests.
"""

import pytest
from signova.operations.phase23_orchestrator import Phase23Orchestrator


def test_split_hierarchy_guard_against_random(tmp_path):
    orch = Phase23Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.phase22_orch.register_annotator("ann_01", qualification_status="QUALIFIED")

    orch.phase22_orch.import_human_annotation(
        annotation_id="ann_s1",
        video_id="v1.mp4",
        annotator_id="ann_01",
        gloss_sequence=["SIGN_A"],
        review_status="VERIFIED"
    )

    # Random split without allow_random flag must raise ValueError
    with pytest.raises(ValueError, match="RANDOM split requires explicit confirmation"):
        orch.freeze_phase23_dataset(split_strategy="RANDOM", allow_random=False)

    # Random split with allow_random=True succeeds
    ds = orch.freeze_phase23_dataset(split_strategy="RANDOM", allow_random=True)
    assert ds["split_strategy"] == "RANDOM"
