"""
Phase 22 Strict Split Hierarchy Tests.
"""

import pytest
from signova.operations.phase22_orchestrator import Phase22Orchestrator


def test_split_hierarchy_guard_against_random(tmp_path):
    orch = Phase22Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    orch.register_annotator("ann_01", qualification_status="QUALIFIED")

    orch.import_human_annotation(
        annotation_id="ann_s_1",
        video_id="vid_1",
        annotator_id="ann_01",
        gloss_sequence=["TEST", "SIGN"]
    )
    orch.review_annotation("ann_s_1", "VERIFIED", reviewer_id="Lead_01")

    # Random split without allow_random flag must raise ValueError
    with pytest.raises(ValueError, match="RANDOM split requires explicit confirmation"):
        orch.build_dataset(split_strategy="RANDOM", allow_random=False)

    # Random split with allow_random=True succeeds
    ds = orch.build_dataset(split_strategy="RANDOM", allow_random=True, freeze=False)
    assert ds["split_strategy"] == "RANDOM"
