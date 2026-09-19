"""
Phase 22 Pilot Video Assignment Tests.
"""

import pytest
from signova.operations.phase22_orchestrator import Phase22Orchestrator


def test_assignment_creation_and_immutability(tmp_path):
    orch = Phase22Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")

    # Register annotators
    orch.register_annotator("ann_01", qualification_status="QUALIFIED")
    orch.register_annotator("ann_02", qualification_status="QUALIFIED")

    # Create pilot assignments (20 primary + 20% double = 24 total)
    manifest = orch.create_pilot_assignments(video_count=20, reassign=False)
    assert manifest.get("video_count") == 20
    assert manifest.get("total_assigned") == 24
    assert manifest.get("annotator_count") == 2

    # Attempting to recreate without reassign flag should not overwrite
    manifest_second = orch.create_pilot_assignments(video_count=20, reassign=False)
    assert manifest_second.get("assignments") == manifest.get("assignments")

    # Recreate with reassign=True succeeds (10 primary + 20% double = 12 total)
    manifest_third = orch.create_pilot_assignments(video_count=10, reassign=True)
    assert manifest_third.get("video_count") == 10
    assert manifest_third.get("total_assigned") == 12
