"""
Phase 24 Checkpoint Provenance & Metadata Verification Tests.
"""

from signova.operations.phase24_orchestrator import Phase24Orchestrator


def test_checkpoint_provenance_inspection(tmp_path):
    orch = Phase24Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    summary = orch.get_dashboard_summary()
    assert summary["training"]["checkpoint"] == "NONE"
    assert summary["model_readiness"]["trained"] is False
