"""
Phase 24 Error Analysis Taxonomy Tests.
"""

from signova.operations.phase24_orchestrator import Phase24Orchestrator


def test_error_analysis_structure(tmp_path):
    orch = Phase24Orchestrator(workspace_root=tmp_path, data_root=tmp_path / "data")
    summary = orch.get_dashboard_summary()
    assert summary["evaluation"]["status"] == "N/A"
    assert summary["evaluation"]["ter"] == "N/A"
