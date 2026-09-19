"""
Phase 18 Entry Points Execution Unit Tests.

Validates:
- check_phase18.py executes without error and outputs diagnostic summary.
- check_phase18_ctc_readiness.py generates all 14 Phase 18 JSON reports.
- smoke_test_phase18_real_data_pipeline.py runs successfully and reports pipeline status.
- run_phase18_verification.py executes the 20-step verification workflow.
"""

from pathlib import Path
import subprocess
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


def test_check_phase18_cli_execution():
    script = WORKSPACE_ROOT / "scripts" / "check_phase18.py"
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert res.returncode == 0, f"check_phase18.py failed with: {res.stderr}"
    assert "SIGNOVA PHASE 18 — LIGHTWEIGHT DIAGNOSTIC CHECK" in res.stdout
    assert "Supervision State:" in res.stdout


def test_smoke_test_phase18_pipeline_execution():
    script = WORKSPACE_ROOT / "scripts" / "smoke_test_phase18_real_data_pipeline.py"
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert res.returncode == 0, f"smoke_test_phase18_real_data_pipeline.py failed with: {res.stderr}"
    assert "Starting Phase 18 Real Data Pipeline Smoke Test" in res.stdout


def test_check_phase18_ctc_readiness_cli_execution():
    script = WORKSPACE_ROOT / "scripts" / "check_phase18_ctc_readiness.py"
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert res.returncode == 0, f"check_phase18_ctc_readiness.py failed with: {res.stderr}"
    assert "SIGNOVA Phase 18 — Human Dataset Qualification & CTC Gate" in res.stdout

    # Verify report files were generated
    reports_dir = WORKSPACE_ROOT / "outputs" / "reports"
    expected_reports = [
        "phase18_supervision_gate.json",
        "phase18_annotation_inventory.json",
        "phase18_annotation_quality.json",
        "phase18_dataset_qualification.json",
        "phase18_agreement.json",
        "phase18_leakage_audit.json",
        "phase18_vocabulary.json",
        "phase18_ctc_feasibility.json",
        "phase18_training.json",
        "phase18_test_metrics.json",
        "phase18_error_analysis.json",
        "phase18_latency.json",
        "phase18_integrity.json",
        "phase18_readiness_summary.json",
    ]
    for r in expected_reports:
        assert (reports_dir / r).exists(), f"Missing report: {r}"


def test_run_phase18_verification_execution():
    script = WORKSPACE_ROOT / "scripts" / "run_phase18_verification.py"
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert res.returncode == 0, f"run_phase18_verification.py failed with: {res.stderr}"
    assert "SIGNOVA PHASE 18 — VERIFICATION" in res.stdout
