"""
Phase 19 Entry Points Execution Unit Tests.

Validates:
- check_phase19.py executes without error and outputs diagnostic summary.
- check_phase19_ctc_readiness.py generates all 15 Phase 19 JSON reports.
- smoke_test_phase19_real_data_pipeline.py runs successfully and reports pipeline status.
- run_phase19_verification.py executes the verification workflow.
"""

from pathlib import Path
import subprocess
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


def test_check_phase19_cli_execution():
    script = WORKSPACE_ROOT / "scripts" / "check_phase19.py"
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert res.returncode == 0, f"check_phase19.py failed with: {res.stderr}"
    assert "SIGNOVA PHASE 19 — LIGHTWEIGHT DIAGNOSTIC CHECK" in res.stdout
    assert "Supervision State:" in res.stdout


def test_smoke_test_phase19_pipeline_execution():
    script = WORKSPACE_ROOT / "scripts" / "smoke_test_phase19_real_data_pipeline.py"
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert res.returncode == 0, f"smoke_test_phase19_real_data_pipeline.py failed with: {res.stderr}"
    assert "Starting Phase 19 Real Data Pipeline Smoke Test" in res.stdout


def test_check_phase19_ctc_readiness_cli_execution():
    script = WORKSPACE_ROOT / "scripts" / "check_phase19_ctc_readiness.py"
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert res.returncode == 0, f"check_phase19_ctc_readiness.py failed with: {res.stderr}"
    assert "SIGNOVA Phase 19 — Human Dataset Qualification & CTC Gate" in res.stdout

    # Verify report files were generated
    reports_dir = WORKSPACE_ROOT / "outputs" / "reports"
    expected_reports = [
        "phase19_supervision_gate.json",
        "phase19_annotation_inventory.json",
        "phase19_annotation_quality.json",
        "phase19_dataset_qualification.json",
        "phase19_agreement.json",
        "phase19_leakage_audit.json",
        "phase19_vocabulary.json",
        "phase19_ctc_feasibility.json",
        "phase19_training.json",
        "phase19_test_metrics.json",
        "phase19_error_analysis.json",
        "phase19_latency.json",
        "phase19_integrity.json",
        "phase19_readiness_summary.json",
        "phase19_assignment_manifest.json",
    ]
    for r in expected_reports:
        assert (reports_dir / r).exists(), f"Missing report: {r}"


def test_run_phase19_verification_execution():
    script = WORKSPACE_ROOT / "scripts" / "run_phase19_verification.py"
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert res.returncode == 0, f"run_phase19_verification.py failed with: {res.stderr}"
    assert "SIGNOVA PHASE 19 — GENUINE HUMAN DATA ACQUISITION" in res.stdout
