"""
Phase 23 Entrypoint & CLI Tests.
"""

import sys
import pytest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))


def test_check_phase23_cli(monkeypatch, capsys):
    from scripts.check_phase23 import main as check23_main
    monkeypatch.setattr(sys, "argv", ["check_phase23.py"])
    check23_main()
    captured = capsys.readouterr()
    assert "PHASE 23" in captured.out.upper()
    assert "STATE_B" in captured.out


def test_run_phase23_data_accounting_cli(monkeypatch, capsys):
    from scripts.run_phase23_data_accounting import main as accounting_main
    monkeypatch.setattr(sys, "argv", ["run_phase23_data_accounting.py"])
    accounting_main()
    captured = capsys.readouterr()
    assert "Data Accounting" in captured.out


def test_run_phase23_ctc_readiness_cli(monkeypatch, capsys):
    from scripts.run_phase23_ctc_readiness import main as readiness_main
    monkeypatch.setattr(sys, "argv", ["run_phase23_ctc_readiness.py"])
    readiness_main()
    captured = capsys.readouterr()
    assert "CTC Readiness" in captured.out


def test_run_phase23_training_cli(monkeypatch, capsys):
    from scripts.run_phase23_training import main as train_main
    monkeypatch.setattr(sys, "argv", ["run_phase23_training.py"])
    train_main()
    captured = capsys.readouterr()
    assert "Gated CTC Training Orchestrator" in captured.out
    assert "BLOCKED" in captured.out or "NOT_REQUESTED" in captured.out


def test_run_phase23_verification_cli(capsys):
    from scripts.run_phase23_verification import run_phase23_verification
    res = run_phase23_verification()
    captured = capsys.readouterr()
    assert "SIGNOVA PHASE 23" in captured.out
    assert "ACQUISITION" in captured.out
    assert "ANNOTATIONS" in captured.out
    assert "TRAINING DATA" in captured.out
    assert "DATASET" in captured.out
    assert "PHASE 19" in captured.out
    assert "PHASE 21" in captured.out
    assert res["final_state"] == "STATE_B"
