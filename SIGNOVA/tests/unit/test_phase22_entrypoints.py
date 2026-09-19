"""
Phase 22 Entrypoint & CLI Tests.
"""

import sys
import pytest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))


def test_check_phase22_cli(monkeypatch, capsys):
    from scripts.check_phase22 import main as check22_main
    monkeypatch.setattr(sys, "argv", ["check_phase22.py"])
    check22_main()
    captured = capsys.readouterr()
    assert "PHASE 22" in captured.out.upper()
    assert "STATE_B" in captured.out


def test_check_phase22_readiness_cli(monkeypatch, capsys):
    from scripts.check_phase22_readiness import main as readiness_main
    monkeypatch.setattr(sys, "argv", ["check_phase22_readiness.py"])
    readiness_main()
    captured = capsys.readouterr()
    assert "Readiness & Gate Evaluation" in captured.out
    assert "STATE_B" in captured.out


def test_check_phase22_ctc_feasibility_cli(monkeypatch, capsys):
    from scripts.check_phase22_ctc_feasibility import main as feasibility_main
    monkeypatch.setattr(sys, "argv", ["check_phase22_ctc_feasibility.py"])
    feasibility_main()
    captured = capsys.readouterr()
    assert "CTC Feasibility Inspection" in captured.out


def test_run_phase22_verification_cli(capsys):
    from scripts.run_phase22_verification import run_phase22_verification
    res = run_phase22_verification()
    captured = capsys.readouterr()
    assert "SIGNOVA PHASE 22" in captured.out
    assert "HUMAN DATA" in captured.out
    assert "ANNOTATIONS" in captured.out
    assert "DATASET" in captured.out
    assert "PHASE 19 GATE" in captured.out
    assert "PHASE 21" in captured.out
    assert res["final_state"] == "STATE_B"
