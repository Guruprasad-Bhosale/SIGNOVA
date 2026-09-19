"""
Phase 21 Entrypoint CLI Execution Tests.
"""

import sys
from scripts.check_phase21 import main as check_main
from scripts.prepare_phase21_dataset import main as prep_main
from scripts.train_phase21_ctc import main as train_main
from scripts.evaluate_phase21_ctc import main as eval_main
from scripts.verify_phase21_checkpoint import main as verify_main
from scripts.integrate_phase21_live_model import main as live_main
from scripts.run_phase21_verification import run_phase21_verification


def test_check_phase21_entrypoint(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check_phase21.py"])
    check_main()
    captured = capsys.readouterr()
    assert "SIGNOVA PHASE 21 -- LIGHTWEIGHT DIAGNOSTIC CHECK" in captured.out
    assert "Supervision State: STATE_B" in captured.out


def test_prepare_phase21_dataset_entrypoint_safe_exit(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prepare_phase21_dataset.py"])
    prep_main()
    captured = capsys.readouterr()
    assert "SIGNOVA PHASE 21 -- DATASET PREPARATION" in captured.out
    assert "FAST EXIT" in captured.out


def test_train_phase21_ctc_entrypoint_refusal(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_phase21_ctc.py"])
    train_main()
    captured = capsys.readouterr()
    assert "SIGNOVA PHASE 21 -- GENUINE CTC MODEL TRAINING" in captured.out
    assert "TRAINING REFUSED" in captured.out


def test_evaluate_phase21_ctc_entrypoint(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["evaluate_phase21_ctc.py"])
    eval_main()
    captured = capsys.readouterr()
    assert "SIGNOVA PHASE 21 -- MODEL EVALUATION" in captured.out


def test_verify_phase21_checkpoint_entrypoint(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["verify_phase21_checkpoint.py"])
    verify_main()
    captured = capsys.readouterr()
    assert "SIGNOVA PHASE 21 -- CHECKPOINT PROVENANCE" in captured.out


def test_integrate_phase21_live_model_entrypoint(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["integrate_phase21_live_model.py"])
    live_main()
    captured = capsys.readouterr()
    assert "SIGNOVA PHASE 21 -- LIVE MODEL INTEGRATION GATE" in captured.out


def test_run_phase21_verification_entrypoint(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_phase21_verification.py"])
    readiness = run_phase21_verification()
    captured = capsys.readouterr()
    assert "SIGNOVA PHASE 21" in captured.out
    assert "SUPERVISION" in captured.out
    assert "FINAL STATE" in captured.out
    assert readiness["supervision_state"] == "STATE_B"
