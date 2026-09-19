"""
Unit tests for Phase 9 Data Gate logic and state reports.
"""

import json
from pathlib import Path
import pytest
from signova.recognition.phase9_gate import Phase9GateStatus, RealCTCTrainingGate


def test_phase9_data_gate_json_report():
    report_path = Path("outputs/reports/phase9_data_gate.json")
    assert report_path.is_file(), "phase9_data_gate.json must exist"

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["phase"] == 9
    assert data["data_gate_state"] == "STATE C"
    assert data["real_ctc_training_permitted"] is False
    assert data["zero_cost_mode"] == "DEFAULT"
    assert data["authorized_project_spend_inr"] == 0
    assert "next_data_action" in data
    assert data["next_data_action"]["recommended_phase10_focus"] == "SIGNOVA Human Annotation Dataset Creation"


def test_data_gate_state_transitions():
    gate_c = Phase9GateStatus(
        data_gate_state="STATE C",
        verified_sequential_glosses=False,
        verified_video_pairing=False,
        valid_license=False,
        leakage_audit_passed=False,
        pilot_validation_passed=False,
    )
    assert gate_c.is_training_permitted is False

    gate_a_incomplete = Phase9GateStatus(
        data_gate_state="STATE A",
        verified_sequential_glosses=True,
        verified_video_pairing=True,
        valid_license=True,
        leakage_audit_passed=False,  # leakage failed
        pilot_validation_passed=True,
    )
    assert gate_a_incomplete.is_training_permitted is False

    gate_a_complete = Phase9GateStatus(
        data_gate_state="STATE A",
        verified_sequential_glosses=True,
        verified_video_pairing=True,
        valid_license=True,
        leakage_audit_passed=True,
        pilot_validation_passed=True,
    )
    assert gate_a_complete.is_training_permitted is True
