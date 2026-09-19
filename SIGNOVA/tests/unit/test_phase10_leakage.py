"""
Unit tests for Phase 10 Signer and Session Leakage Audit Report.
"""

import json
from pathlib import Path
import pytest


def test_phase10_leakage_audit_json_report():
    report_path = Path("outputs/reports/phase10_leakage_audit.json")
    assert report_path.is_file(), "phase10_leakage_audit.json must exist"

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert "signer_independent_split_possible" in data
    assert "leakage_detected" in data
    assert "status" in data
