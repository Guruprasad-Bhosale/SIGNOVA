"""
Unit tests for Phase 10 Annotation Quality Audit Report and Criteria.
"""

import json
from pathlib import Path
import pytest


def test_phase10_quality_audit_json_report():
    report_path = Path("outputs/reports/phase10_annotation_quality.json")
    assert report_path.is_file(), "phase10_annotation_quality.json must exist"

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert "total_samples" in data
    assert "overall_quality_grade" in data
    assert "training_eligible" in data
    # Diagnostic samples lack verified glosses, so training_eligible must be False
    assert data["training_eligible"] is False
