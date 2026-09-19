"""
Phase 6 Data & Manifest Unit Tests for SIGNOVA.

Verifies:
1. Hard Data Gate classification and supervision blocker schema.
2. Dataset discovery report structure and field semantics.
3. Annotation semantics audit verification.
4. Split leakage analysis data integrity.
"""

import json
from pathlib import Path
import pytest
import pandas as pd

from signova.data.windowing import verify_window_split_leakage


def test_phase6_supervision_blocker_report_schema():
    report_path = Path("outputs/reports/phase6_supervision_blocker.json")
    assert report_path.is_file(), "phase6_supervision_blocker.json must exist"
    data = json.loads(report_path.read_text(encoding="utf-8"))

    assert data["gate_status"] == "STATE_C"
    assert "constraints_enforced" in data
    assert len(data["constraints_enforced"]) >= 3
    assert "No fabrication of gloss sequences from English translations" in data["constraints_enforced"]


def test_phase6_dataset_discovery_report():
    report_path = Path("outputs/reports/phase6_dataset_discovery.json")
    assert report_path.is_file(), "phase6_dataset_discovery.json must exist"
    data = json.loads(report_path.read_text(encoding="utf-8"))

    assert data["data_gate_verdict"] == "STATE_C"
    assert "audited_repositories" in data
    assert "ISLTranslate-main" in data["audited_repositories"]
    assert not data["audited_repositories"]["ISLTranslate-main"]["has_sequential_sign_glosses"]


def test_phase6_annotation_audit_semantics():
    report_path = Path("outputs/reports/phase6_annotation_audit.json")
    assert report_path.is_file(), "phase6_annotation_audit.json must exist"
    data = json.loads(report_path.read_text(encoding="utf-8"))

    assert data["supervision_validity"] == "BLOCKED"
    for field in data["fields_analyzed"]:
        if field["dataset"] == "ISLTranslate":
            assert not field["usable_for_ctc_sign_supervision"]


def test_phase6_split_leakage_detection():
    # Clean split
    clean_df = pd.DataFrame([
        {"sample_id": "vid_01", "split": "train"},
        {"sample_id": "vid_02", "split": "val"},
        {"sample_id": "vid_03", "split": "test"},
    ])
    res_clean = verify_window_split_leakage(clean_df, source_video_col="sample_id", split_col="split")
    assert res_clean["is_leakage_free"]
    assert not res_clean["leakage_detected"]

    # Leaking split
    leaking_df = pd.DataFrame([
        {"sample_id": "vid_01", "split": "train"},
        {"sample_id": "vid_01", "split": "test"},
    ])
    res_leak = verify_window_split_leakage(leaking_df, source_video_col="sample_id", split_col="split")
    assert not res_leak["is_leakage_free"]
    assert res_leak["leakage_detected"]
