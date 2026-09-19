"""
Unit tests for Phase 9 Signer and Session Leakage Audit Engine.
"""

import pytest
from signova.data.leakage_audit import audit_split_leakage


def test_leakage_audit_clean_independent():
    train_samples = [{"sample_id": f"tr_{i}", "signer_id": "signer_A", "session_id": "sess_1"} for i in range(5)]
    val_samples = [{"sample_id": f"va_{i}", "signer_id": "signer_B", "session_id": "sess_2"} for i in range(5)]
    test_samples = [{"sample_id": f"te_{i}", "signer_id": "signer_C", "session_id": "sess_3"} for i in range(5)]

    report = audit_split_leakage(train_samples, val_samples, test_samples)
    assert report.signer_independent_split_possible is True
    assert report.leakage_detected is False
    assert report.status == "PASSED"
    assert len(report.signer_overlap.get("train_val", [])) == 0


def test_leakage_audit_signer_overlap_detected():
    train_samples = [{"sample_id": f"tr_{i}", "signer_id": "signer_A"} for i in range(5)]
    val_samples = [{"sample_id": f"va_{i}", "signer_id": "signer_A"} for i in range(5)]  # Leaked signer_A

    report = audit_split_leakage(train_samples, val_samples)
    assert report.signer_independent_split_possible is True
    assert report.leakage_detected is True
    assert report.status == "LEAKAGE_DETECTED"
    assert "signer_A" in report.signer_overlap["train_val"]


def test_leakage_audit_missing_signer_metadata():
    train_samples = [{"sample_id": f"tr_{i}"} for i in range(5)]
    val_samples = [{"sample_id": f"va_{i}"} for i in range(5)]

    report = audit_split_leakage(train_samples, val_samples)
    assert report.signer_independent_split_possible is False
    assert report.status == "SIGNER_INDEPENDENT_SPLIT_NOT_POSSIBLE"
