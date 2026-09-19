"""
Unit tests for Phase 10 Duplicate Audit Module.
"""

import pytest
from signova.data.duplicate_audit import audit_duplicates


def test_duplicate_audit_no_duplicates():
    samples = [
        {"sample_id": f"s_{i}", "signer_id": f"signer_{i}", "session_id": f"sess_{i}", "sha256": f"SHA_{i}"}
        for i in range(5)
    ]
    report = audit_duplicates(samples, enable_near_duplicate=False)
    assert report.duplicate_detected is False
    assert report.exact_duplicates_count == 0
    assert report.metadata_duplicates_count == 0
    assert report.status == "PASSED"
    assert report.near_duplicate_audit_status == "NEAR_DUPLICATE_AUDIT_NOT_IMPLEMENTED"


def test_duplicate_audit_exact_sha_duplicate():
    samples = [
        {"sample_id": "s_1", "sha256": "EXACT_SAME_HASH_123"},
        {"sample_id": "s_2", "sha256": "EXACT_SAME_HASH_123"},  # Duplicate hash
    ]
    report = audit_duplicates(samples)
    assert report.duplicate_detected is True
    assert report.exact_duplicates_count == 1
    assert report.status == "DUPLICATES_DETECTED"
