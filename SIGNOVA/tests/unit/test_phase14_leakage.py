"""
Phase 14 Leakage Audit Unit Tests.
"""

from signova.annotation.schema import VideoAnnotation
from signova.qualification.leakage import audit_phase12_leakage


def test_leakage_audit_detects_duplicate_checksums():
    annots = [
        VideoAnnotation("a1", "sample_01", "u1", False, ["A"], metadata={"source_checksum": "dup_sha"}),
        VideoAnnotation("a2", "sample_02", "u1", False, ["A"], metadata={"source_checksum": "dup_sha"}),
    ]
    report = audit_phase12_leakage(annots)
    assert len(report.duplicate_checksums) > 0
    assert report.audit_status == "FAILED"
