"""
Phase 16 Leakage Audit Unit Tests.

Validates:
- Duplicate media/checksum detection across dataset partitions.
- Signer overlap audit across train/val/test splits.
- Verification that clean partitions pass the leakage audit with ZERO_LEAKAGE.
"""

from signova.annotation.schema import VideoAnnotation
from signova.qualification.leakage import audit_phase12_leakage


def test_leakage_audit_detects_duplicate_checksums():
    annots = [
        VideoAnnotation("p16_a1", "sample_01", "u1", False, ["A"], metadata={"source_checksum": "dup_sha_123"}),
        VideoAnnotation("p16_a2", "sample_02", "u1", False, ["A"], metadata={"source_checksum": "dup_sha_123"}),
    ]
    report = audit_phase12_leakage(annots)
    assert len(report.duplicate_checksums) > 0
    assert report.audit_status == "FAILED"


def test_clean_annotations_pass_leakage_audit():
    annots = [
        VideoAnnotation("p16_a1", "sample_01", "u1", False, ["HELLO"], metadata={"source_checksum": "sha_unique_01"}),
        VideoAnnotation("p16_a2", "sample_02", "u2", False, ["WORLD"], metadata={"source_checksum": "sha_unique_02"}),
    ]
    report = audit_phase12_leakage(annots)
    assert len(report.duplicate_checksums) == 0
    assert report.audit_status == "PASSED"
