"""
Phase 12 Leakage Audit Unit Tests.
"""

from signova.annotation.schema import VideoAnnotation
from signova.qualification.leakage import audit_phase12_leakage


def test_leakage_clean_independent():
    annots = [
        VideoAnnotation(
            annotation_id="a1",
            sample_id="s1",
            annotator_id="u1",
            is_temporally_aligned=False,
            glosses=["HELLO"],
            dataset_split="train",
            metadata={"signer_id": "signer_01", "session_id": "session_01", "source_checksum": "sha_1"},
        ),
        VideoAnnotation(
            annotation_id="a2",
            sample_id="s2",
            annotator_id="u1",
            is_temporally_aligned=False,
            glosses=["WORLD"],
            dataset_split="test",
            metadata={"signer_id": "signer_02", "session_id": "session_02", "source_checksum": "sha_2"},
        ),
    ]

    report = audit_phase12_leakage(annots)
    assert report.signer_independent is True
    assert report.session_independent is True
    assert len(report.signer_overlap) == 0
    assert report.audit_status == "PASSED"


def test_leakage_signer_overlap_detected():
    annots = [
        VideoAnnotation(
            annotation_id="a1",
            sample_id="s1",
            annotator_id="u1",
            is_temporally_aligned=False,
            glosses=["HELLO"],
            dataset_split="train",
            metadata={"signer_id": "signer_01", "session_id": "session_01"},
        ),
        VideoAnnotation(
            annotation_id="a2",
            sample_id="s2",
            annotator_id="u1",
            is_temporally_aligned=False,
            glosses=["WORLD"],
            dataset_split="test",
            metadata={"signer_id": "signer_01", "session_id": "session_02"},  # Overlap
        ),
    ]

    report = audit_phase12_leakage(annots)
    assert report.signer_independent is False
    assert "signer_01" in report.signer_overlap
    assert report.audit_status == "WARNING"
