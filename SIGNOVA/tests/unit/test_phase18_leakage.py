"""
Phase 18 Leakage & Duplicate Audit Unit Tests.

Validates:
- Exact duplicate checksum detection.
- Cross-split contamination detection.
- Sample ID duplicate prevention.
"""

from signova.annotation.schema import VideoAnnotation
from signova.qualification.leakage import audit_phase12_leakage


def test_clean_annotations_pass_leakage_audit():
    ann1 = VideoAnnotation(
        annotation_id="ann_01",
        sample_id="vid_01",
        annotator_id="ann_01",
        is_temporally_aligned=True,
        glosses=["HELLO"],
        provenance_id="prov_01",
        dataset_split="train",
        metadata={"source_checksum": "HASH_A"},
    )
    ann2 = VideoAnnotation(
        annotation_id="ann_02",
        sample_id="vid_02",
        annotator_id="ann_01",
        is_temporally_aligned=True,
        glosses=["WORLD"],
        provenance_id="prov_01",
        dataset_split="val",
        metadata={"source_checksum": "HASH_B"},
    )
    res = audit_phase12_leakage([ann1, ann2])
    assert res.audit_status == "PASSED"
    assert len(res.duplicate_checksums) == 0


def test_duplicate_checksum_detected_as_leakage():
    ann1 = VideoAnnotation(
        annotation_id="ann_01",
        sample_id="vid_01",
        annotator_id="ann_01",
        is_temporally_aligned=True,
        glosses=["HELLO"],
        provenance_id="prov_01",
        dataset_split="train",
        metadata={"source_checksum": "HASH_SAME"},
    )
    ann2 = VideoAnnotation(
        annotation_id="ann_02",
        sample_id="vid_02",
        annotator_id="ann_01",
        is_temporally_aligned=True,
        glosses=["HELLO"],
        provenance_id="prov_01",
        dataset_split="val",
        metadata={"source_checksum": "HASH_SAME"},
    )
    res = audit_phase12_leakage([ann1, ann2])
    assert len(res.duplicate_checksums) > 0
    assert "HASH_SAME" in res.duplicate_checksums[0]
    assert res.audit_status == "FAILED"
