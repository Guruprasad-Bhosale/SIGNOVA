"""
Phase 12 Manifest Unit Tests.
"""

from pathlib import Path
from signova.annotation.schema import VideoAnnotation
from signova.qualification.manifest import generate_phase12_manifest, read_phase12_manifest


def test_generate_and_read_manifest(tmp_path):
    annots = [
        VideoAnnotation(
            annotation_id="annot_01",
            sample_id="sample_01",
            annotator_id="u1",
            reviewer_id="r1",
            is_temporally_aligned=True,
            glosses=["HELLO", "FRIEND"],
            quality_grade="VERIFIED",
            training_eligible=True,
            dataset_split="train",
            metadata={"signer_id": "signer_10", "session_id": "sess_01", "source_checksum": "abc123sha"},
        )
    ]

    out_csv = tmp_path / "phase12_dataset.csv"
    generate_phase12_manifest(annots, output_csv_path=out_csv)

    assert out_csv.exists()
    rows = read_phase12_manifest(out_csv)
    assert len(rows) == 1
    r = rows[0]
    assert r["sample_id"] == "sample_01"
    assert r["annotator_id"] == "u1"
    assert r["reviewer_id"] == "r1"
    assert r["signer_id"] == "signer_10"
    assert r["session_id"] == "sess_01"
    assert r["source_checksum"] == "abc123sha"
    assert r["sequence_length"] == "2"
    assert r["temporal_alignment_available"] == "true"
    assert r["quality_grade"] == "VERIFIED"
    assert r["training_eligible"] == "true"
    assert r["split"] == "train"
