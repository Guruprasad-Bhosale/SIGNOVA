"""
Unit tests for Phase 11 Canonical JSON and Optional EAF Export/Import.
"""

from pathlib import Path
import pytest
from signova.annotation.export import (
    export_annotation_to_eaf,
    export_annotation_to_json,
    import_annotation_from_json,
)
from signova.annotation.schema import (
    TemporalSegment,
    VideoAnnotation,
)


def test_canonical_json_roundtrip(tmp_path):
    seg = TemporalSegment(gloss="NAMASTE", start_frame=10, end_frame=35, start_time_ms=333.3, end_time_ms=1166.7)
    annot = VideoAnnotation(
        annotation_id="a_roundtrip",
        sample_id="s_roundtrip",
        annotator_id="deaf_01",
        is_temporally_aligned=True,
        glosses=["NAMASTE"],
        segments=[seg],
        english_translation="Hello",
        review_status="VERIFIED",
        quality_grade="LINGUIST_REVIEWED",
        reviewer_id="linguist_01",
        dataset_split="train",
        training_eligible=True,
    )

    json_file = tmp_path / "annot.json"
    export_annotation_to_json(annot, json_file)

    loaded = import_annotation_from_json(json_file)
    assert loaded.annotation_id == "a_roundtrip"
    assert loaded.is_temporally_aligned is True
    assert len(loaded.segments) == 1
    assert loaded.segments[0].start_frame == 10
    assert loaded.english_translation == "Hello"
    assert loaded.training_eligible is True


def test_optional_eaf_export(tmp_path):
    annot = VideoAnnotation(
        annotation_id="a_eaf",
        sample_id="s_eaf",
        annotator_id="deaf_01",
        is_temporally_aligned=False,
        glosses=["NAMASTE", "COLLEGE"],
    )
    eaf_file = tmp_path / "annot.eaf"
    res_path = export_annotation_to_eaf(annot, eaf_file)
    assert Path(res_path).exists()
    assert "<ANNOTATION_DOCUMENT" in eaf_file.read_text(encoding="utf-8")
