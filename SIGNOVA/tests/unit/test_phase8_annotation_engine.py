"""
Unit tests for Phase 8 Sequential ISL Annotation Ingestion Engine (ELAN, CSV, JSON, Quality Grades).
"""

from pathlib import Path
import tempfile
import pytest

from signova.data.annotation_engine import (
    CSVAnnotationAdapter,
    ELANAnnotationAdapter,
    JSONAnnotationAdapter,
    SequentialISLAnnotation,
    SupervisionGrade,
    TemporalSegment,
    TRAINING_ELIGIBLE_GRADES,
    validate_annotation_semantics,
)


def test_sequential_isl_annotation_schema_and_eligibility():
    # 1. Unverified annotation
    unverified_ann = SequentialISLAnnotation(
        sample_id="ANN-001",
        video_id="VID-001",
        gloss_sequence=["I", "GO", "COLLEGE"],
        supervision_grade=SupervisionGrade.UNVERIFIED,
    )
    assert unverified_ann.is_training_eligible() is False

    # 2. Verified annotation
    verified_ann = SequentialISLAnnotation(
        sample_id="ANN-002",
        video_id="VID-002",
        gloss_sequence=["I", "GO", "COLLEGE"],
        supervision_grade=SupervisionGrade.VERIFIED,
    )
    assert verified_ann.is_training_eligible() is True

    # 3. Linguist reviewed annotation
    linguist_ann = SequentialISLAnnotation(
        sample_id="ANN-003",
        video_id="VID-003",
        gloss_sequence=["ME", "DEAF", "PROUD"],
        supervision_grade=SupervisionGrade.LINGUIST_REVIEWED,
    )
    assert linguist_ann.is_training_eligible() is True

    # 4. CTC Feasibility check
    feas_ok, _ = verified_ann.check_ctc_feasibility(feature_frames=40)
    assert feas_ok is True
    feas_bad, msg = verified_ann.check_ctc_feasibility(feature_frames=2)  # 2 < 3
    assert feas_bad is False
    assert "CTC constraint violated" in msg


def test_annotation_semantics_validator():
    # Valid annotation
    valid_ann = SequentialISLAnnotation(
        sample_id="ANN-VALID",
        video_id="VID-01",
        gloss_sequence=["BOOK", "READ", "FINISH"],
        temporal_segments=[
            TemporalSegment(gloss="BOOK", start_ms=0, end_ms=500),
            TemporalSegment(gloss="READ", start_ms=600, end_ms=1200),
            TemporalSegment(gloss="FINISH", start_ms=1300, end_ms=2000),
        ],
    )
    is_valid, errors = validate_annotation_semantics(valid_ann)
    assert is_valid is True
    assert len(errors) == 0

    # Invalid annotation (English sentence masquerading as gloss)
    invalid_ann = SequentialISLAnnotation(
        sample_id="ANN-INVALID",
        video_id="VID-02",
        gloss_sequence=["I am going to college today."],
    )
    is_valid_inv, errors_inv = validate_annotation_semantics(invalid_ann)
    assert is_valid_inv is False
    assert any("Multi-word string" in e for e in errors_inv)


def test_elan_annotation_adapter():
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR="Test" DATE="2026-09-18" FORMAT="3.0" VERSION="3.0">
    <TIME_ORDER>
        <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="100"/>
        <TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="600"/>
        <TIME_SLOT TIME_SLOT_ID="ts3" TIME_VALUE="700"/>
        <TIME_SLOT TIME_SLOT_ID="ts4" TIME_VALUE="1400"/>
    </TIME_ORDER>
    <TIER LINGUISTIC_TYPE_REF="gloss" TIER_ID="GLOSS_MAIN">
        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="a1" TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
                <ANNOTATION_VALUE>HELLO</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>
        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="a2" TIME_SLOT_REF1="ts3" TIME_SLOT_REF2="ts4">
                <ANNOTATION_VALUE>WORLD</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>
    </TIER>
    <TIER LINGUISTIC_TYPE_REF="trans" TIER_ID="TRANSLATION_EN">
        <ANNOTATION>
            <ALIGNABLE_ANNOTATION ANNOTATION_ID="a3" TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts4">
                <ANNOTATION_VALUE>Hello world.</ANNOTATION_VALUE>
            </ALIGNABLE_ANNOTATION>
        </ANNOTATION>
    </TIER>
</ANNOTATION_DOCUMENT>
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        eaf_file = Path(tmpdir) / "test.eaf"
        eaf_file.write_text(sample_xml, encoding="utf-8")

        adapter = ELANAnnotationAdapter()
        ann = adapter.parse_file(eaf_file, video_id="vid_test_01", supervision_grade=SupervisionGrade.LINGUIST_REVIEWED)

        assert ann.video_id == "vid_test_01"
        assert ann.gloss_sequence == ["HELLO", "WORLD"]
        assert ann.english_translation == "Hello world."
        assert len(ann.temporal_segments) == 2
        assert ann.temporal_segments[0].start_ms == 100
        assert ann.temporal_segments[0].end_ms == 600
        assert ann.is_training_eligible() is True


def test_csv_and_json_annotation_adapters():
    with tempfile.TemporaryDirectory() as tmpdir:
        # CSV test
        csv_file = Path(tmpdir) / "test.csv"
        csv_content = "video_id,gloss_sequence,english_translation,signer_id,supervision_grade\nVID-10,HELP ME,Please help me.,S-1,VERIFIED\n"
        csv_file.write_text(csv_content, encoding="utf-8")

        csv_adapter = CSVAnnotationAdapter()
        csv_anns = csv_adapter.parse_csv(csv_file)
        assert len(csv_anns) == 1
        assert csv_anns[0].gloss_sequence == ["HELP", "ME"]
        assert csv_anns[0].supervision_grade == SupervisionGrade.VERIFIED
        assert csv_anns[0].is_training_eligible() is True

        # JSON test
        json_file = Path(tmpdir) / "test.json"
        json_content = """[
            {
                "video_id": "VID-20",
                "gloss_sequence": ["WATER", "DRINK"],
                "english_translation": "Drink water.",
                "supervision_grade": "LINGUIST_REVIEWED"
            }
        ]"""
        json_file.write_text(json_content, encoding="utf-8")

        json_adapter = JSONAnnotationAdapter()
        json_anns = json_adapter.parse_json(json_file)
        assert len(json_anns) == 1
        assert json_anns[0].gloss_sequence == ["WATER", "DRINK"]
        assert json_anns[0].is_training_eligible() is True
