"""
Unit tests for Phase 11 Canonical Annotation Schema and Sequence/Temporal Types.
"""

import pytest
from signova.annotation.schema import (
    AnnotationSet,
    GlossToken,
    TemporalSegment,
    VideoAnnotation,
)


def test_sequence_only_annotation_creation():
    annot = VideoAnnotation(
        annotation_id="annot_001",
        sample_id="sample_001",
        annotator_id="annotator_deaf_01",
        is_temporally_aligned=False,
        glosses=["NAMASTE", "MY", "NAME", "FS-RAHUL"],
        segments=[],
        english_translation="Hello, my name is Rahul.",
    )
    assert annot.is_temporally_aligned is False
    assert len(annot.glosses) == 4
    assert len(annot.segments) == 0
    assert annot.to_dict()["sample_id"] == "sample_001"


def test_temporally_aligned_annotation_creation():
    seg1 = TemporalSegment(gloss="NAMASTE", start_frame=10, end_frame=30, start_time_ms=333.3, end_time_ms=1000.0)
    seg2 = TemporalSegment(gloss="COLLEGE", start_frame=31, end_frame=60, start_time_ms=1033.3, end_time_ms=2000.0)

    annot = VideoAnnotation(
        annotation_id="annot_002",
        sample_id="sample_002",
        annotator_id="annotator_deaf_02",
        is_temporally_aligned=True,
        glosses=["NAMASTE", "COLLEGE"],
        segments=[seg1, seg2],
    )
    assert annot.is_temporally_aligned is True
    assert len(annot.segments) == 2
    assert annot.segments[0].start_frame == 10
    assert annot.segments[1].end_frame == 60


def test_annotation_set_container():
    annot_set = AnnotationSet(sample_id="s_100", video_id="vid_100")
    annot1 = VideoAnnotation(annotation_id="a1", sample_id="s_100", annotator_id="user_A", is_temporally_aligned=False, glosses=["I", "GO"])
    annot2 = VideoAnnotation(annotation_id="a2", sample_id="s_100", annotator_id="user_B", is_temporally_aligned=False, glosses=["I", "WALK"])

    annot_set.add_annotation(annot1)
    annot_set.add_annotation(annot2)

    assert len(annot_set.annotations) == 2
    assert annot_set.get_annotator_annotation("user_A").glosses == ["I", "GO"]
    assert annot_set.get_annotator_annotation("user_B").glosses == ["I", "WALK"]
