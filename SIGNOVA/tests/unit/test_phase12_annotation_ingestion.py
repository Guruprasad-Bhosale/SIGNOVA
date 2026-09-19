"""
Phase 12 Annotation Ingestion Unit Tests.
"""

import json
from pathlib import Path
import pytest

from signova.annotation.constants import (
    QUALITY_LINGUIST_REVIEWED,
    QUALITY_VERIFIED,
    REVIEW_STATE_VERIFIED,
)
from signova.annotation.export import export_annotation_to_json
from signova.annotation.schema import VideoAnnotation
from signova.qualification.ingestion import AnnotationIngestionEngine


@pytest.fixture
def temp_annotation_dir(tmp_path):
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    return ann_dir


def test_ingestion_valid_annotation(temp_annotation_dir, tmp_path):
    landmarks_dir = tmp_path / "landmarks"
    (landmarks_dir / "train").mkdir(parents=True, exist_ok=True)
    (landmarks_dir / "train" / "sample_001.npz").write_bytes(b"mock_npz")

    annot = VideoAnnotation(
        annotation_id="annot_001",
        sample_id="sample_001",
        annotator_id="annot_user_01",
        reviewer_id="reviewer_01",
        is_temporally_aligned=False,
        glosses=["NAMASTE", "THANK-YOU"],
        review_status=REVIEW_STATE_VERIFIED,
        quality_grade=QUALITY_LINGUIST_REVIEWED,
        dataset_split="train",
        training_eligible=True,
    )
    file_path = temp_annotation_dir / "sample_001.json"
    export_annotation_to_json(annot, file_path)

    engine = AnnotationIngestionEngine(landmarks_dir=landmarks_dir)
    res_annot, issues = engine.ingest_annotation_file(file_path)

    assert len(issues) == 0
    assert res_annot.training_eligible is True
    assert res_annot.glosses == ["NAMASTE", "THANK-YOU"]


def test_ingestion_rejects_same_annotator_and_reviewer(temp_annotation_dir, tmp_path):
    landmarks_dir = tmp_path / "landmarks"
    (landmarks_dir / "train").mkdir(parents=True, exist_ok=True)
    (landmarks_dir / "train" / "sample_002.npz").write_bytes(b"mock_npz")

    annot = VideoAnnotation(
        annotation_id="annot_002",
        sample_id="sample_002",
        annotator_id="user_same",
        reviewer_id="user_same",  # Same user -> violation
        is_temporally_aligned=False,
        glosses=["NAMASTE"],
        review_status=REVIEW_STATE_VERIFIED,
        quality_grade=QUALITY_VERIFIED,
        dataset_split="train",
        training_eligible=True,
    )
    file_path = temp_annotation_dir / "sample_002.json"
    export_annotation_to_json(annot, file_path)

    engine = AnnotationIngestionEngine(landmarks_dir=landmarks_dir)
    res_annot, issues = engine.ingest_annotation_file(file_path)

    assert "ANNOTATOR_CANNOT_BE_REVIEWER" in issues
