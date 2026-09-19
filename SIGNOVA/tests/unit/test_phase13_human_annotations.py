"""
Phase 13 Human Annotation Ingestion Unit Tests.
"""

from pathlib import Path
import pytest

from signova.annotation.export import export_annotation_to_json
from signova.annotation.schema import VideoAnnotation
from signova.qualification.ingestion import AnnotationIngestionEngine


def test_real_data_smoke_test_skipped_when_no_data(tmp_path):
    empty_dir = tmp_path / "empty_annotations"
    empty_dir.mkdir(parents=True, exist_ok=True)

    engine = AnnotationIngestionEngine()
    res = engine.batch_ingest(empty_dir)

    assert res["total_files"] == 0
    assert len(res["valid_annotations"]) == 0
    assert len(res["rejected_annotations"]) == 0


def test_ingestion_validates_structural_and_reviewer_integrity(tmp_path):
    ann_dir = tmp_path / "ann"
    ann_dir.mkdir(parents=True, exist_ok=True)
    landmarks_dir = tmp_path / "landmarks"
    (landmarks_dir / "train").mkdir(parents=True, exist_ok=True)
    (landmarks_dir / "train" / "sample_p13.npz").write_bytes(b"landmarks")

    annot = VideoAnnotation(
        annotation_id="p13_01",
        sample_id="sample_p13",
        annotator_id="user_annot",
        reviewer_id="user_review",
        is_temporally_aligned=False,
        glosses=["NAMASTE", "WELCOME"],
        review_status="VERIFIED",
        quality_grade="VERIFIED",
        dataset_split="train",
        training_eligible=True,
    )
    fpath = ann_dir / "sample_p13.json"
    export_annotation_to_json(annot, fpath)

    engine = AnnotationIngestionEngine(landmarks_dir=landmarks_dir)
    res_annot, issues = engine.ingest_annotation_file(fpath)

    assert len(issues) == 0
    assert res_annot.training_eligible is True
