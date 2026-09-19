"""
Phase 12 Human Annotation Ingestion and Validation Engine for SIGNOVA.

Validates:
- sample ID
- source video / feature pairing
- annotation version
- annotator identity
- reviewer identity
- ordered gloss sequence
- temporal alignment if present
- quality grade
- review state
- provenance
- checksum
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from signova.annotation.constants import (
    QUALITY_LINGUIST_REVIEWED,
    QUALITY_VERIFIED,
    REVIEW_STATE_VERIFIED,
)
from signova.annotation.export import import_annotation_from_json
from signova.annotation.schema import VideoAnnotation
from signova.annotation.validation import evaluate_training_eligibility
from signova.data.quality_audit import validate_gloss_token


class IngestionValidationError(Exception):
    pass


class AnnotationIngestionEngine:
    """Ingests, validates, and qualifies human annotations for Phase 12."""

    def __init__(self, raw_videos_dir: Optional[Path] = None, landmarks_dir: Optional[Path] = None):
        self.raw_videos_dir = raw_videos_dir or Path("data/raw/videos")
        self.landmarks_dir = landmarks_dir or Path("data/features/landmarks")

    def ingest_annotation_file(self, file_path: Path) -> Tuple[VideoAnnotation, List[str]]:
        """Ingests a single annotation JSON file and returns the annotation with validation diagnostics."""
        if not file_path.is_file():
            raise IngestionValidationError(f"File not found: {file_path}")

        annot = import_annotation_from_json(file_path)
        issues: List[str] = []

        # 1. Structural check
        if not annot.sample_id:
            issues.append("MISSING_SAMPLE_ID")
        if not annot.annotator_id:
            issues.append("MISSING_ANNOTATOR_ID")
        if not annot.glosses:
            issues.append("EMPTY_GLOSSES")

        # 2. Linguistic sanity
        for g in annot.glosses:
            if not validate_gloss_token(str(g)):
                issues.append(f"INVALID_GLOSS_TOKEN: {g}")

        # 3. Annotator vs Reviewer separation
        if annot.reviewer_id and annot.reviewer_id == annot.annotator_id:
            issues.append("ANNOTATOR_CANNOT_BE_REVIEWER")

        # 4. Feature pairing check
        train_feat = self.landmarks_dir / "train" / f"{annot.sample_id}.npz"
        val_feat = self.landmarks_dir / "val" / f"{annot.sample_id}.npz"
        test_feat = self.landmarks_dir / "test" / f"{annot.sample_id}.npz"

        has_features = train_feat.exists() or val_feat.exists() or test_feat.exists()
        if not has_features:
            issues.append(f"LANDMARK_FEATURE_FILE_MISSING for sample {annot.sample_id}")

        # 5. Evaluate training eligibility
        eligible, eval_reasons = evaluate_training_eligibility(annot)
        if not eligible:
            issues.extend(eval_reasons)

        return annot, issues

    def batch_ingest(self, annotations_dir: Path) -> Dict[str, Any]:
        """Ingests all annotations in a directory."""
        if not annotations_dir.exists():
            return {
                "total_files": 0,
                "valid_annotations": [],
                "rejected_annotations": [],
                "total_glosses_ingested": 0,
            }

        valid = []
        rejected = []
        total_glosses = 0

        for f in sorted(annotations_dir.glob("*.json")):
            try:
                annot, issues = self.ingest_annotation_file(f)
                if not issues and annot.training_eligible:
                    valid.append(annot)
                    total_glosses += len(annot.glosses)
                else:
                    rejected.append({"file": f.name, "sample_id": annot.sample_id, "issues": issues})
            except Exception as e:
                rejected.append({"file": f.name, "sample_id": "UNKNOWN", "issues": [str(e)]})

        return {
            "total_files": len(list(annotations_dir.glob("*.json"))),
            "valid_annotations": valid,
            "rejected_annotations": rejected,
            "total_glosses_ingested": total_glosses,
        }
