"""
Phase 12 Sequential ISL Dataset Manifest Generator for SIGNOVA.

Outputs canonical CSV manifest data/manifests/phase12_sequential_dataset.csv.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from signova.annotation.schema import VideoAnnotation


MANIFEST_COLUMNS = [
    "sample_id",
    "video_id",
    "annotation_id",
    "annotator_id",
    "reviewer_id",
    "signer_id",
    "session_id",
    "source_checksum",
    "annotation_version",
    "vocabulary_version",
    "sequence_length",
    "temporal_alignment_available",
    "quality_grade",
    "training_eligible",
    "split",
]


def generate_phase12_manifest(
    annotations: List[VideoAnnotation],
    output_csv_path: Optional[Path] = None,
    vocabulary_version: str = "0.1.0",
) -> Path:
    """Generates canonical CSV manifest from verified human annotations."""
    if output_csv_path is None:
        output_csv_path = Path("data/manifests/phase12_sequential_dataset.csv")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for a in annotations:
        meta = a.metadata or {}
        rows.append({
            "sample_id": a.sample_id,
            "video_id": meta.get("video_id", f"{a.sample_id}.mp4"),
            "annotation_id": a.annotation_id,
            "annotator_id": a.annotator_id or "UNKNOWN",
            "reviewer_id": a.reviewer_id or "UNKNOWN",
            "signer_id": meta.get("signer_id", "UNKNOWN"),
            "session_id": meta.get("session_id", "UNKNOWN"),
            "source_checksum": meta.get("source_checksum", "UNKNOWN"),
            "annotation_version": a.version or "0.1.0",
            "vocabulary_version": vocabulary_version,
            "sequence_length": len(a.glosses),
            "temporal_alignment_available": str(a.is_temporally_aligned).lower(),
            "quality_grade": a.quality_grade,
            "training_eligible": str(a.training_eligible).lower(),
            "split": a.dataset_split,
        })

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return output_csv_path


def read_phase12_manifest(csv_path: Path) -> List[Dict[str, str]]:
    """Reads the Phase 12 CSV manifest into dictionaries."""
    if not csv_path.exists():
        return []

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
