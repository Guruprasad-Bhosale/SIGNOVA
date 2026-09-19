"""
Phase 13 Pilot Dataset Manager for SIGNOVA.

Selects deterministic pilot samples and writes data/manifests/phase13_annotation_pilot.csv.
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

PILOT_MANIFEST_COLUMNS = [
    "sample_id",
    "video_id",
    "source_checksum",
    "selection_seed",
    "selection_reason",
    "annotation_status",
    "quality_grade",
    "training_eligible",
    "signer_id",
    "session_id",
]


class PilotDatasetManager:
    """Manages Phase 13 human annotation pilot sample selection and manifest."""

    def __init__(
        self,
        landmarks_dir: Optional[Path] = None,
        annotations_dir: Optional[Path] = None,
        seed: int = 42,
    ):
        self.landmarks_dir = landmarks_dir or Path("data/features/landmarks/train")
        self.annotations_dir = annotations_dir or Path("data/annotations/phase11/human_gold")
        self.seed = seed

    def generate_pilot_manifest(
        self,
        output_csv_path: Optional[Path] = None,
        num_pilot_samples: int = 20,
    ) -> Path:
        """Generates phase13_annotation_pilot.csv based on available landmarks and annotations."""
        if output_csv_path is None:
            output_csv_path = Path("data/manifests/phase13_annotation_pilot.csv")

        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        feature_files = sorted(self.landmarks_dir.glob("*.npz")) if self.landmarks_dir.exists() else []
        pilot_files = feature_files[:num_pilot_samples]

        rows = []
        for p in pilot_files:
            sample_id = p.stem
            annot_file = self.annotations_dir / f"{sample_id}.json"
            status = "UNANNOTATED"
            quality = "UNVERIFIED"
            eligible = "false"
            signer_id = "UNKNOWN"
            session_id = "UNKNOWN"

            if annot_file.exists():
                try:
                    data = json.loads(annot_file.read_text(encoding="utf-8"))
                    status = data.get("review_status", "UNANNOTATED")
                    quality = data.get("quality_grade", "UNVERIFIED")
                    eligible = str(data.get("training_eligible", False)).lower()
                    meta = data.get("metadata", {})
                    signer_id = meta.get("signer_id", "UNKNOWN")
                    session_id = meta.get("session_id", "UNKNOWN")
                except Exception:
                    pass

            sha = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]

            rows.append({
                "sample_id": sample_id,
                "video_id": f"{sample_id}.mp4",
                "source_checksum": sha,
                "selection_seed": str(self.seed),
                "selection_reason": "DIAGNOSTIC_PILOT_SELECTION",
                "annotation_status": status,
                "quality_grade": quality,
                "training_eligible": eligible,
                "signer_id": signer_id,
                "session_id": session_id,
            })

        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PILOT_MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        return output_csv_path
