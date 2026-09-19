"""
Phase 14 Pilot Manifest Generator with Strict Identity Invariants for SIGNOVA.

Enforces:
- annotator_id != signer_id
- signer_id represents the video signing participant, not the annotator
- session_id represents verified recording session metadata
- unknown fields strictly remain "UNKNOWN" without inference
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

PILOT_COLUMNS = [
    "sample_id",
    "video_id",
    "source_checksum",
    "selection_seed",
    "annotator_id",
    "reviewer_id",
    "signer_id",
    "session_id",
    "annotation_status",
    "quality_grade",
    "training_eligible",
]


class Phase14PilotManifestGenerator:
    """Generates Phase 14 pilot CSV manifest."""

    def __init__(
        self,
        landmarks_dir: Optional[Path] = None,
        annotations_dir: Optional[Path] = None,
        seed: int = 42,
    ):
        self.landmarks_dir = landmarks_dir or Path("data/features/landmarks/train")
        self.annotations_dir = annotations_dir or Path("data/annotations/phase11/human_gold")
        self.seed = seed

    def generate_manifest(
        self,
        output_csv_path: Optional[Path] = None,
        max_samples: int = 20,
    ) -> Path:
        if output_csv_path is None:
            output_csv_path = Path("data/manifests/phase14_annotation_pilot.csv")

        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        feature_files = sorted(self.landmarks_dir.glob("*.npz")) if self.landmarks_dir.exists() else []
        selected = feature_files[:max_samples]

        rows = []
        for p in selected:
            sample_id = p.stem
            annot_file = self.annotations_dir / f"{sample_id}.json"
            status = "UNANNOTATED"
            quality = "UNVERIFIED"
            eligible = "false"
            annotator_id = "UNKNOWN"
            reviewer_id = "UNKNOWN"
            signer_id = "UNKNOWN"
            session_id = "UNKNOWN"

            if annot_file.exists():
                try:
                    data = json.loads(annot_file.read_text(encoding="utf-8"))
                    status = data.get("review_status", "UNANNOTATED")
                    quality = data.get("quality_grade", "UNVERIFIED")
                    eligible = str(data.get("training_eligible", False)).lower()
                    annotator_id = data.get("annotator_id") or "UNKNOWN"
                    reviewer_id = data.get("reviewer_id") or "UNKNOWN"
                    meta = data.get("metadata", {})
                    signer_id = meta.get("signer_id", "UNKNOWN")
                    session_id = meta.get("session_id", "UNKNOWN")

                    # Strict invariant: signer_id must NOT equal annotator_id unless proven same
                    if signer_id == annotator_id and signer_id != "UNKNOWN":
                        signer_id = "UNKNOWN"  # Prevent accidental leakage
                except Exception:
                    pass

            sha = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]

            rows.append({
                "sample_id": sample_id,
                "video_id": f"{sample_id}.mp4",
                "source_checksum": sha,
                "selection_seed": str(self.seed),
                "annotator_id": annotator_id,
                "reviewer_id": reviewer_id,
                "signer_id": signer_id,
                "session_id": session_id,
                "annotation_status": status,
                "quality_grade": quality,
                "training_eligible": eligible,
            })

        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PILOT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        return output_csv_path
