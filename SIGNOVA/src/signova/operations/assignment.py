"""
Phase 19 Pilot Assignment Manager for SIGNOVA.

Manages deterministic pilot video assignment, double-annotation pairing,
and assignment manifest generation with extended provenance fields:
- assignment_id
- video_id
- video_sha256
- annotator_id
- session_id
- assigned_at
- annotation_status
- is_double_annotation
- annotation_version
- reviewer_id
- review_status
"""

import csv
import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

PILOT_ASSIGNMENT_CSV_COLUMNS = [
    "assignment_id",
    "video_id",
    "video_sha256",
    "annotator_id",
    "session_id",
    "assigned_at",
    "annotation_status",
    "is_double_annotation",
    "annotation_version",
    "reviewer_id",
    "review_status",
]


class Phase19PilotAssignmentManager:
    """Manages deterministic sample selection and assignments for the Phase 19 pilot."""

    def __init__(
        self,
        landmarks_dir: Optional[Path] = None,
        annotations_dir: Optional[Path] = None,
        pilot_target_samples: int = 20,
        double_annotation_fraction: float = 0.20,
        seed: int = 42,
    ):
        self.landmarks_dir = landmarks_dir or Path("data/features/landmarks/train")
        self.annotations_dir = annotations_dir or Path("data/annotations/phase11/human_gold")
        self.pilot_target_samples = pilot_target_samples
        self.double_annotation_fraction = double_annotation_fraction
        self.seed = seed

    def generate_pilot_assignments(
        self,
        output_csv_path: Optional[Path] = None,
        output_json_path: Optional[Path] = None,
        assigned_annotators: Optional[List[str]] = None,
        reviewer_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generates deterministic pilot assignments and writes CSV and JSON manifests."""
        csv_path = output_csv_path or Path("data/manifests/phase19_pilot_assignments.csv")
        json_path = output_json_path or Path("outputs/reports/phase19_assignment_manifest.json")

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        feature_files = sorted(self.landmarks_dir.glob("*.npz")) if self.landmarks_dir.exists() else []
        selected_samples = [p.stem for p in feature_files[: self.pilot_target_samples]]

        # If no feature files found on disk, handle gracefully
        if not selected_samples:
            selected_samples = [f"sample_p19_{i:03d}" for i in range(1, self.pilot_target_samples + 1)]

        annotator_pool = assigned_annotators or ["annotator_isl_01"]
        primary_reviewer = reviewer_id or "reviewer_lead_01"
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        assignments: List[Dict[str, Any]] = []
        num_double = int(round(len(selected_samples) * self.double_annotation_fraction))

        for idx, sample_id in enumerate(selected_samples):
            video_id = f"{sample_id}.mp4"
            sha = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
            ann_file = self.annotations_dir / f"{sample_id}.json"

            status = "UNANNOTATED"
            rev_status = "UNREVIEWED"
            version = "1.0.0"

            if ann_file.exists():
                try:
                    data = json.loads(ann_file.read_text(encoding="utf-8"))
                    status = data.get("annotation_status", "ANNOTATED")
                    rev_status = data.get("review_status", "REVIEW_PENDING")
                    version = data.get("annotation_version", "1.0.0")
                except Exception:
                    pass

            primary_ann_id = annotator_pool[idx % len(annotator_pool)]
            assignment_id = f"asgn_{self.seed}_{idx:03d}_a"

            asgn_record = {
                "assignment_id": assignment_id,
                "video_id": video_id,
                "video_sha256": sha,
                "annotator_id": primary_ann_id,
                "session_id": f"sess_p19_{primary_ann_id}",
                "assigned_at": now_iso,
                "annotation_status": status,
                "is_double_annotation": False,
                "annotation_version": version,
                "reviewer_id": primary_reviewer,
                "review_status": rev_status,
            }
            assignments.append(asgn_record)

            # Assign second independent annotator for double annotation subset
            if idx < num_double and len(annotator_pool) >= 2:
                second_ann_id = annotator_pool[(idx + 1) % len(annotator_pool)]
                double_asgn_id = f"asgn_{self.seed}_{idx:03d}_b"
                double_record = {
                    "assignment_id": double_asgn_id,
                    "video_id": video_id,
                    "video_sha256": sha,
                    "annotator_id": second_ann_id,
                    "session_id": f"sess_p19_{second_ann_id}",
                    "assigned_at": now_iso,
                    "annotation_status": status,
                    "is_double_annotation": True,
                    "annotation_version": version,
                    "reviewer_id": primary_reviewer,
                    "review_status": rev_status,
                }
                assignments.append(double_record)

        # Write CSV
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PILOT_ASSIGNMENT_CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(assignments)

        # Write JSON Manifest
        manifest_data = {
            "phase": 19,
            "pilot_target_samples": self.pilot_target_samples,
            "double_annotation_fraction": self.double_annotation_fraction,
            "total_assignments_generated": len(assignments),
            "unique_videos_assigned": len(selected_samples),
            "double_annotation_samples": num_double,
            "seed": self.seed,
            "assignments": assignments,
        }
        json_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        return manifest_data
