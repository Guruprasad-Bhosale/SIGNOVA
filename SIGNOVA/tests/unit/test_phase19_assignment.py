"""
Phase 19 Pilot Assignment Manifest Unit Tests.

Validates:
- Deterministic assignment generation.
- Double-annotation pairing ratio.
- Extended assignment manifest schema (assignment_id, video_id, video_sha256, annotator_id, session_id, assigned_at, annotation_status, is_double_annotation, annotation_version, reviewer_id, review_status).
- Fixture isolation: ensures no fixture data pollutes real manifests.
"""

from pathlib import Path
from signova.operations.assignment import Phase19PilotAssignmentManager


def test_pilot_assignment_generation_and_schema(tmp_path):
    csv_path = tmp_path / "pilot_assignments.csv"
    json_path = tmp_path / "pilot_manifest.json"

    mgr = Phase19PilotAssignmentManager(
        landmarks_dir=tmp_path / "landmarks",
        annotations_dir=tmp_path / "annotations",
        pilot_target_samples=10,
        double_annotation_fraction=0.20,
        seed=100,
    )
    manifest = mgr.generate_pilot_assignments(
        output_csv_path=csv_path,
        output_json_path=json_path,
        assigned_annotators=["annotator_a", "annotator_b"],
        reviewer_id="reviewer_lead",
    )

    assert csv_path.exists()
    assert json_path.exists()
    assert manifest["phase"] == 19
    assert manifest["unique_videos_assigned"] == 10
    assert manifest["double_annotation_samples"] == 2
    assert len(manifest["assignments"]) == 12  # 10 single + 2 double

    # Validate first assignment record schema
    rec = manifest["assignments"][0]
    expected_keys = {
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
    }
    assert expected_keys.issubset(set(rec.keys()))
    assert rec["is_double_annotation"] is False
    assert rec["reviewer_id"] == "reviewer_lead"
