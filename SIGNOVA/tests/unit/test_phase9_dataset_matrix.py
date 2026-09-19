"""
Unit tests for Phase 9 Dataset Matrix schema, columns, rejection taxonomy, and eligibility rules.
"""

from pathlib import Path
import pandas as pd
import pytest


EXPECTED_COLUMNS = [
    "dataset_id",
    "dataset_name",
    "continuous",
    "video_available",
    "gloss_available",
    "ordered_gloss",
    "frame_alignment",
    "english_translation",
    "signer_metadata",
    "session_metadata",
    "license",
    "commercial_use",
    "research_use",
    "download_status",
    "annotation_quality",
    "sample_count",
    "signer_count",
    "vocab_size",
    "training_eligible",
    "reason",
]

ALLOWED_REASONS = {
    "NO_CONTINUOUS_VIDEO",
    "NO_ORDERED_GLOSS",
    "ENGLISH_ONLY",
    "ISOLATED_ONLY",
    "UNVERIFIED_ANNOTATION",
    "LICENSE_UNCLEAR",
    "VIDEO_ANNOTATION_MISMATCH",
    "INSUFFICIENT_METADATA",
    "DUPLICATE_RISK",
    "ACCESS_RESTRICTED",
    "PAID",
    "OTHER",
}


def test_dataset_matrix_schema_and_reasons():
    matrix_path = Path("outputs/reports/phase9_dataset_matrix.csv")
    assert matrix_path.is_file(), "phase9_dataset_matrix.csv must exist"

    df = pd.read_csv(matrix_path)
    assert list(df.columns) == EXPECTED_COLUMNS, f"Columns mismatch: {list(df.columns)}"
    assert len(df) >= 7, "Matrix should evaluate all audited candidate datasets"

    # All non-training eligible rows must have a valid standardized reason
    for idx, row in df.iterrows():
        assert row["reason"] in ALLOWED_REASONS, f"Invalid rejection reason: {row['reason']} for dataset {row['dataset_id']}"
        if not bool(row["training_eligible"]):
            assert row["reason"] != "", f"Missing rejection reason for non-eligible dataset {row['dataset_id']}"

    # Verify no unverified dataset is marked training_eligible = True
    assert df["training_eligible"].sum() == 0, "No external dataset should currently be marked training_eligible"
