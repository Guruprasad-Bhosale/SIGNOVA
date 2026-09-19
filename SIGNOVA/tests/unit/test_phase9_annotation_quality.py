"""
Unit tests for Phase 9 Annotation Quality Audit Engine and CTC feasibility checks.
"""

import pytest
from signova.data.quality_audit import (
    audit_annotation_dataset,
    validate_gloss_token,
)


def test_validate_gloss_token():
    assert validate_gloss_token("SCHOOL") is True
    assert validate_gloss_token("ICE-CREAM") is True
    assert validate_gloss_token("FS-DELHI") is True
    assert validate_gloss_token("NUM-2024") is True
    assert validate_gloss_token("WRITE::PAPER") is True

    # Invalid tokens
    assert validate_gloss_token("school") is False
    assert validate_gloss_token("SCHOOL!") is False
    assert validate_gloss_token("") is False
    assert validate_gloss_token("   ") is False


def test_audit_annotation_dataset_clean():
    clean_samples = [
        {
            "sample_id": f"s_{i}",
            "glosses": ["I", "GO", "COLLEGE"],
            "num_frames": 90,
            "start_time": 0.0,
            "end_time": 3.0,
        }
        for i in range(10)
    ]
    report = audit_annotation_dataset(clean_samples)
    assert report.total_samples == 10
    assert report.empty_gloss_rate == 0.0
    assert report.malformed_token_rate == 0.0
    assert report.ctc_feasibility_rate == 1.0
    assert report.overall_quality_grade == "VERIFIED"
    assert report.training_eligible is True


def test_audit_annotation_dataset_defects():
    defective_samples = [
        {"sample_id": "s_1", "glosses": [], "num_frames": 30},
        {"sample_id": "s_2", "glosses": ["bad_lowercase"], "num_frames": 30},
        {"sample_id": "s_3", "glosses": ["GOOD", "TOKEN"], "num_frames": 2, "downsample_factor": 2},  # CTC infeasible
    ]
    report = audit_annotation_dataset(defective_samples, downsample_factor=2)
    assert report.total_samples == 3
    assert report.empty_gloss_samples == 1
    assert report.malformed_tokens_count == 1
    assert report.overall_quality_grade in {"WEAK", "PARTIAL"}
    assert report.training_eligible is False
