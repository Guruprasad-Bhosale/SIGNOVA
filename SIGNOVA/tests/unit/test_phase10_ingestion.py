"""
Unit tests for Phase 10 Canonical Data Ingestion Module (100% Mocked/Offline).
"""

from pathlib import Path
import pytest
from signova.data.ingestion import (
    CanonicalIngestionEngine,
    IngestionStatus,
)


def test_ingest_valid_sample(tmp_path):
    engine = CanonicalIngestionEngine(interim_dir=tmp_path / "interim")

    raw_record = {
        "dataset_id": "test_dataset",
        "sample_id": "sample_001",
        "signer_id": "signer_01",
        "session_id": "sess_01",
        "video_id": "vid_001",
        "duration_sec": 3.0,
        "frame_count": 90,
        "fps": 30.0,
        "resolution": "1920x1080",
        "glosses": ["I", "GO", "COLLEGE"],
        "english_text": "I am going to college.",
    }

    sample = engine.ingest_sample(raw_record)
    assert sample.status == IngestionStatus.VALID.value
    assert sample.quality_status == "VERIFIED"
    assert sample.failure_reason == ""
    assert sample.gloss_count == 3
    assert (tmp_path / "interim" / "test_dataset_sample_001.json").exists()


def test_ingest_invalid_empty_gloss_sample(tmp_path):
    engine = CanonicalIngestionEngine(interim_dir=tmp_path / "interim")

    raw_record = {
        "dataset_id": "test_dataset",
        "sample_id": "sample_002",
        "signer_id": "signer_01",
        "frame_count": 90,
        "glosses": [],  # Empty glosses
    }

    sample = engine.ingest_sample(raw_record)
    assert sample.status == IngestionStatus.INVALID.value
    assert "EMPTY_GLOSS_SEQUENCE" in sample.failure_reason
    assert sample.quality_status == "WEAK"


def test_ingest_requires_review_malformed_token_sample(tmp_path):
    engine = CanonicalIngestionEngine(interim_dir=tmp_path / "interim")

    raw_record = {
        "dataset_id": "test_dataset",
        "sample_id": "sample_003",
        "signer_id": "signer_01",
        "frame_count": 90,
        "duration_sec": 3.0,
        "glosses": ["bad_lowercase_token"],
    }

    sample = engine.ingest_sample(raw_record)
    assert sample.status == IngestionStatus.REQUIRES_REVIEW.value
    assert "MALFORMED_GLOSS_TOKENS" in sample.failure_reason
