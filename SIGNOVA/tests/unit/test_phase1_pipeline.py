"""
Unit tests for Phase 1 canonical sample schemas, adapters, and split validators.
"""

from pathlib import Path
import pytest
from signova.data.isltranslate import ISLTranslateAdapter
from signova.data.models import SampleAvailability, SignSample
from signova.data.schema import SignSampleSchema


def test_sign_sample_canonical_model():
    sample = SignSample(
        sample_id="test_sample_01",
        dataset="ISLTranslate",
        video_reference="1782bea75c7d-1",
        split="train",
        source_language="Indian Sign Language (ISL)",
        target_language="English",
        target_translation="Hello world",
        signer_id=None,
        session_id="1782bea75c7d",
        availability=SampleAvailability.REMOTE_ONLY,
        metadata={"note": "unit_test"},
    )
    d = sample.to_dict()
    assert d["sample_id"] == "test_sample_01"
    assert d["availability"] == "REMOTE_ONLY"
    assert d["signer_id"] is None

    reconstructed = SignSample.from_dict(d)
    assert reconstructed.sample_id == "test_sample_01"
    assert reconstructed.availability == SampleAvailability.REMOTE_ONLY
    assert reconstructed.session_id == "1782bea75c7d"


def test_sign_sample_pydantic_schema_validation():
    schema = SignSampleSchema(
        sample_id="test_schema_01",
        dataset="ISLTranslate",
        video_reference="uid_123",
        split="val",
        target_translation="Valid translation string",
        availability=SampleAvailability.REMOTE_ONLY,
    )
    assert schema.sample_id == "test_schema_01"
    assert schema.availability == SampleAvailability.REMOTE_ONLY


def test_isltranslate_adapter_get_sample():
    # Point to the real local CSV
    csv_path = Path("g:/SingLang/ISLTranslate-main/data/ISLTranslate.csv")
    adapter = ISLTranslateAdapter(csv_path=csv_path)

    sample = adapter.get_sample("1782bea75c7d-1")
    assert sample is not None
    assert sample.sample_id == "1782bea75c7d-1"
    assert sample.video_reference == "1782bea75c7d-1"
    assert sample.session_id == "1782bea75c7d"
    assert sample.signer_id is None
    assert sample.availability == SampleAvailability.REMOTE_ONLY
    assert sample.target_translation == "Page 111"


def test_isltranslate_adapter_summary():
    csv_path = Path("g:/SingLang/ISLTranslate-main/data/ISLTranslate.csv")
    adapter = ISLTranslateAdapter(csv_path=csv_path)
    summary = adapter.summary()

    assert summary["dataset_name"] == "ISLTranslate"
    assert summary["total_pairs"] == 31222
    assert summary["local_videos"] == 0
    assert summary["signer_metadata_status"] == "UNKNOWN (Unverified in raw CSV)"
