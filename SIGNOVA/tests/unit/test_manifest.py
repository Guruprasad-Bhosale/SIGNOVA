"""
Unit tests for SIGNOVA manifest operations and validation.
"""

from pathlib import Path
import pytest
from signova.data.manifest import Manifest, ManifestEntry
from signova.data.validation import validate_manifest, validate_manifest_entry


def test_manifest_entry_serialization():
    entry = ManifestEntry(
        sample_id="test-001",
        dataset="ISLTranslate",
        split="train",
        translation="Hello world",
        signer_id="signer_01",
        duration_sec=2.5,
        num_frames=75,
        fps=30.0,
        metadata={"domain": "greetings"},
    )
    d = entry.to_dict()
    assert d["sample_id"] == "test-001"
    assert d["duration_sec"] == 2.5

    reconstructed = ManifestEntry.from_dict(d)
    assert reconstructed.sample_id == "test-001"
    assert reconstructed.metadata == {"domain": "greetings"}


def test_manifest_validation():
    valid_entry = ManifestEntry(
        sample_id="test-002",
        dataset="ISLTranslate",
        split="val",
        translation="Good morning",
    )
    assert len(validate_manifest_entry(valid_entry)) == 0

    invalid_entry = ManifestEntry(
        sample_id="",
        dataset="ISLTranslate",
        split="invalid_split",
    )
    errors = validate_manifest_entry(invalid_entry)
    assert len(errors) >= 2


def test_manifest_export_and_import(tmp_path: Path):
    manifest = Manifest()
    manifest.append(ManifestEntry(sample_id="s1", dataset="test", split="train", translation="one"))
    manifest.append(ManifestEntry(sample_id="s2", dataset="test", split="val", translation="two"))

    csv_path = tmp_path / "test_manifest.csv"
    json_path = tmp_path / "test_manifest.json"

    manifest.to_csv(csv_path)
    manifest.to_json(json_path)

    loaded = Manifest.from_csv(csv_path)
    assert len(loaded) == 2
    assert loaded.entries[0].sample_id == "s1"
    assert loaded.entries[1].translation == "two"
