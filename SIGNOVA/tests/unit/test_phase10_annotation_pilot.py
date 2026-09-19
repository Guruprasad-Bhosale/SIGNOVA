"""
Unit tests for Phase 10 Human Annotation Pilot Protocol and Deterministic Sampling.
"""

import json
from pathlib import Path
import pytest


def test_annotation_sample_report_schema():
    report_path = Path("outputs/reports/phase10_annotation_sample.json")
    assert report_path.is_file(), "phase10_annotation_sample.json must exist"

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["phase"] == 10
    assert data["sampling_seed"] == 42
    assert "selected_samples" in data
    assert len(data["selected_samples"]) == 10

    categories = {s["category"] for s in data["selected_samples"]}
    assert "COMPOUND_SIGN" in categories
    assert "FINGERSPELLING" in categories
    assert "NUMBERS" in categories
    assert "CLASSIFIER_PREDICATE" in categories


def test_annotation_pilot_doc_exists():
    doc_path = Path("docs/phase10-human-annotation-pilot.md")
    assert doc_path.is_file(), "docs/phase10-human-annotation-pilot.md must exist"
    content = doc_path.read_text(encoding="utf-8")
    assert "BLOCKED_HUMAN_RESOURCE" in content
    assert "ICE-CREAM" in content
    assert "FS-DELHI" in content
