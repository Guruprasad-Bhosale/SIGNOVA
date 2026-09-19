"""
Unit tests for SHA-256 generation, verification, and provenance chain tracker.
"""

from pathlib import Path
import pytest
from signova.data.provenance import (
    ProvenanceChain,
    ProvenanceNode,
    ZeroCostPolicyEnforcer,
    compute_sha256,
    generate_checksum_records,
)


def test_sha256_computation(tmp_path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("Hello ISL Recognition", encoding="utf-8")

    sha = compute_sha256(test_file)
    assert len(sha) == 64
    assert sha.isupper()

    # Re-computing on same content yields identical hash
    sha2 = compute_sha256(test_file)
    assert sha == sha2


def test_provenance_chain_traceability():
    chain = ProvenanceChain(sample_id="test_isl_001")
    assert not chain.is_complete()

    chain.add_step("SOURCE", "https://data.example.org/isl")
    chain.add_step("DOWNLOADED_FILE", "isl_raw.mp4", sha256="ABCDEF1234567890")
    chain.add_step("ANNOTATION", "isl_raw.eaf", sha256="1234567890ABCDEF")
    chain.add_step("VIDEO", "isl_raw.mp4")
    chain.add_step("FEATURE", "isl_raw.npz", sha256="CAFEBABE11223344")
    chain.add_step("MODEL_SAMPLE", "sample_001")

    assert chain.is_complete()
    chain_dict = chain.to_dict()
    assert chain_dict["sample_id"] == "test_isl_001"
    assert chain_dict["chain_length"] == 6


def test_checksum_report_exists():
    report_path = Path("outputs/reports/phase9_checksums.csv")
    assert report_path.is_file(), "phase9_checksums.csv must exist"
