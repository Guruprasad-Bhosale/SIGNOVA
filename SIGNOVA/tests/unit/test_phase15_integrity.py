"""
Phase 15 Reference Repository Cryptographic Integrity Baseline Tests.

Verifies:
1. All 44/44 reference files in ISLTranslate-main and isl-translator-main match baseline SHA-256 hashes.
2. Read-only policy preservation across external codebases during Phase 15.
3. Phase 15 integrity report correctness.
"""

import hashlib
import json
from pathlib import Path
import pytest


def test_reference_repositories_sha256_integrity_phase15():
    baseline_path = Path("data/manifests/reference_integrity_baseline.json")
    assert baseline_path.is_file(), "Reference integrity baseline JSON manifest must exist"

    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert "repositories" in data

    total_files = 0
    matches = 0
    mismatches = []

    for repo_name, repo_info in data["repositories"].items():
        for rel_path, file_info in repo_info.get("files", {}).items():
            total_files += 1
            target_path = Path("..") / repo_name / rel_path
            assert target_path.is_file(), f"Reference file missing: {target_path}"

            actual_sha = hashlib.sha256(target_path.read_bytes()).hexdigest().upper()
            expected_sha = file_info["sha256"].upper()

            if actual_sha == expected_sha:
                matches += 1
            else:
                mismatches.append(f"Hash mismatch on {target_path}: expected {expected_sha}, got {actual_sha}")

    assert matches == 44, f"Expected 44 matching reference files, got {matches}. Mismatches: {mismatches}"
    assert len(mismatches) == 0


def test_phase15_integrity_report_json():
    rep_path = Path("outputs/reports/phase15_integrity.json")
    assert rep_path.is_file(), "outputs/reports/phase15_integrity.json must exist"

    data = json.loads(rep_path.read_text(encoding="utf-8"))
    assert data["phase"] == 15
    assert data["matching_files"] == 44
    assert data["all_44_files_unchanged"] is True
    assert data["integrity_status"] == "PASSED"
