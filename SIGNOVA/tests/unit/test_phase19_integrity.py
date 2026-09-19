"""
Phase 19 Reference Repository Cryptographic Integrity Unit Tests.

Validates:
- All 44 protected reference repository files in ISLTranslate-main and isl-translator-main match baseline SHA-256 hashes.
- Absolute immutability of protected references.
"""

import hashlib
import json
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


def test_reference_repositories_sha256_integrity_phase19():
    baseline_path = WORKSPACE_ROOT / "data" / "manifests" / "reference_integrity_baseline.json"
    assert baseline_path.is_file(), "Reference integrity baseline manifest missing."

    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    repositories = data.get("repositories", {})

    total_checked = 0
    mismatches = []

    for repo_name, repo_info in repositories.items():
        files = repo_info.get("files", {})
        for rel_path, file_info in files.items():
            expected_sha = file_info["sha256"].upper()
            target_path = WORKSPACE_ROOT / ".." / repo_name / rel_path

            assert target_path.is_file(), f"Protected reference file missing: {target_path}"

            actual_sha = hashlib.sha256(target_path.read_bytes()).hexdigest().upper()
            if actual_sha != expected_sha:
                mismatches.append((str(target_path), expected_sha, actual_sha))

            total_checked += 1

    assert total_checked == 44, f"Expected 44 reference files, checked {total_checked}"
    assert len(mismatches) == 0, f"Integrity violations detected: {mismatches}"
