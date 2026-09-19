"""
Data validation utilities for SIGNOVA datasets and samples.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from signova.data.manifest import Manifest, ManifestEntry


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


def validate_manifest_entry(entry: ManifestEntry, check_files_exist: bool = False) -> List[str]:
    """
    Validate a single manifest entry.

    Args:
        entry: ManifestEntry instance.
        check_files_exist: If True, checks that video/annotation paths physically exist.

    Returns:
        List of error message strings (empty if valid).
    """
    errors: List[str] = []

    if not entry.sample_id or not str(entry.sample_id).strip():
        errors.append("Sample ID cannot be empty.")

    if not entry.dataset or not str(entry.dataset).strip():
        errors.append(f"Dataset name missing for sample {entry.sample_id}.")

    if entry.split not in ("train", "val", "test", "unassigned"):
        errors.append(f"Invalid split '{entry.split}' for sample {entry.sample_id}. Must be train/val/test/unassigned.")

    if check_files_exist:
        if entry.video_path and not Path(entry.video_path).is_file():
            errors.append(f"Video file not found: {entry.video_path} (sample: {entry.sample_id})")
        if entry.features_path and not Path(entry.features_path).is_file():
            errors.append(f"Features file not found: {entry.features_path} (sample: {entry.sample_id})")

    return errors


def validate_manifest(manifest: Manifest, check_files_exist: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate an entire manifest collection.

    Args:
        manifest: Manifest instance.
        check_files_exist: If True, checks that referenced files exist.

    Returns:
        Tuple of (is_valid: bool, errors: List[str]).
    """
    all_errors: List[str] = []
    seen_ids = set()

    for entry in manifest.entries:
        if entry.sample_id in seen_ids:
            all_errors.append(f"Duplicate sample_id detected: {entry.sample_id}")
        seen_ids.add(entry.sample_id)

        entry_errors = validate_manifest_entry(entry, check_files_exist=check_files_exist)
        all_errors.extend(entry_errors)

    is_valid = len(all_errors) == 0
    return is_valid, all_errors
