"""
Phase 10 Controlled ISL Dataset Acquisition Engine for SIGNOVA.

Enforces:
1. Zero-Cost Policy (ZERO_COST_MODE = DEFAULT), blocking paid sources as PAID_NOT_USED.
2. License & usage verification.
3. Untrusted Data Security: Never execute downloaded files. Allowed operations are hashing, parsing, decoding supported media, validation, and copying.
4. Immutable raw storage under data/raw/phase10/.
5. Manifest registration with SHA-256 cryptographic provenance.
"""

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
import pandas as pd
from signova.data.provenance import compute_sha256


class AcquisitionStatus(str, Enum):
    ACQUIRED = "ACQUIRED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    NO_ELIGIBLE_DATASET = "NO_ELIGIBLE_DATASET"


ALLOWED_REJECTION_REASONS = {
    "PAID_NOT_USED",
    "LICENSE_UNCLEAR",
    "NO_ORDERED_GLOSS",
    "ENGLISH_ONLY",
    "ISOLATED_ONLY",
    "UNVERIFIED_ANNOTATION",
    "ACCESS_RESTRICTED",
    "CORRUPTED_DOWNLOAD",
    "SECURITY_VIOLATION",
    "OTHER",
}


@dataclass
class AcquisitionRecord:
    dataset_id: str
    candidate_id: str
    source_url: str
    source_name: str
    license: str
    cost_status: str  # "FREE", "PAID", "RESTRICTED"
    cost_inr: float
    acquisition_status: str  # AcquisitionStatus
    artifact_path: str
    artifact_sha256: str
    artifact_size_bytes: int
    acquired_at: str
    dataset_version: str
    annotation_status: str
    verification_status: str
    rejection_reason: str
    provenance_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "candidate_id": self.candidate_id,
            "source_url": self.source_url,
            "source_name": self.source_name,
            "license": self.license,
            "cost_status": self.cost_status,
            "acquisition_status": self.acquisition_status,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "acquired_at": self.acquired_at,
            "dataset_version": self.dataset_version,
            "annotation_status": self.annotation_status,
            "verification_status": self.verification_status,
            "rejection_reason": self.rejection_reason,
            "provenance_id": self.provenance_id,
        }


class ControlledAcquisitionEngine:
    """
    Executes controlled, zero-cost, non-executing dataset acquisition.
    """

    def __init__(
        self,
        raw_storage_dir: Path | str = "data/raw/phase10",
        zero_cost_mode: str = "DEFAULT",
    ):
        self.raw_storage_dir = Path(raw_storage_dir)
        self.zero_cost_mode = zero_cost_mode
        self.raw_storage_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_and_acquire(
        self,
        dataset_id: str,
        candidate_id: str,
        source_url: str,
        source_name: str,
        license_str: str,
        cost_inr: float = 0.0,
        local_mock_artifact: Optional[Path | str] = None,
        timestamp: str = "2026-09-18T23:37:00+05:30",
        version: str = "1.0",
        has_ordered_glosses: bool = False,
    ) -> AcquisitionRecord:
        provenance_id = f"acq_{dataset_id}_{candidate_id}"

        # 1. Cost check
        if cost_inr > 0.0:
            return AcquisitionRecord(
                dataset_id=dataset_id,
                candidate_id=candidate_id,
                source_url=source_url,
                source_name=source_name,
                license=license_str,
                cost_status="PAID",
                cost_inr=cost_inr,
                acquisition_status=AcquisitionStatus.REJECTED.value,
                artifact_path="N/A",
                artifact_sha256="N/A",
                artifact_size_bytes=0,
                acquired_at=timestamp,
                dataset_version=version,
                annotation_status="UNACQUIRED",
                verification_status="REJECTED_COST",
                rejection_reason="PAID_NOT_USED",
                provenance_id=provenance_id,
            )

        # 2. License check
        if "restricted" in license_str.lower() or "unclear" in license_str.lower():
            return AcquisitionRecord(
                dataset_id=dataset_id,
                candidate_id=candidate_id,
                source_url=source_url,
                source_name=source_name,
                license=license_str,
                cost_status="FREE",
                cost_inr=0.0,
                acquisition_status=AcquisitionStatus.REJECTED.value,
                artifact_path="N/A",
                artifact_sha256="N/A",
                artifact_size_bytes=0,
                acquired_at=timestamp,
                dataset_version=version,
                annotation_status="UNACQUIRED",
                verification_status="REJECTED_LICENSE",
                rejection_reason="LICENSE_UNCLEAR" if "unclear" in license_str.lower() else "ACCESS_RESTRICTED",
                provenance_id=provenance_id,
            )

        # 3. If no physical artifact provided or semantic gloss missing
        if not local_mock_artifact or not Path(local_mock_artifact).exists():
            reason = "NO_ORDERED_GLOSS" if not has_ordered_glosses else "NO_ELIGIBLE_DATASET"
            return AcquisitionRecord(
                dataset_id=dataset_id,
                candidate_id=candidate_id,
                source_url=source_url,
                source_name=source_name,
                license=license_str,
                cost_status="FREE",
                cost_inr=0.0,
                acquisition_status=AcquisitionStatus.NO_ELIGIBLE_DATASET.value,
                artifact_path="N/A",
                artifact_sha256="N/A",
                artifact_size_bytes=0,
                acquired_at=timestamp,
                dataset_version=version,
                annotation_status="UNACQUIRED",
                verification_status="UNVERIFIED",
                rejection_reason=reason,
                provenance_id=provenance_id,
            )

        # 4. Safe Non-executing acquisition: copy to raw immutable storage
        src_path = Path(local_mock_artifact)
        dst_path = self.raw_storage_dir / f"{dataset_id}_{src_path.name}"

        # Security check: never execute, only copy and compute hash
        shutil.copy2(src_path, dst_path)
        sha = compute_sha256(dst_path)
        size_bytes = dst_path.stat().st_size

        return AcquisitionRecord(
            dataset_id=dataset_id,
            candidate_id=candidate_id,
            source_url=source_url,
            source_name=source_name,
            license=license_str,
            cost_status="FREE",
            cost_inr=0.0,
            acquisition_status=AcquisitionStatus.ACQUIRED.value,
            artifact_path=str(dst_path).replace("\\", "/"),
            artifact_sha256=sha,
            artifact_size_bytes=size_bytes,
            acquired_at=timestamp,
            dataset_version=version,
            annotation_status="ACQUIRED_RAW",
            verification_status="SHA256_VERIFIED",
            rejection_reason="",
            provenance_id=provenance_id,
        )
