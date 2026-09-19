"""
Phase 9 Provenance Chain, Cryptographic Hashing, and Zero-Cost Policy Module.

Enforces:
1. End-to-end cryptographic traceability:
   SOURCE -> DOWNLOADED FILE -> ANNOTATION -> VIDEO -> FEATURE -> MODEL SAMPLE
2. Zero-Cost Policy (ZERO_COST_MODE = DEFAULT), blocking paid sources as PAID_NOT_USED.
"""

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


ZERO_COST_MODE = "DEFAULT"


@dataclass
class ProvenanceNode:
    node_type: str  # "SOURCE", "ACQUISITION", "DOWNLOAD", "RAW_ARTIFACT", "INGESTION", "ANNOTATION", "VIDEO", "FEATURE", "MODEL_SAMPLE"
    identifier: str
    sha256: Optional[str] = None
    parent_provenance_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceChain:
    sample_id: str
    nodes: List[ProvenanceNode] = field(default_factory=list)

    def add_step(
        self,
        node_type: str,
        identifier: str,
        sha256: Optional[str] = None,
        parent_provenance_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ProvenanceChain":
        self.nodes.append(
            ProvenanceNode(
                node_type=node_type,
                identifier=identifier,
                sha256=sha256,
                parent_provenance_id=parent_provenance_id,
                metadata=metadata or {},
            )
        )
        return self

    def is_complete(self) -> bool:
        required_types = {"SOURCE", "ANNOTATION", "VIDEO", "FEATURE"}
        present_types = {n.node_type for n in self.nodes}
        return required_types.issubset(present_types)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "chain_length": len(self.nodes),
            "is_complete": self.is_complete(),
            "nodes": [
                {
                    "node_type": n.node_type,
                    "identifier": n.identifier,
                    "sha256": n.sha256,
                    "parent_provenance_id": n.parent_provenance_id,
                    "metadata": n.metadata,
                }
                for n in self.nodes
            ],
        }


def compute_sha256(file_path: Path | str, chunk_size: int = 65536) -> str:
    """Compute cryptographic SHA-256 hash for a file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for hash calculation: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def generate_checksum_records(
    target_files: List[Path | str],
    dataset_id: str = "workspace",
) -> List[Dict[str, Any]]:
    """Generate SHA-256 records for a list of files."""
    records = []
    for file_path in target_files:
        path = Path(file_path)
        if path.is_file():
            size_bytes = path.stat().st_size
            sha = compute_sha256(path)
            records.append({
                "dataset_id": dataset_id,
                "file": str(path).replace("\\", "/"),
                "sha256": sha,
                "size_bytes": size_bytes,
            })
    return records


class ZeroCostPolicyEnforcer:
    """
    Enforces ZERO_COST_MODE = DEFAULT across all acquisition pipelines.
    Blocks any source or action requiring payment.
    """

    def __init__(self, mode: str = ZERO_COST_MODE):
        self.mode = mode

    def evaluate_acquisition_request(
        self,
        dataset_id: str,
        cost_required: bool,
        cost_inr: float = 0.0,
        license_type: str = "open",
    ) -> Dict[str, Any]:
        if cost_required or cost_inr > 0:
            return {
                "permitted": False,
                "dataset_id": dataset_id,
                "status": "PAID_NOT_USED",
                "reason": "Payment required; zero-cost policy forbids commercial spending.",
                "authorized_spend_inr": 0.0,
            }
        return {
            "permitted": True,
            "dataset_id": dataset_id,
            "status": "APPROVED_FREE",
            "reason": "Free/open resource compatible with zero-cost policy.",
            "authorized_spend_inr": 0.0,
        }
