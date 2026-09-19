"""
Phase 12 Signer & Session Leakage Auditor for SIGNOVA.

Audits:
- Signer independence across train/val/test splits
- Session independence across splits
- Checksum / Video ID duplicate collisions
- Missing metadata characterization
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from signova.annotation.schema import VideoAnnotation


@dataclass
class LeakageAuditReport:
    total_samples: int
    train_count: int
    val_count: int
    test_count: int
    signer_independent: bool
    session_independent: bool
    signer_overlap: List[str]
    session_overlap: List[str]
    duplicate_checksums: List[str]
    duplicate_samples: List[str]
    missing_signer_metadata_count: int
    missing_session_metadata_count: int
    audit_status: str  # "PASSED", "WARNING", "FAILED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "train_count": self.train_count,
            "val_count": self.val_count,
            "test_count": self.test_count,
            "signer_independent": self.signer_independent,
            "session_independent": self.session_independent,
            "signer_overlap": self.signer_overlap,
            "session_overlap": self.session_overlap,
            "duplicate_checksums": self.duplicate_checksums,
            "duplicate_samples": self.duplicate_samples,
            "missing_signer_metadata_count": self.missing_signer_metadata_count,
            "missing_session_metadata_count": self.missing_session_metadata_count,
            "audit_status": self.audit_status,
        }


def audit_phase12_leakage(annotations: List[VideoAnnotation]) -> LeakageAuditReport:
    """Audits a collection of annotations for split leakage and metadata completeness."""
    train_annots = [a for a in annotations if a.dataset_split == "train"]
    val_annots = [a for a in annotations if a.dataset_split == "val"]
    test_annots = [a for a in annotations if a.dataset_split == "test"]

    train_signers: Set[str] = set()
    val_signers: Set[str] = set()
    test_signers: Set[str] = set()

    train_sessions: Set[str] = set()
    val_sessions: Set[str] = set()
    test_sessions: Set[str] = set()

    seen_checksums: Dict[str, str] = {}
    duplicate_checksums: List[str] = []
    seen_samples: Set[str] = set()
    duplicate_samples: List[str] = []

    missing_signers = 0
    missing_sessions = 0

    for a in annotations:
        if a.sample_id in seen_samples:
            duplicate_samples.append(a.sample_id)
        seen_samples.add(a.sample_id)

        meta = a.metadata or {}
        chk = meta.get("source_checksum")
        if chk and chk != "UNKNOWN":
            if chk in seen_checksums and seen_checksums[chk] != a.sample_id:
                duplicate_checksums.append(f"Checksum {chk} shared by {seen_checksums[chk]} and {a.sample_id}")
            seen_checksums[chk] = a.sample_id

        signer = meta.get("signer_id", "UNKNOWN")
        session = meta.get("session_id", "UNKNOWN")

        if signer == "UNKNOWN":
            missing_signers += 1
        if session == "UNKNOWN":
            missing_sessions += 1

        if a.dataset_split == "train":
            if signer != "UNKNOWN":
                train_signers.add(signer)
            if session != "UNKNOWN":
                train_sessions.add(session)
        elif a.dataset_split == "val":
            if signer != "UNKNOWN":
                val_signers.add(signer)
            if session != "UNKNOWN":
                val_sessions.add(session)
        elif a.dataset_split == "test":
            if signer != "UNKNOWN":
                test_signers.add(signer)
            if session != "UNKNOWN":
                test_sessions.add(session)

    # Check overlaps
    signer_overlap = list(train_signers.intersection(test_signers).union(train_signers.intersection(val_signers)))
    session_overlap = list(train_sessions.intersection(test_sessions).union(train_sessions.intersection(val_sessions)))

    signer_indep = (len(signer_overlap) == 0 and len(train_signers) > 0 and len(test_signers) > 0)
    session_indep = (len(session_overlap) == 0 and len(train_sessions) > 0 and len(test_sessions) > 0)

    status = "PASSED"
    if duplicate_samples or duplicate_checksums:
        status = "FAILED"
    elif signer_overlap:
        status = "WARNING"

    return LeakageAuditReport(
        total_samples=len(annotations),
        train_count=len(train_annots),
        val_count=len(val_annots),
        test_count=len(test_annots),
        signer_independent=signer_indep,
        session_independent=session_indep,
        signer_overlap=signer_overlap,
        session_overlap=session_overlap,
        duplicate_checksums=duplicate_checksums,
        duplicate_samples=duplicate_samples,
        missing_signer_metadata_count=missing_signers,
        missing_session_metadata_count=missing_sessions,
        audit_status=status,
    )
