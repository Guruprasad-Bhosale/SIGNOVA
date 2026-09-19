"""
Phase 9 Signer and Session Leakage Audit Engine for SIGNOVA.

Measures:
- Signer overlap across splits
- Session overlap across splits
- Duplicate video IDs
- Duplicate gloss sequence overlaps
- Fallback reporting: SIGNER_INDEPENDENT_SPLIT_NOT_POSSIBLE when metadata is missing.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import pandas as pd


@dataclass
class LeakageAuditReport:
    signer_independent_split_possible: bool
    signer_metadata_available: bool
    session_metadata_available: bool
    signer_overlap: Dict[str, List[str]]  # e.g. {"train_val": [...], "train_test": [...]}
    session_overlap: Dict[str, List[str]]
    duplicate_video_ids_across_splits: List[str]
    duplicate_gloss_sequences_across_splits: int
    leakage_detected: bool
    status: str  # "PASSED", "LEAKAGE_DETECTED", "SIGNER_INDEPENDENT_SPLIT_NOT_POSSIBLE"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signer_independent_split_possible": self.signer_independent_split_possible,
            "signer_metadata_available": self.signer_metadata_available,
            "session_metadata_available": self.session_metadata_available,
            "signer_overlap": self.signer_overlap,
            "session_overlap": self.session_overlap,
            "duplicate_video_ids_across_splits": self.duplicate_video_ids_across_splits,
            "duplicate_gloss_sequences_across_splits": self.duplicate_gloss_sequences_across_splits,
            "leakage_detected": self.leakage_detected,
            "status": self.status,
            "details": self.details,
        }


def audit_split_leakage(
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    test_samples: Optional[List[Dict[str, Any]]] = None,
) -> LeakageAuditReport:
    """
    Audits split dictionaries for signer, session, video ID, and gloss sequence leakage.
    """
    test_samples = test_samples or []

    # Check signer presence
    train_signers = {s.get("signer_id") for s in train_samples if s.get("signer_id")}
    val_signers = {s.get("signer_id") for s in val_samples if s.get("signer_id")}
    test_signers = {s.get("signer_id") for s in test_samples if s.get("signer_id")}

    has_signer = bool(train_signers and val_signers)
    signer_overlap = {}
    if has_signer:
        signer_overlap["train_val"] = sorted(list(train_signers.intersection(val_signers)))
        if test_signers:
            signer_overlap["train_test"] = sorted(list(train_signers.intersection(test_signers)))
            signer_overlap["val_test"] = sorted(list(val_signers.intersection(test_signers)))
    else:
        signer_overlap["train_val"] = []

    # Check session presence
    train_sessions = {s.get("session_id") for s in train_samples if s.get("session_id")}
    val_sessions = {s.get("session_id") for s in val_samples if s.get("session_id")}
    test_sessions = {s.get("session_id") for s in test_samples if s.get("session_id")}

    has_session = bool(train_sessions and val_sessions)
    session_overlap = {}
    if has_session:
        session_overlap["train_val"] = sorted(list(train_sessions.intersection(val_sessions)))
        if test_sessions:
            session_overlap["train_test"] = sorted(list(train_sessions.intersection(test_sessions)))
            session_overlap["val_test"] = sorted(list(val_sessions.intersection(test_sessions)))
    else:
        session_overlap["train_val"] = []

    # Video ID duplicates across splits
    train_vids = {s.get("video_id") or s.get("sample_id") for s in train_samples if s.get("video_id") or s.get("sample_id")}
    val_vids = {s.get("video_id") or s.get("sample_id") for s in val_samples if s.get("video_id") or s.get("sample_id")}
    test_vids = {s.get("video_id") or s.get("sample_id") for s in test_samples if s.get("video_id") or s.get("sample_id")}

    dup_vids = list(train_vids.intersection(val_vids))
    if test_vids:
        dup_vids.extend(list(train_vids.intersection(test_vids)))
        dup_vids.extend(list(val_vids.intersection(test_vids)))
    dup_vids = sorted(list(set(dup_vids)))

    # Gloss sequence string matches across splits
    def to_gloss_str(sample: Dict[str, Any]) -> str:
        glosses = sample.get("glosses") or sample.get("ordered_glosses") or []
        return " ".join(glosses) if isinstance(glosses, list) else str(glosses)

    train_seqs = {to_gloss_str(s) for s in train_samples if to_gloss_str(s)}
    val_seqs = {to_gloss_str(s) for s in val_samples if to_gloss_str(s)}
    test_seqs = {to_gloss_str(s) for s in test_samples if to_gloss_str(s)}

    dup_seq_count = len(train_seqs.intersection(val_seqs))
    if test_seqs:
        dup_seq_count += len(train_seqs.intersection(test_seqs))
        dup_seq_count += len(val_seqs.intersection(test_seqs))

    has_signer_leak = any(len(v) > 0 for v in signer_overlap.values())
    has_session_leak = any(len(v) > 0 for v in session_overlap.values())
    has_vid_leak = len(dup_vids) > 0

    leakage_detected = has_signer_leak or has_session_leak or has_vid_leak

    if not has_signer:
        status = "SIGNER_INDEPENDENT_SPLIT_NOT_POSSIBLE"
        signer_possible = False
    elif leakage_detected:
        status = "LEAKAGE_DETECTED"
        signer_possible = True
    else:
        status = "PASSED"
        signer_possible = True

    return LeakageAuditReport(
        signer_independent_split_possible=signer_possible,
        signer_metadata_available=has_signer,
        session_metadata_available=has_session,
        signer_overlap=signer_overlap,
        session_overlap=session_overlap,
        duplicate_video_ids_across_splits=dup_vids,
        duplicate_gloss_sequences_across_splits=dup_seq_count,
        leakage_detected=leakage_detected,
        status=status,
        details={
            "train_samples_count": len(train_samples),
            "val_samples_count": len(val_samples),
            "test_samples_count": len(test_samples),
            "unique_train_signers": len(train_signers),
            "unique_val_signers": len(val_signers),
            "unique_test_signers": len(test_signers),
        },
    )
