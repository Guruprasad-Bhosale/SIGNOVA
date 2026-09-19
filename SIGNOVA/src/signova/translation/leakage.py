"""
Leakage detection and split integrity validation for Translation datasets.

Guiding Principles:
1. Prevents duplicate sample overlap across train, validation, and test splits.
2. Identifies exact source token sequence duplicates.
3. Checks for signer and session leakage across splits when metadata is present.
4. Never manufactures missing signer or session IDs.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from signova.translation.dataset import TranslationSample


@dataclass
class LeakageReport:
    """
    Summary report of data leakage and split contamination checks.
    """
    is_leakage_free: bool
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    train_val_overlap_count: int = 0
    train_test_overlap_count: int = 0
    val_test_overlap_count: int = 0
    duplicate_source_sequences: int = 0
    duplicate_pairs: int = 0
    signer_overlap_detected: bool = False
    session_overlap_detected: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class TranslationLeakageDetector:
    """
    Audits translation datasets and splits for exact duplicates and cross-split contamination.
    """

    def __init__(self):
        pass

    def check_splits(
        self,
        train_samples: Sequence[TranslationSample],
        val_samples: Sequence[TranslationSample],
        test_samples: Sequence[TranslationSample],
    ) -> LeakageReport:
        """
        Perform comprehensive leakage checks across train, validation, and test sets.
        """
        train_pairs = {(tuple(s.source_tokens), s.target_text.strip().lower()): s for s in train_samples}
        val_pairs = {(tuple(s.source_tokens), s.target_text.strip().lower()): s for s in val_samples}
        test_pairs = {(tuple(s.source_tokens), s.target_text.strip().lower()): s for s in test_samples}

        # Exact pair overlap
        train_val_overlap = set(train_pairs.keys()) & set(val_pairs.keys())
        train_test_overlap = set(train_pairs.keys()) & set(test_pairs.keys())
        val_test_overlap = set(val_pairs.keys()) & set(test_pairs.keys())

        # Source sequence duplicate count across entire collection
        all_samples = list(train_samples) + list(val_samples) + list(test_samples)
        source_seq_counts = Counter(tuple(s.source_tokens) for s in all_samples)
        dup_sources = sum(c - 1 for c in source_seq_counts.values() if c > 1)

        pair_counts = Counter((tuple(s.source_tokens), s.target_text.strip().lower()) for s in all_samples)
        dup_pairs = sum(c - 1 for c in pair_counts.values() if c > 1)

        # Signer leakage (only if signer IDs exist)
        train_signers = {s.signer_id for s in train_samples if s.signer_id is not None}
        val_signers = {s.signer_id for s in val_samples if s.signer_id is not None}
        test_signers = {s.signer_id for s in test_samples if s.signer_id is not None}

        signer_overlap = False
        if len(train_signers) > 0 and len(val_signers) > 0:
            if bool(train_signers & val_signers) or bool(train_signers & test_signers):
                signer_overlap = True

        # Session leakage (only if session IDs exist)
        train_sessions = {s.session_id for s in train_samples if s.session_id is not None}
        val_sessions = {s.session_id for s in val_samples if s.session_id is not None}
        test_sessions = {s.session_id for s in test_samples if s.session_id is not None}

        session_overlap = False
        if len(train_sessions) > 0 and len(val_sessions) > 0:
            if bool(train_sessions & val_sessions) or bool(train_sessions & test_sessions):
                session_overlap = True

        is_clean = (
            len(train_val_overlap) == 0
            and len(train_test_overlap) == 0
            and len(val_test_overlap) == 0
        )

        return LeakageReport(
            is_leakage_free=is_clean,
            train_count=len(train_samples),
            val_count=len(val_samples),
            test_count=len(test_samples),
            train_val_overlap_count=len(train_val_overlap),
            train_test_overlap_count=len(train_test_overlap),
            val_test_overlap_count=len(val_test_overlap),
            duplicate_source_sequences=dup_sources,
            duplicate_pairs=dup_pairs,
            signer_overlap_detected=signer_overlap,
            session_overlap_detected=session_overlap,
            details={
                "train_val_overlapping_samples": list(train_val_overlap)[:5],
                "train_test_overlapping_samples": list(train_test_overlap)[:5],
            },
        )
