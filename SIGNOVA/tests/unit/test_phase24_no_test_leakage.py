"""
tests/unit/test_phase24_no_test_leakage.py
Unit tests verifying strict test-set isolation and leakage prevention for Phase 24.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from signova.data.splits import deterministic_split
from signova.data.leakage_audit import audit_split_leakage
from signova.operations.phase24_orchestrator import Phase24Orchestrator


class TestPhase24NoTestLeakage:
    """Tests ensuring no sample or signer leakage between train, val, and test partitions."""

    def test_sample_level_isolation(self) -> None:
        """Verify train, val, and test splits have mutually exclusive sample IDs."""
        sample_ids = [f"sample_{i:03d}" for i in range(100)]
        
        train, val, test = deterministic_split(
            items=sample_ids,
            id_getter=lambda x: x,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )
        
        train_set = set(train)
        val_set = set(val)
        test_set = set(test)
        
        assert train_set.isdisjoint(test_set), "Train and Test sets must be strictly disjoint"
        assert val_set.isdisjoint(test_set), "Val and Test sets must be strictly disjoint"
        assert train_set.isdisjoint(val_set), "Train and Val sets must be strictly disjoint"
        assert len(train_set) + len(val_set) + len(test_set) == 100

    def test_leakage_audit_engine_detection(self) -> None:
        """Verify audit_split_leakage catches overlapping signers or samples."""
        train_samples = [
            {"video_id": "v1", "signer_id": "signer_1", "gloss_sequence": ["HELLO"]},
            {"video_id": "v2", "signer_id": "signer_2", "gloss_sequence": ["THANK_YOU"]},
        ]
        test_samples_clean = [
            {"video_id": "v3", "signer_id": "signer_3", "gloss_sequence": ["ISL"]},
        ]
        val_samples_clean = [
            {"video_id": "v4", "signer_id": "signer_4", "gloss_sequence": ["SIGN"]},
        ]
        
        report_clean = audit_split_leakage(train_samples, val_samples_clean, test_samples_clean)
        assert not report_clean.leakage_detected
        assert report_clean.status == "PASSED"
        
        # Now introduce signer leakage into test set
        test_samples_leaked = [
            {"video_id": "v5", "signer_id": "signer_1", "gloss_sequence": ["ISL"]},
        ]
        report_leaked = audit_split_leakage(train_samples, val_samples_clean, test_samples_leaked)
        assert report_leaked.leakage_detected
        assert "signer_1" in report_leaked.signer_overlap["train_test"]
