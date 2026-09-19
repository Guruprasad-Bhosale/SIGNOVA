"""
tests/unit/test_phase25_split_leakage.py
Unit tests verifying strict train/val/test split isolation and absence of data leakage in Phase 25.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from signova.data.splits import deterministic_split
from signova.data.leakage_audit import audit_split_leakage


class TestPhase25SplitLeakage:
    """Tests ensuring no train, val, or test leakage in Phase 25 partitions."""

    def test_split_sample_isolation(self) -> None:
        """Verify train, validation, and test partitions are strictly disjoint."""
        samples = [f"sample_{i:04d}" for i in range(50)]
        train, val, test = deterministic_split(
            items=samples,
            id_getter=lambda x: x,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed_salt="phase25_isolation_test",
        )

        train_set = set(train)
        val_set = set(val)
        test_set = set(test)

        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)
        assert train_set.isdisjoint(val_set)
        assert len(train_set) + len(val_set) + len(test_set) == 50
