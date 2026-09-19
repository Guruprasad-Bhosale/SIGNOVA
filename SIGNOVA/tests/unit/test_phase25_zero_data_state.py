"""
tests/unit/test_phase25_zero_data_state.py
Unit tests verifying fast-exit and state preservation under zero human data conditions.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from signova.operations.phase25_orchestrator import evaluate_phase25_readiness


class TestPhase25ZeroDataState:
    """Tests verifying fast exit behavior when no human annotations exist."""

    def test_zero_data_fast_exit_state_b(self) -> None:
        """Verify zero data state cleanly returns STATE_B and NOT_STARTED acquisition status."""
        readiness = evaluate_phase25_readiness()
        assert readiness["supervision_state"] == "STATE_B"
        assert readiness["acquisition_status"] == "NOT_STARTED"
        assert readiness["human_annotations"]["total"] == 0
        assert readiness["human_annotations"]["training_eligible"] == 0
        assert readiness["dataset"]["status"] == "NONE"
        assert not readiness["authorization"]["phase19_authorized"]
        assert not readiness["operational_status"]["training_ready"]
        assert readiness["final_state"] == "STATE_B"
        assert "Acquire genuine human sequential ISL annotations" in readiness["next_physical_action"]
