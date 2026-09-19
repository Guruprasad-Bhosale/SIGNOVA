"""
tests/unit/test_phase25_authorization_gate.py
Unit tests verifying Phase 19 gate authority and refusal to bypass Phase 19/21/24.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from signova.operations.phase25_orchestrator import evaluate_phase25_readiness


class TestPhase25AuthorizationGate:
    """Tests verifying Phase 19 gate supremacy and training readiness."""

    def test_phase19_remains_sole_authorization_authority(self) -> None:
        """Verify Phase 25 does not bypass Phase 19 gate decision."""
        readiness = evaluate_phase25_readiness()
        assert not readiness["authorization"]["phase19_authorized"]
        assert readiness["supervision_state"] == "STATE_B"
        assert not readiness["operational_status"]["training_ready"]
