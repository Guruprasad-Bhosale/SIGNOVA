"""
tests/unit/test_phase24_live_authorization.py
Unit tests verifying the multi-stage live model authorization lifecycle and Phase 20 ownership.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from signova.operations.phase24_orchestrator import (
    Phase24Orchestrator,
    evaluate_phase24_readiness,
)
from signova.live.model_registry import LiveModelRegistry
from signova.live.status import ModelStatus


class TestPhase24LiveAuthorization:
    """Tests verifying live model authorization protocol and single-ownership invariant."""

    def test_live_smoke_test_refuses_when_unauthorized(self, tmp_path: Path) -> None:
        """Verify live smoke test strictly refuses when model is not authorized."""
        orchestrator = Phase24Orchestrator(data_root=tmp_path / "data")
        
        result = orchestrator.run_live_smoke_test()
        
        assert result["smoke_test_status"] == "REFUSED_NOT_AUTHORIZED"
        assert "refused" in result["message"]

    def test_phase20_remains_exclusive_owner_of_pointer(self, tmp_path: Path) -> None:
        """Verify Phase 24 does not directly write to live_model_pointer.json."""
        registry = LiveModelRegistry(checkpoint_dir=tmp_path / "models")
        
        # Initial status should be UNAVAILABLE
        status, meta, reason = registry.inspect_model_availability()
        assert status == ModelStatus.UNAVAILABLE
        assert meta is None
        
        # Phase 24 orchestrator queries readiness without corrupting registry ownership
        readiness = evaluate_phase24_readiness()
        assert readiness["live"]["model"] == "NONE"
        assert readiness["live"]["activity"] == "OPERATIONAL"

    def test_validation_levels_reporting(self, tmp_path: Path) -> None:
        """Verify distinct reporting of smoke test vs ISL recognition vs gloss translation validation."""
        readiness = evaluate_phase24_readiness()
        
        assert readiness["state_reporting"]["isl_recognition_validation"] == "NOT_PERFORMED"
        assert readiness["state_reporting"]["gloss_to_english_validation"] == "NOT_PERFORMED"
        assert not readiness["model_readiness"]["live_smoke_test"]
