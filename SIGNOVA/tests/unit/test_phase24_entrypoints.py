"""
tests/unit/test_phase24_entrypoints.py
Unit tests verifying CLI parsing and safe execution of all Phase 24 entrypoint scripts.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


class TestPhase24Entrypoints:
    """Tests verifying CLI argument parsing, flags, and fast-exit semantics across Phase 24 scripts."""

    def test_check_phase24_script(self) -> None:
        """Verify check_phase24.py executes and reports STATE_B."""
        cmd = [sys.executable, "scripts/check_phase24.py", "--json"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0
        assert "STATE_B" in proc.stdout

    def test_run_phase24_verification_script(self) -> None:
        """Verify run_phase24_verification.py passes all checks and preserves references."""
        cmd = [sys.executable, "scripts/run_phase24_verification.py"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0
        assert "SIGNOVA PHASE 24" in proc.stdout
        assert "44 / 44" in proc.stdout
        assert "STATE_B" in proc.stdout

    def test_run_phase24_training_without_train_flag_refuses(self) -> None:
        """Verify run_phase24_training.py refuses without explicit --train or when gate is blocked."""
        cmd = [sys.executable, "scripts/run_phase24_training.py"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert "TRAINING BLOCKED" in proc.stdout or "REFUSED" in proc.stdout

    def test_run_phase24_evaluation_without_checkpoint_refuses(self) -> None:
        """Verify run_phase24_evaluation.py refuses without verified checkpoint."""
        cmd = [sys.executable, "scripts/run_phase24_evaluation.py"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert "REFUSED" in proc.stdout or "No verified checkpoint" in proc.stdout or proc.returncode != 0

    def test_run_phase24_live_smoke_test_without_auth_refuses(self) -> None:
        """Verify run_phase24_live_smoke_test.py refuses when LIVE_MODEL_AUTHORIZED != TRUE."""
        cmd = [sys.executable, "scripts/run_phase24_live_smoke_test.py"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert "REFUSED" in proc.stdout or proc.returncode != 0

    def test_run_signova_phase24_options(self) -> None:
        """Verify run_signova.py supports Phase 24 flags and options."""
        cmd = [sys.executable, "run_signova.py", "--phase24-check"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0
        assert "STATE_B" in proc.stdout
