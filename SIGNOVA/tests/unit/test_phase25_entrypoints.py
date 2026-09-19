"""
tests/unit/test_phase25_entrypoints.py
Unit tests verifying CLI script execution, flag parsing, and dashboard output in Phase 25.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pytest


class TestPhase25Entrypoints:
    """Tests verifying CLI argument parsing and execution across Phase 25 scripts."""

    def test_check_phase25_script(self) -> None:
        """Verify check_phase25.py runs and reports current state."""
        cmd = [sys.executable, "scripts/check_phase25.py", "--json"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0
        assert "STATE_B" in proc.stdout

    def test_run_phase25_verification_script(self) -> None:
        """Verify run_phase25_verification.py displays multi-dimensional dashboard."""
        cmd = [sys.executable, "scripts/run_phase25_verification.py"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0
        assert "SIGNOVA PHASE 25" in proc.stdout
        assert "44 / 44" in proc.stdout
        assert "STATE_B" in proc.stdout

    def test_run_signova_phase25_options(self) -> None:
        """Verify run_signova.py supports Phase 25 flags."""
        cmd = [sys.executable, "run_signova.py", "--phase25-check"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0
        assert "Phase 25" in proc.stdout
