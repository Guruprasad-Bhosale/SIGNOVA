"""
tests/unit/test_phase25_ctc_feasibility.py
Unit tests verifying repeated-token CTC feasibility calculations for Phase 25.
"""

from __future__ import annotations

import pytest

from signova.operations.phase25_orchestrator import Phase25Orchestrator


class TestPhase25CTCFeasibility:
    """Tests verifying sample-level CTC timestep requirements and feasibility calculation."""

    def test_ctc_required_timesteps_calculation(self) -> None:
        """Verify T_req = L + sum(1 for y_i == y_{i+1})."""
        # Distinct tokens: L=3, repeats=0 -> T_req=3
        assert Phase25Orchestrator.calculate_ctc_required_timesteps(["HELLO", "WORLD", "ISL"]) == 3

        # Repeated adjacent tokens: ["A", "A", "B"] -> L=3, repeats=1 -> T_req=4
        assert Phase25Orchestrator.calculate_ctc_required_timesteps(["HELLO", "HELLO", "WORLD"]) == 4

        # All identical: ["A", "A", "A"] -> L=3, repeats=2 -> T_req=5
        assert Phase25Orchestrator.calculate_ctc_required_timesteps(["HELLO", "HELLO", "HELLO"]) == 5

    def test_feasibility_boundary_evaluation(self) -> None:
        """Verify sample feasibility flags under limited available timesteps."""
        orch = Phase25Orchestrator()
        
        # 64 frames available, 3 tokens -> Feasible
        res_ok = orch.check_sample_ctc_feasibility(["HELLO", "WORLD"], available_timesteps=64)
        assert res_ok["ctc_feasible"]

        # Only 2 frames available, 3 required -> Infeasible
        res_fail = orch.check_sample_ctc_feasibility(["HELLO", "HELLO"], available_timesteps=2)
        assert not res_fail["ctc_feasible"]
