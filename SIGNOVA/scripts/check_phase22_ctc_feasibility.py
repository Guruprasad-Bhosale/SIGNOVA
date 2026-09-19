#!/usr/bin/env python3
"""
SIGNOVA Phase 22 - CTC Feasibility Inspector.

Evaluates sample-level and dataset-level repeated-token CTC feasibility
under the exact equation:
    T_required = L + sum(I(y_i == y_{i+1})) <= T_features
"""

import sys
import json
import argparse
from pathlib import Path

# Ensure src is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from signova.operations.phase22_orchestrator import Phase22Orchestrator


def parse_args():
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 22 CTC Feasibility Inspector")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--json", action="store_true", help="Output JSON result")
    return parser.parse_args()


def main():
    args = parse_args()
    orchestrator = Phase22Orchestrator(args.data_root)
    result = orchestrator.check_ctc_feasibility()

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 65)
        print("SIGNOVA Phase 22 -- CTC Feasibility Inspection")
        print("=" * 65)
        print(f"Total Samples Evaluated: {result.get('total_samples', 0)}")
        print(f"Feasible Samples:        {result.get('feasible_samples', 0)}")
        print(f"Infeasible Samples:      {result.get('infeasible_samples', 0)}")
        print(f"Dataset CTC Feasible:    {result.get('dataset_ctc_feasible', False)}")
        print("-" * 65)

        sample_evals = result.get("sample_evaluations", [])
        if not sample_evals:
            print("No verified samples available for CTC feasibility check.")
        else:
            for s in sample_evals:
                status = "PASS" if s.get("ctc_feasible") else "FAIL"
                print(f"[{status}] Sample: {s.get('sample_id')} | T_features: {s.get('t_features')} | T_required: {s.get('t_required')} | Reason: {s.get('reason')}")

        print("=" * 65)


if __name__ == "__main__":
    main()
