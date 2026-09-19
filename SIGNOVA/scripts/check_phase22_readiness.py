#!/usr/bin/env python3
"""
SIGNOVA Phase 22 - Supervision & CTC Readiness Gate Check.

Evaluates human acquisition, qualification, verified annotations,
CTC feasibility, dataset freeze status, and Phase 19/21 readiness gates.
"""

import sys
import json
import argparse
from pathlib import Path

# Ensure src is on pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from signova.operations.phase22_orchestrator import evaluate_phase22_readiness


def parse_args():
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 22 Readiness Checker")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--json", action="store_true", help="Output JSON result")
    return parser.parse_args()


def main():
    args = parse_args()
    result = evaluate_phase22_readiness(args.data_root)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 65)
        print("SIGNOVA Phase 22 -- Readiness & Gate Evaluation")
        print("=" * 65)
        print(f"Acquisition Mode:        {result.get('acquisition_mode')}")
        print(f"Acquisition Status:      {result.get('acquisition_status')}")
        print(f"Annotator Count:         {result.get('annotator_count')}")
        print(f"Qualified Count:         {result.get('qualified_annotator_count')}")
        print(f"Total Annotations:       {result.get('total_annotations')}")
        print(f"Verified Annotations:    {result.get('verified_annotations')}")
        print(f"Training Eligible:       {result.get('training_eligible_count')}")
        print(f"CTC Feasible:            {result.get('ctc_feasible_count')}")
        print(f"Dataset Status:          {result.get('dataset_status')}")
        print(f"Phase 19 Gate State:     {result.get('phase19_gate_state')}")
        print(f"Phase 19 Authorized:     {result.get('phase19_authorized')}")
        print(f"Phase 21 Training:       {result.get('phase21_training_status')}")
        print(f"Final State:             {result.get('final_state')}")
        print(f"Next Physical Action:    {result.get('next_physical_action')}")
        print("=" * 65)


if __name__ == "__main__":
    main()
