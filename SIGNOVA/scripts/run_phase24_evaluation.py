#!/usr/bin/env python3
"""
SIGNOVA Phase 24 - Held-Out Test Evaluation & Error Analysis.

Evaluates trained checkpoint on held-out test split and writes
phase24_error_analysis.json. Refuses if no verified checkpoint exists.
"""

import sys
import json
import argparse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase24_orchestrator import Phase24Orchestrator


def parse_args():
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 24 Held-Out Evaluation")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--json", action="store_true", help="Output JSON result")
    return parser.parse_args()


def main():
    args = parse_args()
    orch = Phase24Orchestrator(workspace_root=WORKSPACE_ROOT, data_root=args.data_root)

    print("=" * 65)
    print(" SIGNOVA Phase 24 -- Held-Out Evaluation & Error Analysis")
    print("=" * 65)

    res = orch.run_held_out_evaluation()

    if args.json:
        print(json.dumps(res, indent=2))
        return

    if res.get("evaluation_status") != "COMPLETED":
        print(f"[!] Evaluation Status: {res.get('evaluation_status')}")
        print(f"[!] Message: {res.get('message', res.get('error'))}")
    else:
        ev = res.get("evaluation", {})
        err = res.get("error_analysis", {})
        print(f"[+] Evaluation Status: COMPLETED")
        print(f"  * Total Error Rate (TER): {ev.get('ter', 'N/A')}")
        print(f"  * Exact Match Rate:       {ev.get('exact_match', 'N/A')}")
        print(f"  * Token F1 Score:         {ev.get('token_f1', 'N/A')}")
        print("-" * 65)
        print("Error Taxonomy:")
        print(f"  * Insertions:             {err.get('errors_by_type', {}).get('insertions', 0)}")
        print(f"  * Deletions:              {err.get('errors_by_type', {}).get('deletions', 0)}")
        print(f"  * Substitutions:          {err.get('errors_by_type', {}).get('substitutions', 0)}")

    print("=" * 65)


if __name__ == "__main__":
    main()
