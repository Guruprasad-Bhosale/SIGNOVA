#!/usr/bin/env python3
"""
SIGNOVA Phase 23 - Granular Data Accounting & Quality Report.

Reports 14-point batch data accounting, double annotation status,
and sample-level CTC deficit metrics.
"""

import sys
import json
import argparse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase23_orchestrator import Phase23Orchestrator


def parse_args():
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 23 Data Accounting")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--json", action="store_true", help="Output JSON result")
    return parser.parse_args()


def main():
    args = parse_args()
    orch = Phase23Orchestrator(workspace_root=WORKSPACE_ROOT, data_root=args.data_root)
    acc = orch.get_data_accounting()

    if args.json:
        print(json.dumps(acc, indent=2))
    else:
        print("=" * 65)
        print(" SIGNOVA Phase 23 -- Data Accounting & Deficit Report")
        print("=" * 65)
        print(f"Total Discovered:       {acc.get('total_discovered', 0)}")
        print(f"Valid Annotations:      {acc.get('annotations_valid', 0)}")
        print(f"Invalid Annotations:    {acc.get('annotations_invalid', 0)}")
        print(f"Submitted:              {acc.get('submitted', 0)}")
        print(f"Verified:               {acc.get('verified', 0)}")
        print(f"Rejected:               {acc.get('rejected', 0)}")
        print(f"Revision Required:      {acc.get('revision_required', 0)}")
        print(f"Training Eligible:      {acc.get('training_eligible', 0)}")
        print(f"Training Ineligible:    {acc.get('training_ineligible', 0)}")
        print("-" * 65)

        double = acc.get("double_annotation", {})
        print("Double Annotation Summary:")
        print(f"  * Double Assigned:    {double.get('double_assigned', 0)}")
        print(f"  * Double Completed:   {double.get('double_completed', 0)}")
        print(f"  * Agreement Status:   {double.get('agreement_status', 'NOT_COMPUTABLE')}")
        print(f"  * Agreement Rate:     {double.get('agreement_rate', 0.0)}")
        print("-" * 65)

        rejections = acc.get("rejection_reasons", {})
        if rejections:
            print("Rejection Reasons Breakdown:")
            for r, c in rejections.items():
                print(f"  * {r}: {c}")
            print("-" * 65)

        deficits = acc.get("worst_ctc_deficits", [])
        if deficits:
            print("Worst Sample CTC Temporal Deficits:")
            for d in deficits:
                print(f"  * Sample: {d.get('sample_id')} | Length: {d.get('gloss_length')} | Req: {d.get('required_timesteps')} | Avail: {d.get('available_timesteps')} | Deficit: {d.get('deficit')}")
        else:
            print("Zero CTC temporal deficit samples detected.")
        print("=" * 65)


if __name__ == "__main__":
    main()
