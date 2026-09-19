#!/usr/bin/env python3
"""
SIGNOVA Phase 23 - CTC Readiness & Dataset Freezing Gate.

Evaluates dataset qualification, versioned dataset freezing,
and compatibility with Phase 19 readiness gate.
"""

import sys
import json
import argparse
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase23_orchestrator import Phase23Orchestrator
from signova.pilot.constants import (
    SPLIT_STRATEGY_RANDOM,
    SPLIT_STRATEGY_SESSION_INDEPENDENT,
    SPLIT_STRATEGY_SIGNER_INDEPENDENT,
    SPLIT_STRATEGY_SOURCE_GROUP_INDEPENDENT,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 23 CTC Readiness & Dataset Freeze")
    parser.add_argument("--freeze", action="store_true", help="Freeze training-eligible dataset into versioned artifact")
    parser.add_argument("--version-tag", help="Explicit version tag (e.g., phase23_dataset_v001)")
    parser.add_argument("--split-strategy", default=SPLIT_STRATEGY_SIGNER_INDEPENDENT, choices=[
        SPLIT_STRATEGY_SIGNER_INDEPENDENT,
        SPLIT_STRATEGY_SESSION_INDEPENDENT,
        SPLIT_STRATEGY_SOURCE_GROUP_INDEPENDENT,
        SPLIT_STRATEGY_RANDOM,
    ], help="Split strategy (default: SIGNER_INDEPENDENT)")
    parser.add_argument("--allow-random-split", action="store_true", help="Explicit confirmation required if using RANDOM split")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--json", action="store_true", help="Output JSON result")
    return parser.parse_args()


def main():
    args = parse_args()
    orch = Phase23Orchestrator(workspace_root=WORKSPACE_ROOT, data_root=args.data_root)

    if args.freeze:
        try:
            res = orch.freeze_phase23_dataset(
                version_tag=args.version_tag,
                split_strategy=args.split_strategy,
                allow_random=args.allow_random_split,
            )
            print(f"[+] Successfully froze dataset: {res.get('dataset_version')} with SHA-256: {res.get('dataset_sha256')}")
        except Exception as e:
            print(f"[!] Error during dataset freeze: {e}", file=sys.stderr)
            sys.exit(1)

    dash = orch.get_dashboard_summary()

    if args.json:
        print(json.dumps(dash, indent=2))
    else:
        print("=" * 65)
        print(" SIGNOVA Phase 23 -- CTC Readiness & Dataset Summary")
        print("=" * 65)
        print(f"Supervision State:      {dash.get('supervision_state')}")
        print(f"Training Readiness:     {dash.get('training_readiness')}")
        print(f"Dataset Status:         {dash.get('dataset', {}).get('status')}")
        print(f"Dataset Version:        {dash.get('dataset', {}).get('version')}")
        print(f"Dataset Fingerprint:    {dash.get('dataset', {}).get('fingerprint')}")
        print(f"Split Strategy:         {dash.get('dataset', {}).get('split')}")
        print(f"Phase 19 Gate State:    {dash.get('phase19', {}).get('state')}")
        print(f"Phase 19 Authorized:    {dash.get('phase19', {}).get('authorized')}")
        print(f"Phase 21 Training:      {dash.get('phase21', {}).get('training_status')}")
        print("-" * 65)
        print(f"Next Physical Action:   {dash.get('next_physical_action')}")
        print("=" * 65)


if __name__ == "__main__":
    main()
