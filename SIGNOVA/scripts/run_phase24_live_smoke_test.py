#!/usr/bin/env python3
"""
SIGNOVA Phase 24 - Live Model Smoke Test & Integration Gate.

Executes live runtime compatibility checks and validates live model
integration through LiveModelRegistry. Refuses if LIVE_MODEL_AUTHORIZED != True.
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
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 24 Live Model Smoke Test")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--json", action="store_true", help="Output JSON result")
    return parser.parse_args()


def main():
    args = parse_args()
    orch = Phase24Orchestrator(workspace_root=WORKSPACE_ROOT, data_root=args.data_root)

    print("=" * 65)
    print(" SIGNOVA Phase 24 -- Live Model Smoke Test")
    print("=" * 65)

    res = orch.run_live_smoke_test()

    if args.json:
        print(json.dumps(res, indent=2))
        return

    if res.get("smoke_test_status") != "PASSED":
        print(f"[!] Smoke Test Status: {res.get('smoke_test_status')}")
        print(f"[!] Message: {res.get('message')}")
        print("-" * 65)
        print("Readiness Stages:")
        for stage, passed in res.get("stages", {}).items():
            print(f"  * {stage}: {'PASSED' if passed else 'NOT PASSED'}")
    else:
        print(f"[+] Smoke Test Status: PASSED")
        print(f"[+] Live Runtime Status: {res.get('runtime_status')}")

    print("=" * 65)


if __name__ == "__main__":
    main()
