"""
scripts/check_phase25.py
Lightweight Phase 25 diagnostic and human acquisition readiness check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signova.operations.phase25_orchestrator import evaluate_phase25_readiness


def main() -> None:
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 25 Diagnostic Check")
    parser.add_argument("--json", action="store_true", help="Output raw JSON summary")
    parser.add_argument("--data-root", type=str, default=None, help="Optional data root directory")
    args = parser.parse_args()

    readiness = evaluate_phase25_readiness(data_root=args.data_root)

    if args.json:
        print(json.dumps(readiness, indent=2))
        return

    print("=================================================================")
    print(" SIGNOVA Phase 25 -- Human Annotation Acquisition & Gating Check")
    print("=================================================================")
    print(f"Supervision State:      {readiness['supervision_state']}")
    print(f"Acquisition Status:     {readiness['acquisition_status']}")
    print(f"Human Annotations:      {readiness['human_annotations']['total']} total | {readiness['human_annotations']['training_eligible']} eligible")
    print(f"Dataset State:          {readiness['dataset']['status']}")
    print(f"Phase 19 Authorized:    {'YES' if readiness['authorization']['phase19_authorized'] else 'NO'}")
    print(f"Training Ready:         {'YES' if readiness['operational_status']['training_ready'] else 'NO'}")
    print(f"Phase 24 Status:        {readiness['authorization']['phase24_status']}")
    print(f"Final State:            {readiness['final_state']}")
    print(f"Next Physical Action:   {readiness['next_physical_action']}")
    print("=================================================================")


if __name__ == "__main__":
    main()
