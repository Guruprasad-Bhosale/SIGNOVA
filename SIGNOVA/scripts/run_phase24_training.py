#!/usr/bin/env python3
"""
SIGNOVA Phase 24 - Gated Real CTC Training Execution.

Enforces:
1. Canonical Phase 19 Gate == AUTHORIZED
2. Dataset / Model compatibility gate == PASSED
3. Hard --train confirmation flag requirement

Delegates actual CTC training to the canonical Phase 21 trainer.
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
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 24 Gated CTC Training Execution")
    parser.add_argument("--train", action="store_true", help="Explicit confirmation flag required to execute real CTC training")
    parser.add_argument("--allow-random-split", action="store_true", help="Explicit confirmation if fallback RANDOM split is used")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--data-root", default="data", help="Root data directory")
    parser.add_argument("--json", action="store_true", help="Output JSON result")
    return parser.parse_args()


def main():
    args = parse_args()
    orch = Phase24Orchestrator(workspace_root=WORKSPACE_ROOT, data_root=args.data_root)

    print("=" * 65)
    print(" SIGNOVA Phase 24 -- Gated Real CTC Training Execution")
    print("=" * 65)

    res = orch.run_gated_training(
        train_flag=args.train,
        allow_random_split=args.allow_random_split,
        epochs=args.epochs,
        lr=args.lr,
    )

    if args.json:
        print(json.dumps(res, indent=2))
        return

    print(f"Supervision State:      {res.get('supervision_state')}")
    print(f"Phase 19 Authorized:    {'YES' if res.get('authorized') else 'NO'}")
    print(f"Experiment Status:      {res.get('experiment_status')}")
    print(f"Status Reason:          {res.get('reason', 'Execution initiated.')}")
    print("-" * 65)

    if res.get("experiment_status") == "NOT_STARTED":
        print("[!] TRAINING ACTION: NOT REQUESTED")
        print("[!] To execute training on authorized data, explicitly provide --train:")
        print("    python scripts/run_phase24_training.py --train")
    elif res.get("experiment_status") == "BLOCKED":
        print(f"[!] TRAINING BLOCKED: {res.get('reason')}")
    elif res.get("experiment_status") == "COMPLETED":
        print("[+] TRAINING COMPLETED SUCCESSFULLY via Phase 21.")
        print(f"[+] Run details: {res.get('run_result')}")
    elif res.get("experiment_status") == "FAILED":
        print(f"[!] TRAINING FAILED: {res.get('run_result')}")

    print("=" * 65)


if __name__ == "__main__":
    main()
