"""
Dataset Preparation CLI for SIGNOVA Phase 21.

Constructs the canonical real ISL dataset from genuine human annotations.
If Phase 19 reports no data or blocked status, this script FAST-EXITS safely without training or modifying state.
"""

import argparse
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase21_orchestrator import Phase21Orchestrator


def main():
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 21 Dataset Preparation")
    parser.add_argument("--allow-random-split", action="store_true", help="Explicitly permit fallback random split when signer/session metadata is insufficient")
    args = parser.parse_args()

    print("===========================================================")
    print(" SIGNOVA PHASE 21 -- DATASET PREPARATION")
    print("===========================================================")

    orch = Phase21Orchestrator(workspace_root=WORKSPACE_ROOT)
    
    # Check gate snapshot first
    snapshot = orch.create_gate_snapshot()
    if snapshot["fast_exit_required"]:
        print(f"\n[!] FAST EXIT: Supervision State is {snapshot['supervision_state']}.")
        print(f"[!] Real CTC training is BLOCKED: {snapshot['training_authorization'].get('reason')}")
        print("\nNext Action: Acquire and verify genuine human sequential ISL annotations through the existing annotation platform.")
        print("\n===========================================================")
        print(f" Dataset preparation finished safely. State: {snapshot['supervision_state']}")
        print("===========================================================\n")
        return

    try:
        prep = orch.build_genuine_dataset(allow_random_split=args.allow_random_split)
    except ValueError as e:
        print(f"\n[!] Split Error: {e}")
        return

    if prep["status"] == "FAST_EXIT_BLOCKED":
        print(f"\n[!] FAST EXIT: {prep['reason']}")
        return

    samples = prep["samples"]
    vocab = prep["vocabulary"]
    splits = prep["splits"]
    fp = prep["dataset_fingerprint"]

    print(f"\n[+] Genuine Dataset Prepared Successfully:")
    print(f"  Total Samples:        {len(samples)}")
    print(f"  Vocabulary Size:      {vocab.size} tokens")
    print(f"  Split Strategy:       {splits['strategy']} ({splits['status']})")
    print(f"  Train Samples:        {len(splits['train_sample_ids'])}")
    print(f"  Val Samples:          {len(splits['val_sample_ids'])}")
    print(f"  Test Samples:         {len(splits['test_sample_ids'])}")
    print(f"  Dataset SHA-256:      {fp['dataset_sha256']}")

    print("\n===========================================================")
    print(f" Dataset preparation completed. Ready for genuine training.")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
