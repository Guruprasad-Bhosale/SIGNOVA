"""
Genuine Continuous ISL CTC Model Training CLI for SIGNOVA Phase 21.

Trains BiGRU Continuous CTC model on genuine human-annotated sequential ISL data.
STRICTLY REFUSES TO TRAIN unless the canonical Phase 19 readiness gate authorizes training.
"""

import argparse
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase21_orchestrator import Phase21Orchestrator


def main():
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 21 Real CTC Training")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension (default: 128)")
    parser.add_argument("--num-layers", type=int, default=2, help="BiGRU layers (default: 2)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--tag", type=str, default="baseline", help="Run tag name (default: baseline)")
    parser.add_argument("--allow-random-split", action="store_true", help="Allow fallback random split")
    args = parser.parse_args()

    print("===========================================================")
    print(" SIGNOVA PHASE 21 -- GENUINE CTC MODEL TRAINING")
    print("===========================================================")

    orch = Phase21Orchestrator(workspace_root=WORKSPACE_ROOT)

    # 1. Gate Inspection
    snapshot = orch.create_gate_snapshot()
    if snapshot["fast_exit_required"]:
        print(f"\n[!] TRAINING REFUSED: Supervision State is {snapshot['supervision_state']}.")
        print(f"[!] Reason: {snapshot['training_authorization'].get('reason')}")
        print("\n[!] Zero checkpoints, zero optimizer states, and zero fake metrics will be generated.")
        print("\nNext Action: Acquire and verify genuine human sequential ISL annotations through the existing annotation platform.")
        print("\n===========================================================")
        print(f" Training safely terminated. State: {snapshot['supervision_state']}")
        print("===========================================================\n")
        return

    print("\n[+] Canonical Gate Check Passed: Supervision State is AUTHORIZED.")
    print(f"    Starting genuine CTC training with seed={args.seed}, epochs={args.epochs}, lr={args.lr} ...\n")

    try:
        res = orch.train_real_ctc(
            batch_size=args.batch_size,
            learning_rate=args.lr,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            seed=args.seed,
            allow_random_split=args.allow_random_split,
            tag=args.tag,
        )
    except Exception as e:
        print(f"\n[!] Training Execution Error: {e}")
        return

    if res["status"] != "TRAINED":
        print(f"\n[!] Training Blocked: {res.get('reason')}")
        return

    print("\n[+] Real CTC Training Completed Successfully!")
    print(f"  Run Directory:    {res['run_dir']}")
    print(f"  Checkpoint Path:  {res['checkpoint_path']}")
    print(f"  Checkpoint SHA:   {res['checkpoint_sha256']}")
    eval_m = res["evaluation"]
    print(f"  Held-out TER:     {eval_m.get('ter')}")
    print(f"  Exact Match Rate: {eval_m.get('exact_match') * 100:.1f}%")

    print("\n===========================================================")
    print(f" Checkpoint generated and saved with full provenance.")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
