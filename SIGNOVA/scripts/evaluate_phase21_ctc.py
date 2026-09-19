"""
Evaluation and Error Analysis CLI for SIGNOVA Phase 21 CTC Model.

Evaluates trained checkpoint on held-out genuine test split.
"""

import argparse
import json
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase21_orchestrator import Phase21Orchestrator


def main():
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 21 CTC Evaluation")
    parser.add_argument("--run-dir", type=str, default=None, help="Specific run directory to evaluate")
    args = parser.parse_args()

    print("===========================================================")
    print(" SIGNOVA PHASE 21 -- MODEL EVALUATION & ERROR ANALYSIS")
    print("===========================================================")

    orch = Phase21Orchestrator(workspace_root=WORKSPACE_ROOT)
    
    # Check if run exists
    target_dir = None
    if args.run_dir:
        target_dir = Path(args.run_dir)
    else:
        pointer_file = orch.experiments_dir / "live_model_pointer.json"
        if pointer_file.exists():
            try:
                data = json.loads(pointer_file.read_text(encoding="utf-8"))
                p = Path(data.get("active_run_dir", ""))
                if p.exists():
                    target_dir = p
            except Exception:
                pass
        if target_dir is None and orch.experiments_dir.exists():
            runs = sorted([d for d in orch.experiments_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
            if runs:
                target_dir = runs[-1]

    if target_dir is None or not (target_dir / "evaluation.json").exists():
        print("\n[!] No trained genuine CTC model run found in models/experiments/phase21_real_ctc/.")
        print("[!] Supervision State remains STATE_B (No fake metrics produced).")
        print("\nNext Action: Acquire genuine annotations and train model when authorized.")
        print("\n===========================================================")
        print(" Evaluation finished safely.")
        print("===========================================================\n")
        return

    eval_data = json.loads((target_dir / "evaluation.json").read_text(encoding="utf-8"))
    meta = json.loads((target_dir / "model_metadata.json").read_text(encoding="utf-8"))

    print(f"\n[+] Evaluating Model Run: {target_dir.name}")
    print(f"  Model ID:             {meta.get('model_id')}")
    print(f"  Supervision State:    {meta.get('training_state')}")
    print(f"  Dataset Scale:        {meta.get('dataset_scale')}")
    print(f"  Split Strategy:       {meta.get('split_strategy')}")
    print(f"  Checkpoint SHA-256:   {meta.get('checkpoint_sha256')}")

    print("\n--- Quantitative Metrics ---")
    print(f"  Total Test Samples:   {eval_data.get('total_samples')}")
    print(f"  Total Ref Tokens:     {eval_data.get('total_reference_tokens')}")
    print(f"  Token Error Rate:     {eval_data.get('ter') * 100:.2f}%")
    print(f"  Exact Sequence Match: {eval_data.get('exact_match') * 100:.2f}%")
    print(f"  Insertions:           {eval_data.get('insertions')}")
    print(f"  Deletions:            {eval_data.get('deletions')}")
    print(f"  Substitutions:        {eval_data.get('substitutions')}")

    print("\n--- Sample Predictions (Masked IDs) ---")
    samples = eval_data.get("sample_predictions", [])
    for s in samples[:5]:
        print(f"  * Sample [{s['sample_id']}]:")
        print(f"    Ref:  {' '.join(s['reference']) if s['reference'] else 'EMPTY'}")
        print(f"    Hyp:  {' '.join(s['hypothesis']) if s['hypothesis'] else 'EMPTY'}")
        print(f"    Conf: {s.get('mean_confidence', 0.0)}")

    print("\n===========================================================")
    print(" Evaluation complete.")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
