"""
Controlled Dataset Formation and Freeze CLI for SIGNOVA Phase 22.

Builds and freezes dataset_manifest.json, dataset_manifest.csv, and dataset_sha256
from verified, training-eligible human annotations.
"""

import argparse
from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.operations.phase22_orchestrator import Phase22Orchestrator


def main():
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 22 Dataset Formation")
    parser.add_argument("--allow-random-split", action="store_true", help="Explicitly permit fallback random split on limited data")
    args = parser.parse_args()

    print("===========================================================")
    print(" SIGNOVA PHASE 22 -- CONTROLLED DATASET FORMATION")
    print("===========================================================")

    orch = Phase22Orchestrator(workspace_root=WORKSPACE_ROOT)
    try:
        res = orch.build_and_freeze_dataset(allow_random_split=args.allow_random_split)
    except ValueError as ve:
        print(f"\n[!] Split Error: {ve}")
        return

    print(f"\n[+] Dataset Formation Status: {res['status']}")
    print(f"  Dataset State:     {res['dataset_state']}")
    print(f"  Eligible Samples:  {res['eligible_count']}")
    print(f"  Ineligible Samples:{res['ineligible_count']}")

    if res["status"] == "DATASET_FROZEN":
        print(f"  Vocabulary Size:   {res['vocabulary_size']} tokens")
        print(f"  Split Strategy:    {res['split_strategy']}")
        print(f"  Dataset SHA-256:   {res['dataset_sha256']}")
        print(f"  JSON Manifest:     {res['manifest_json_path']}")
        print(f"  CSV Manifest:      {res['manifest_csv_path']}")
    else:
        print("\n[!] Zero training-eligible samples available. Dataset remains unfrozen.")

    print("\n===========================================================\n")


if __name__ == "__main__":
    main()
