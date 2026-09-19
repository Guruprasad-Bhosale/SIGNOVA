"""
scripts/run_phase25_build_dataset.py
Controlled CLI dataset formation runner constructing candidate dataset from eligible human annotations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signova.operations.phase25_orchestrator import Phase25Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 25 Dataset Formation Runner")
    parser.add_argument("--version", type=str, default="phase25_v001", help="Dataset version identifier")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio (default: 0.1)")
    parser.add_argument("--data-root", type=str, default=None, help="Optional data root directory")
    parser.add_argument("--json", action="store_true", help="Output raw JSON summary")
    args = parser.parse_args()

    orchestrator = Phase25Orchestrator(data_root=args.data_root)
    result = orchestrator.build_dataset(
        dataset_version=args.version,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=================================================================")
    print(" SIGNOVA Phase 25 -- Dataset Formation Execution")
    print("=================================================================")
    print(f"Status:            {result['dataset_status']}")
    print(f"Dataset Version:   {result.get('dataset_version')}")
    print(f"Eligible Samples:  {result.get('total_samples', 0)}")
    if result.get("dataset_status") == "DATASET_READY":
        print(f"Train / Val / Test:{result.get('train_count')} / {result.get('val_count')} / {result.get('test_count')}")
        print(f"Leakage Audit:     {result.get('leakage_status')}")
        print(f"Manifest Path:     {result.get('manifest_path')}")
    else:
        print(f"Message:           {result.get('message')}")
    print("=================================================================")


if __name__ == "__main__":
    main()
