"""
scripts/run_phase25_freeze_dataset.py
Controlled CLI dataset freeze tool generating cryptographic fingerprint locks.
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
    parser = argparse.ArgumentParser(description="SIGNOVA Phase 25 Dataset Freeze Tool")
    parser.add_argument("--version", type=str, default="phase25_v001", help="Dataset version identifier")
    parser.add_argument("--data-root", type=str, default=None, help="Optional data root directory")
    parser.add_argument("--json", action="store_true", help="Output raw JSON summary")
    args = parser.parse_args()

    orchestrator = Phase25Orchestrator(data_root=args.data_root)
    result = orchestrator.freeze_dataset(dataset_version=args.version)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("=================================================================")
    print(" SIGNOVA Phase 25 -- Dataset Freeze & Fingerprint Lock")
    print("=================================================================")
    print(f"Status:            {result['status']}")
    print(f"Dataset Version:   {result.get('dataset_version')}")
    if "composite_sha256" in result:
        print(f"Composite SHA-256: {result['composite_sha256']}")
        print(f"Manifest Path:     {result['manifest_path']}")
        print(f"Freeze Lock:       {result['lock_path']}")
    elif "message" in result:
        print(f"Message:           {result['message']}")
    print("=================================================================")


if __name__ == "__main__":
    main()
