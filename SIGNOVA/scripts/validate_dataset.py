#!/usr/bin/env python3
"""
SIGNOVA Dataset Validation CLI
Validates annotation integrity, path references, and manifest constraints.
"""

import argparse
import sys
from pathlib import Path

# Add src to pythonpath
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from signova.config.loader import ConfigManager
from signova.data.isltranslate import ISLTranslateAdapter
from signova.data.validation import validate_manifest


def main():
    parser = argparse.ArgumentParser(description="Validate dataset annotations and integrity.")
    parser.add_argument("--dataset", type=str, default="isltranslate", choices=["isltranslate", "include"], help="Dataset to validate.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records to validate (for quick checks).")
    parser.add_argument("--workers", type=int, default=4, help="Worker threads.")
    args = parser.parse_args()

    print("=" * 60)
    print(f"             SIGNOVA DATASET VALIDATION: {args.dataset.upper()}             ")
    print("=" * 60)

    cm = ConfigManager()
    cfg = cm.load_dataset_config()

    if args.dataset == "isltranslate":
        dataset_path = (project_root.parent / "ISLTranslate-main").resolve()
        csv_path = dataset_path / "data" / "ISLTranslate.csv"
        adapter = ISLTranslateAdapter(root_dir=dataset_path, csv_path=csv_path)
    else:
        print(f"Dataset {args.dataset} validation requires external mount.")
        return 0

    print("Step 1: Checking raw dataset files...")
    raw_val = adapter.validate()
    print("  Status:", "OK" if raw_val.get("valid") else "FAILED")
    for k, v in raw_val.items():
        print(f"  - {k}: {v}")

    if not raw_val.get("valid"):
        print("\n[FATAL] Raw dataset validation failed.")
        return 1

    print("\nStep 2: Building and validating canonical manifest...")
    manifest = adapter.build_manifest()
    if args.limit:
        manifest.entries = manifest.entries[:args.limit]
        print(f"  (Limited validation to {args.limit} entries)")

    is_valid, errors = validate_manifest(manifest, check_files_exist=False)
    print(f"  Total manifest entries tested: {len(manifest)}")
    print(f"  Manifest validation result: {'VALID' if is_valid else 'INVALID'}")

    if not is_valid:
        print(f"  Encountered {len(errors)} validation errors:")
        for err in errors[:5]:
            print(f"    - {err}")
        return 1

    print("\nStep 3: Manifest summary metrics:")
    summary = manifest.summary()
    for k, v in summary.items():
        print(f"  - {k}: {v}")

    print("\n" + "=" * 60)
    print(f"Validation successful. Dataset {args.dataset} passed all integrity checks.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
