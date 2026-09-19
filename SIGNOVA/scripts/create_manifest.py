#!/usr/bin/env python3
"""
SIGNOVA Manifest Generator
Generates canonical CSV and JSON manifests for supported datasets.
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
    parser = argparse.ArgumentParser(description="Generate dataset manifests for SIGNOVA.")
    parser.add_argument("--dataset", type=str, default="isltranslate", choices=["isltranslate", "include"], help="Dataset to build manifest for.")
    parser.add_argument("--output-dir", type=str, default="data/manifests", help="Output directory for manifests.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio.")
    args = parser.parse_args()

    cm = ConfigManager()
    out_dir = (project_root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building manifest for dataset: {args.dataset}")
    if args.dataset == "isltranslate":
        dataset_path = (project_root.parent / "ISLTranslate-main").resolve()
        csv_path = dataset_path / "data" / "ISLTranslate.csv"
        adapter = ISLTranslateAdapter(root_dir=dataset_path, csv_path=csv_path)
    else:
        print(f"Dataset {args.dataset} requires external media paths.")
        return

    manifest = adapter.build_manifest(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    is_valid, errors = validate_manifest(manifest, check_files_exist=False)
    if not is_valid:
        print(f"Manifest validation failed with {len(errors)} errors:")
        for err in errors[:5]:
            print(f"  - {err}")
        return

    csv_out = out_dir / f"{args.dataset}_manifest.csv"
    json_out = out_dir / f"{args.dataset}_manifest.json"

    manifest.to_csv(csv_out)
    manifest.to_json(json_out)

    print(f"[OK] Successfully wrote {len(manifest)} entries:")
    print(f"  - CSV : {csv_out}")
    print(f"  - JSON: {json_out}")
    print("Summary:", manifest.summary())


if __name__ == "__main__":
    main()
