#!/usr/bin/env python3
"""
SIGNOVA Dataset Inspector & Auditor
Scans the local reference repositories without modifying them and displays statistics.
"""

import os
import sys
from pathlib import Path

# Add src to pythonpath
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from signova.config.loader import ConfigManager
from signova.data.include import INCLUDEAdapter
from signova.data.isltranslate import ISLTranslateAdapter


def inspect_repository(name: str, path: Path):
    print(f"\n==================== {name} ====================")
    print(f"Path: {path.resolve()}")
    if not path.is_dir():
        print(f"Status: Directory does not exist.")
        return

    total_files = 0
    total_size_bytes = 0
    extensions = {}

    for p in path.glob("**/*"):
        if p.is_file():
            total_files += 1
            sz = p.stat().st_size
            total_size_bytes += sz
            ext = p.suffix.lower() or "(no extension)"
            extensions[ext] = extensions.get(ext, 0) + 1

    print(f"Total Files       : {total_files}")
    print(f"Total Size on Disk: {total_size_bytes / (1024*1024):.2f} MB ({total_size_bytes / (1024*1024*1024):.4f} GB)")
    print(f"File Types Breakdown:")
    for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {ext}: {count} files")


def main():
    print("=" * 60)
    print("           SIGNOVA DATASET AUDIT & INSPECTION           ")
    print("=" * 60)

    cm = ConfigManager()
    cfg = cm.load_dataset_config()

    workspace_root = project_root.parent
    isl_trans_path = workspace_root / "ISLTranslate-main"
    isl_ref_path = workspace_root / "isl-translator-main"

    inspect_repository("ISLTranslate Reference Repository", isl_trans_path)
    inspect_repository("isl-translator Reference Implementation", isl_ref_path)

    print("\n------------------------------------------------------------")
    print("DATASET ADAPTER COMPATIBILITY CHECKS")
    print("------------------------------------------------------------")

    # Audit ISLTranslate
    csv_path = isl_trans_path / "data" / "ISLTranslate.csv"
    if csv_path.is_file():
        adapter = ISLTranslateAdapter(root_dir=isl_trans_path, csv_path=csv_path)
        summary = adapter.summary()
        print(f"[ISLTranslate Adapter]")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    else:
        print(f"[ISLTranslate Adapter] CSV not found at {csv_path}")

    # Audit INCLUDE Adapter
    include_adapter = INCLUDEAdapter(root_dir=workspace_root / "data" / "raw" / "include")
    include_summary = include_adapter.summary()
    print(f"\n[INCLUDE Adapter]")
    for k, v in include_summary.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("Audit completed. Reference repositories are read-only and intact.")
    print("=" * 60)


if __name__ == "__main__":
    main()
