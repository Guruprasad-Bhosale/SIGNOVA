#!/usr/bin/env python3
"""
CTC Pre-Flight Data & Supervision Validator for SIGNOVA.

Validates:
1. Target vocabulary integrity & blank token indexing.
2. Input length vs target length feasibility (T_in >= T_target).
3. Missing / empty / corrupted target sequences.
4. NaN / Inf feature anomalies.
5. Split integrity and window leakage.

Fails loudly with clear diagnostics when invalid supervision is detected.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signova.data.continuous_dataset import ContinuousSignDataset
from signova.data.windowing import verify_window_split_leakage


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-flight validator for CTC continuous datasets and manifests.")
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest CSV or JSON containing continuous sequences and optional targets.",
    )
    parser.add_argument(
        "--blank-idx",
        type=int,
        default=0,
        help="Index reserved for the CTC blank token (default: 0).",
    )
    parser.add_argument(
        "--check-features",
        action="store_true",
        help="Whether to load and inspect underlying feature archives for NaN/Inf.",
    )
    return parser.parse_args()


def validate_ctc_data(manifest_path: Path, blank_idx: int = 0, check_features: bool = False) -> dict:
    print("=" * 70)
    print("SIGNOVA CTC PRE-FLIGHT SUPERVISION VALIDATION")
    print("=" * 70)
    print(f"Manifest Path: {manifest_path}")
    print(f"Blank Token Index: {blank_idx}")

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if manifest_path.suffix.lower() == ".json":
        df = pd.read_json(manifest_path)
    else:
        df = pd.read_csv(manifest_path)

    report = {
        "manifest_path": str(manifest_path),
        "total_rows": len(df),
        "columns": list(df.columns),
        "has_target_sequence": "target_sequence" in df.columns,
        "has_feature_path": "feature_path" in df.columns,
        "violations": [],
        "passed": False,
    }

    # 1. Check basic columns
    if not report["has_feature_path"]:
        report["violations"].append("Missing required 'feature_path' column.")

    if not report["has_target_sequence"]:
        report["violations"].append(
            "SUPERVISION MISSING: Manifest has no 'target_sequence' column. "
            "Continuous CTC training is blocked until valid sequential sign labels exist."
        )
        print(f"\n[BLOCKED] {report['violations'][-1]}")
        return report

    # 2. Check targets and length constraints
    invalid_targets = 0
    shorter_inputs = 0
    missing_files = 0
    nan_features = 0

    for idx, row in df.iterrows():
        feat_path = Path(row["feature_path"])
        if not feat_path.is_file():
            missing_files += 1
            continue

        raw_target = row.get("target_sequence")
        if pd.isna(raw_target):
            invalid_targets += 1
            continue

        # Parse target tokens
        tokens = []
        if isinstance(raw_target, str):
            try:
                import ast
                tokens = ast.literal_eval(raw_target)
            except Exception:
                tokens = [int(x.strip()) for x in raw_target.split(",") if x.strip().isdigit()]
        elif isinstance(raw_target, (list, tuple)):
            tokens = list(raw_target)

        target_len = len(tokens)
        if target_len == 0:
            invalid_targets += 1
            continue

        # Check for blank_idx appearing inside actual target tokens
        if blank_idx in tokens:
            report["violations"].append(f"Row {idx}: Target sequence contains blank token ID {blank_idx}.")

        num_frames = row.get("num_frames")
        if pd.notna(num_frames):
            if int(num_frames) < target_len:
                shorter_inputs += 1

        if check_features:
            data = np.load(feat_path)
            lm = data.get("landmarks")
            if lm is not None:
                if np.isnan(lm).any() or np.isinf(lm).any():
                    nan_features += 1

    if missing_files > 0:
        report["violations"].append(f"{missing_files} referenced feature files do not exist.")
    if invalid_targets > 0:
        report["violations"].append(f"{invalid_targets} rows contain empty or unparseable target sequences.")
    if shorter_inputs > 0:
        report["violations"].append(
            f"{shorter_inputs} samples violate CTC length constraint: input frames < target length."
        )
    if nan_features > 0:
        report["violations"].append(f"{nan_features} feature archives contain NaN or Inf values.")

    # 3. Check split leakage if sample_id and split exist
    if "sample_id" in df.columns and "split" in df.columns:
        leak_audit = verify_window_split_leakage(df)
        if leak_audit["leakage_detected"]:
            report["violations"].append(f"Split leakage detected across {leak_audit['violating_video_count']} videos.")
        report["split_leakage_audit"] = leak_audit

    report["passed"] = len(report["violations"]) == 0

    print("-" * 70)
    if report["passed"]:
        print("[SUCCESS] All CTC data integrity checks passed. Dataset is CTC-ready.")
    else:
        print(f"[FAIL] CTC Data Validator detected {len(report['violations'])} issue(s):")
        for v in report["violations"]:
            print(f"  - {v}")
    print("=" * 70)

    return report


def main():
    args = parse_args()
    report = validate_ctc_data(
        manifest_path=Path(args.manifest),
        blank_idx=args.blank_idx,
        check_features=args.check_features,
    )
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
