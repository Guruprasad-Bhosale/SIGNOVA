#!/usr/bin/env python3
"""
Feature Integrity & Numerical Health Diagnostic CLI for SIGNOVA.

Scans extracted .npz landmark archives across splits, validates tensor shapes,
checks coordinate bounds, detects NaN/Inf corruptions, and verifies temporal monotonicity.
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signova.features.storage import load_landmark_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify numerical sanity and shape integrity of extracted landmark .npz archives."
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "features" / "landmarks"),
        help="Directory containing extracted .npz files.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(PROJECT_ROOT / "data" / "manifests" / "feature_manifest.csv"),
        help="Feature manifest CSV if available.",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Inspect a specific sample ID.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file diagnostics.",
    )
    return parser.parse_args()


def inspect_feature_file(path: Path) -> Dict:
    result = {
        "file_path": str(path),
        "sample_id": path.stem,
        "is_valid": True,
        "errors": [],
        "num_frames": 0,
        "landmark_shape": None,
        "mask_shape": None,
        "has_nan": False,
        "has_inf": False,
        "coord_min": 0.0,
        "coord_max": 0.0,
        "temporal_monotonic": True,
        "file_size_kb": round(path.stat().st_size / 1024, 2),
    }

    try:
        landmarks, masks, timestamps, frame_indices, metadata = load_landmark_features(path)
        T = landmarks.shape[0]
        result["num_frames"] = T
        result["landmark_shape"] = list(landmarks.shape)
        result["mask_shape"] = list(masks.shape)

        # 1. Shape check
        if landmarks.ndim != 3 or landmarks.shape[1] != 543 or landmarks.shape[2] != 3:
            result["is_valid"] = False
            result["errors"].append(f"Invalid landmark shape: {landmarks.shape}, expected (T, 543, 3)")

        if masks.ndim != 2 or masks.shape[1] != 4:
            result["is_valid"] = False
            result["errors"].append(f"Invalid mask shape: {masks.shape}, expected (T, 4)")

        # 2. NaN / Inf check
        if np.isnan(landmarks).any() or np.isnan(masks).any():
            result["is_valid"] = False
            result["has_nan"] = True
            result["errors"].append("NaN values detected in feature arrays")

        if np.isinf(landmarks).any() or np.isinf(masks).any():
            result["is_valid"] = False
            result["has_inf"] = True
            result["errors"].append("Inf values detected in feature arrays")

        # 3. Coordinate range
        if T > 0:
            coords = landmarks[:, :, :2]
            c_min = float(np.min(coords))
            c_max = float(np.max(coords))
            result["coord_min"] = round(c_min, 4)
            result["coord_max"] = round(c_max, 4)
            if c_min < -15.0 or c_max > 15.0:
                result["is_valid"] = False
                result["errors"].append(f"Extreme coordinate values: [{c_min}, {c_max}]")

        # 4. Temporal Monotonicity
        if len(timestamps) > 1:
            diffs = np.diff(timestamps)
            if np.any(diffs < 0):
                result["is_valid"] = False
                result["temporal_monotonic"] = False
                result["errors"].append("Non-monotonic timestamps detected")

    except Exception as exc:
        result["is_valid"] = False
        result["errors"].append(f"Failed to load file: {exc}")

    return result


def main():
    args = parse_args()
    features_dir = Path(args.features_dir)

    print("=" * 70)
    print("SIGNOVA Feature Integrity & Numerical Health Diagnostic")
    print(f"Scanning directory: {features_dir}")
    print("=" * 70)

    if not features_dir.exists():
        print(f"[ERROR] Features directory does not exist: {features_dir}")
        sys.exit(1)

    files = list(features_dir.rglob("*.npz"))
    if args.sample_id:
        files = [f for f in files if f.stem == args.sample_id]

    if not files:
        print("No .npz feature files found in target directory.")
        sys.exit(0)

    print(f"Discovered {len(files)} feature file(s) across splits.\n")

    results = []
    valid_count = 0
    corrupt_count = 0

    for f in files:
        res = inspect_feature_file(f)
        results.append(res)
        if res["is_valid"]:
            valid_count += 1
            if args.verbose:
                print(f"[PASS] {f.stem}: {res['num_frames']} frames, range [{res['coord_min']}, {res['coord_max']}] ({res['file_size_kb']} KB)")
        else:
            corrupt_count += 1
            print(f"[FAIL] {f.stem}: {'; '.join(res['errors'])}")

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print(f"Total Files Checked:   {len(files)}")
    print(f"Passed Integrity:      {valid_count} ({valid_count / len(files) * 100:.1f}%)")
    print(f"Corrupt / Failed:      {corrupt_count}")
    print("=" * 70)

    if corrupt_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
