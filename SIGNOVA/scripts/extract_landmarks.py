#!/usr/bin/env python3
"""
Landmark Feature Extraction CLI for SIGNOVA.

Extracts 543 skeletal landmarks per frame from video files into .npz feature archives.
Enforces strict safety limits (explicit --allow-bulk required for >100 samples),
tracks progress and metrics, evaluates extraction health, and produces feature manifests.
"""

import argparse
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signova.data.cache import VideoCacheManager
from signova.data.remote import RemoteVideoResolver
from signova.preprocessing.video import RobustVideoDecoder
from signova.preprocessing.normalization import normalize_landmark_sequence
from signova.features.mediapipe_extractor import MediaPipeHolisticExtractor
from signova.features.quality import LandmarkQualityEvaluator
from signova.features.storage import EXTRACTOR_SCHEMA_VERSION, save_landmark_features
from signova.features.visualization import create_pilot_visualization_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Holistic landmarks from video dataset into SIGNOVA .npz archives."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(PROJECT_ROOT / "data" / "manifests" / "isltranslate_manifest.csv"),
        help="Path to dataset manifest CSV.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Dataset split to filter by.",
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Single sample ID to extract.",
    )
    parser.add_argument(
        "--sample-ids",
        type=str,
        default=None,
        help="Comma-separated list of sample IDs to extract.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Maximum number of samples to process (default 25). Set to 0 for unlimited.",
    )
    parser.add_argument(
        "--allow-bulk",
        action="store_true",
        help="Safety gate flag required to extract >100 samples or full splits.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry-run to verify video resolution without running neural extractor.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate landmark overlay visualization images in outputs/visualizations/landmark_pilot/.",
    )
    parser.add_argument(
        "--vis-output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "visualizations" / "landmark_pilot"),
        help="Directory to save landmark visualizations.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "features" / "landmarks"),
        help="Directory to save extracted .npz files.",
    )
    parser.add_argument(
        "--manifest-output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "manifests" / "feature_manifest.csv"),
        help="Path to output feature manifest CSV.",
    )
    parser.add_argument(
        "--failures-output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "manifests" / "feature_failures.csv"),
        help="Path to output failure log CSV.",
    )
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Disable .npz compression (default: compression enabled).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("SIGNOVA Landmark Feature Extraction CLI")
    print(f"Extractor Schema Version: {EXTRACTOR_SCHEMA_VERSION}")
    print(f"Manifest: {args.manifest}")
    print(f"Output Dir: {args.output_dir}")
    print("=" * 70)

    # 1. Enforce Bulk Safety Gate
    effective_limit = args.limit if args.limit > 0 else 999999
    if effective_limit > 100 and not args.allow_bulk:
        print("\n[ERROR] Bulk Extraction Safety Gate Triggered!")
        print("Extracting > 100 samples or running a full split requires the explicit `--allow-bulk` flag.")
        print("This prevents accidental multi-gigabyte disk saturation during pilot phases.")
        print("Usage: python scripts/extract_landmarks.py --limit 25  (or pass --allow-bulk)")
        sys.exit(1)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"[ERROR] Manifest file not found: {manifest_path}")
        sys.exit(1)

    df_manifest = pd.read_csv(manifest_path)
    print(f"Loaded manifest with {len(df_manifest):,} total samples.")

    # 2. Filter Samples
    if args.sample_id:
        df_target = df_manifest[df_manifest["sample_id"] == args.sample_id]
    elif args.sample_ids:
        target_ids = [s.strip() for s in args.sample_ids.split(",") if s.strip()]
        df_target = df_manifest[df_manifest["sample_id"].isin(target_ids)]
    else:
        df_target = df_manifest.copy()
        if args.split != "all":
            df_target = df_target[df_target["split"] == args.split]

    if args.limit > 0:
        df_target = df_target.head(args.limit)

    print(f"Targeting {len(df_target)} sample(s) for processing.")

    # 3. Setup Components
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = Path(args.vis_output_dir)
    if args.visualize:
        vis_dir.mkdir(parents=True, exist_ok=True)

    cache_mgr = VideoCacheManager()
    resolver = RemoteVideoResolver(cache_manager=cache_mgr)
    evaluator = LandmarkQualityEvaluator()
    use_compression = not args.no_compression

    success_records: List[Dict] = []
    failure_records: List[Dict] = []

    # Initialize feature extractor only if not dry-run
    extractor = None
    if not args.dry_run:
        extractor = MediaPipeHolisticExtractor()

    t_start = time.time()

    for idx, (_, row) in enumerate(df_target.iterrows(), 1):
        sample_id = str(row["sample_id"])
        split = str(row.get("split", "train"))
        target_npz = out_dir / split / f"{sample_id}.npz"

        print(f"[{idx}/{len(df_target)}] Processing {sample_id} ({split})...", end=" ")

        # Resolve Video File
        video_path, status = resolver.resolve_video(row.to_dict())
        if video_path is None or not video_path.is_file():
            print(f"[FAILED] Video unresolved: {status}")
            failure_records.append({
                "sample_id": sample_id,
                "split": split,
                "status": status,
                "error": f"Video unavailable locally: {status}",
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            continue

        if args.dry_run:
            print(f"[DRY-RUN OK] Video found at {video_path}")
            success_records.append({
                "sample_id": sample_id,
                "split": split,
                "video_path": str(video_path),
                "status": "DRY_RUN_RESOLVED",
            })
            continue

        # Extract Landmarks
        try:
            decoder = RobustVideoDecoder(video_path)
            meta = decoder.read_metadata()
            if not meta.is_valid:
                print(f"[FAILED] Video invalid: {meta.error}")
                failure_records.append({
                    "sample_id": sample_id,
                    "split": split,
                    "status": "CORRUPT_VIDEO",
                    "error": meta.error,
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
                continue

            # Stream & Extract
            stream = decoder.stream_frames()
            result = extractor.extract_from_video_stream(stream, total_source_frames=meta.total_frames)

            if result.extracted_frames == 0:
                print("[FAILED] 0 frames extracted.")
                failure_records.append({
                    "sample_id": sample_id,
                    "split": split,
                    "status": "EMPTY_EXTRACTION",
                    "error": "No frames decoded from video container",
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
                continue

            # Spatial Normalization
            normalized_landmarks = normalize_landmark_sequence(result.landmarks)

            # Quality Assessment
            quality = evaluator.evaluate(normalized_landmarks, result.detection_masks)

            # Save Features
            saved_path = save_landmark_features(
                output_path=target_npz,
                landmarks=normalized_landmarks,
                detection_masks=result.detection_masks,
                timestamps_ms=result.timestamps_ms,
                frame_indices=result.frame_indices,
                sample_id=sample_id,
                split=split,
                total_source_frames=meta.total_frames,
                use_compression=use_compression,
                extra_metadata=quality.to_dict(),
            )

            # Optional Visualization Grid
            if args.visualize:
                vis_img_path = vis_dir / f"{sample_id}_overlay.png"
                create_pilot_visualization_grid(normalized_landmarks, sample_id, vis_img_path)

            file_size_kb = round(saved_path.stat().st_size / 1024, 1)
            print(f"[OK] {result.extracted_frames} frames -> {saved_path.name} ({file_size_kb} KB, usable={quality.is_usable})")

            success_records.append({
                "sample_id": sample_id,
                "split": split,
                "feature_path": str(saved_path),
                "num_frames": result.extracted_frames,
                "fps": meta.fps,
                "pose_rate": quality.pose_detection_rate,
                "face_rate": quality.face_detection_rate,
                "left_hand_rate": quality.left_hand_detection_rate,
                "right_hand_rate": quality.right_hand_detection_rate,
                "is_usable": quality.is_usable,
                "file_size_bytes": saved_path.stat().st_size,
                "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

        except Exception as exc:
            print(f"[ERROR] Extraction exception: {exc}")
            failure_records.append({
                "sample_id": sample_id,
                "split": split,
                "status": "RUNTIME_ERROR",
                "error": str(exc),
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

    if extractor is not None:
        extractor.close()

    elapsed = round(time.time() - t_start, 2)

    # 4. Save Feature and Failure Manifests
    if success_records:
        df_success = pd.DataFrame(success_records)
        manifest_out = Path(args.manifest_output)
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        if manifest_out.is_file():
            df_existing = pd.read_csv(manifest_out)
            df_success = pd.concat([df_existing, df_success]).drop_duplicates(subset=["sample_id"], keep="last")
        df_success.to_csv(manifest_out, index=False)
        print(f"\nSaved feature manifest to {manifest_out} ({len(df_success)} entries)")

    if failure_records:
        df_fail = pd.DataFrame(failure_records)
        failures_out = Path(args.failures_output)
        failures_out.parent.mkdir(parents=True, exist_ok=True)
        if failures_out.is_file():
            df_existing_fail = pd.read_csv(failures_out)
            df_fail = pd.concat([df_existing_fail, df_fail]).drop_duplicates(subset=["sample_id"], keep="last")
        df_fail.to_csv(failures_out, index=False)
        print(f"Saved failure manifest to {failures_out} ({len(df_fail)} entries)")

    # 5. Print Execution Summary
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print(f"Total Target Samples: {len(df_target)}")
    print(f"Succeeded:            {len(success_records)}")
    print(f"Failed / Missing:     {len(failure_records)}")
    print(f"Elapsed Time:         {elapsed}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
