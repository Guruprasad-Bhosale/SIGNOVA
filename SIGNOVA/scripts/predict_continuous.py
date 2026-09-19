#!/usr/bin/env python3
"""
Continuous Sign Language Inference CLI for SIGNOVA.

Runs windowed temporal modeling on continuous ISL video or extracted .npz features,
generating per-frame temporal embeddings, candidate boundary proposals, and sequence diagnostics.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signova.features.feature_groups import LandmarkGroup
from signova.features.storage import load_landmark_features
from signova.inference.streaming import WindowedOfflineInference
from signova.models.continuous_encoder import ContinuousTemporalEncoder
from signova.models.ctc_recognizer import CTCContinuousRecognizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run continuous temporal sequence inference on ISL video/features.")
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Path to pre-extracted .npz feature file.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to raw continuous video file (will extract landmarks if needed).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to model checkpoint (.pt).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="gru",
        choices=["gru", "tcn"],
        help="Temporal backbone architecture to instantiate if no checkpoint is supplied.",
    )
    parser.add_argument(
        "--landmark-group",
        type=str,
        default="hands_pose",
        choices=["hands", "hands_pose", "full"],
        help="Anatomical landmark group to process.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=64,
        help="Sliding window length in frames (default: 64).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
        help="Sliding window stride in frames (default: 32).",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to save JSON prediction results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.features and not args.video:
        print("[ERROR] Must provide either --features or --video argument.")
        sys.exit(1)

    print("=" * 70)
    print("SIGNOVA CONTINUOUS SEQUENCE INFERENCE")
    print("=" * 70)

    # 1. Load landmarks
    if args.features:
        feat_path = Path(args.features)
        if not feat_path.is_file():
            print(f"[ERROR] Features file not found: {feat_path}")
            sys.exit(1)
        raw_lm, det_mask, timestamps, frame_indices, meta = load_landmark_features(feat_path)
        fps = meta.get("fps", 30.0) if meta else 30.0
    else:
        video_path = Path(args.video)
        if not video_path.is_file():
            print(f"[ERROR] Video file not found: {video_path}")
            sys.exit(1)
        from signova.features.mediapipe_extractor import MediaPipeHolisticExtractor
        extractor = MediaPipeHolisticExtractor()
        res = extractor.extract_from_video(video_path)
        raw_lm = res.landmarks
        timestamps = res.timestamps_ms
        fps = res.metadata.fps if res.metadata else 30.0

    print(f"Loaded Landmark Sequence: {raw_lm.shape[0]} frames ({raw_lm.shape[0] / fps:.2f}s) @ {fps} fps")

    # 2. Build or load continuous model
    num_landmarks = 75 if args.landmark_group == "hands_pose" else (42 if args.landmark_group == "hands" else 543)
    if args.checkpoint and Path(args.checkpoint).is_file():
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model = ContinuousTemporalEncoder(backbone=args.backbone, num_landmarks=num_landmarks)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded checkpoint from: {args.checkpoint}")
    else:
        model = ContinuousTemporalEncoder(backbone=args.backbone, num_landmarks=num_landmarks)
        print(f"Instantiated {args.backbone.upper()} Continuous Temporal Encoder ({num_landmarks} landmarks)")

    # 3. Run Windowed Offline Inference
    pipeline = WindowedOfflineInference(
        model=model,
        window_size=args.window_size,
        stride=args.stride,
        landmark_group=args.landmark_group,
    )

    results = pipeline.process_sequence(raw_lm, fps=fps)

    print("\n--- CONTINUOUS INFERENCE TIMELINE & CANDIDATE BOUNDARIES ---")
    print(f"Total Frames Processed : {results['total_frames']}")
    print(f"Total Windows          : {results['window_count']}")
    print(f"Embedding Dimensions   : {results['embedding_dimension']}")
    print(f"Candidate Segments     : {len(results['candidate_segments'])}")

    print("\nCandidate Action / Sign Segments (Heuristic Proposals):")
    for idx, seg in enumerate(results["candidate_segments"], 1):
        print(
            f"  [{idx:02d}] {seg['start_sec']:05.2f}s - {seg['end_sec']:05.2f}s "
            f"(Frames {seg['start_frame']:03d}..{seg['end_frame']:03d}, Dur: {seg['duration_sec']:.2f}s) "
            f"-> Status: {seg['label_status']} (Mean Energy: {seg['mean_energy']:.3f})"
        )

    # Clean embeddings for JSON export (remove raw numpy arrays)
    export_data = {
        "source": args.features or args.video,
        "total_frames": results["total_frames"],
        "duration_sec": results["duration_sec"],
        "window_count": results["window_count"],
        "embedding_dimension": results["embedding_dimension"],
        "candidate_segments": results["candidate_segments"],
        "candidate_boundaries": results["candidate_boundaries"],
        "mode": results["mode"],
        "scientific_disclaimer": "Heuristic motion proposals for visualization. Not ground-truth boundaries.",
    }

    if args.json_output:
        out_p = Path(args.json_output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        print(f"\n[OK] Saved prediction results to {out_p}")

    print("=" * 70)


if __name__ == "__main__":
    main()
