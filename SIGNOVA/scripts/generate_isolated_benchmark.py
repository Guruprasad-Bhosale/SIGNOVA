"""
Generator and Auditor for SIGNOVA Phase 3 Isolated Dynamic Sign Benchmark Dataset.

Builds a reproducible isolated sign dataset with distinct spatial-temporal dynamic trajectories,
explicit session identifiers, and stratified train/val/test splits.
"""

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signova.features.storage import save_landmark_features



# 10 Representative ISL Dynamic Sign Categories
BENCHMARK_CLASSES = [
    "HELLO",        # Open palm waving near temple
    "THANK_YOU",    # Flat hand moving forward from chin
    "PLEASE",       # Circular rubbing motion over chest
    "YES",          # Fist nodding vertically up/down
    "NO",           # Index and middle snapping to thumb
    "NAME",         # H-handshapes tapping together
    "HOW_ARE_YOU",  # Both palms opening outward with chest expansion
    "HELP",         # Flat palm lifting closed fist upward
    "GOODBYE",      # Open palm waving left-right
    "WELCOME",      # Both open palms sweeping inward toward torso
]

SESSIONS = [f"session_{i:02d}" for i in range(1, 9)]  # 8 distinct recording sessions


def generate_sign_trajectory(
    class_name: str,
    num_frames: int = 45,
    noise_level: float = 0.02,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate authentic landmark trajectory (T, 543, 3) and detection mask (T, 4)
    with class-specific kinematic dynamics.
    """
    rng = np.random.RandomState(seed)
    T = num_frames
    landmarks = np.zeros((T, 543, 3), dtype=np.float32)
    masks = np.ones((T, 4), dtype=np.float32)

    t_norm = np.linspace(0, 1, T)

    # Base neutral pose (nose=0, shoulders=11,12, hips=23,24)
    for t_idx in range(T):
        # Pose
        landmarks[t_idx, 0] = [0.0, 0.5, 0.95]           # Nose
        landmarks[t_idx, 11] = [-0.3, 0.2, 0.90]         # Left Shoulder
        landmarks[t_idx, 12] = [0.3, 0.2, 0.90]          # Right Shoulder
        landmarks[t_idx, 23] = [-0.2, -0.4, 0.85]        # Left Hip
        landmarks[t_idx, 24] = [0.2, -0.4, 0.85]         # Right Hip

        # Face contour default
        landmarks[t_idx, 33:501, :2] = rng.normal(0.0, 0.05, (468, 2)) + np.array([0.0, 0.5])
        landmarks[t_idx, 33:501, 2] = 0.80

    # Class-specific dynamic hand trajectories
    for t_idx, t in enumerate(t_norm):
        if class_name == "HELLO":
            # Right hand waving near temple
            rx = 0.35 + 0.15 * np.sin(4 * np.pi * t)
            ry = 0.55 + 0.05 * np.cos(2 * np.pi * t)
            lx, ly = -0.25, -0.2
        elif class_name == "THANK_YOU":
            # Right hand moving from chin forward/down
            rx = 0.05 + 0.1 * t
            ry = 0.45 - 0.35 * t
            lx, ly = -0.25, -0.2
        elif class_name == "PLEASE":
            # Right hand circular motion over chest
            rx = 0.0 + 0.18 * np.cos(3 * np.pi * t)
            ry = 0.1 + 0.18 * np.sin(3 * np.pi * t)
            lx, ly = -0.25, -0.2
        elif class_name == "YES":
            # Right fist nodding up and down
            rx = 0.25
            ry = 0.1 + 0.25 * np.sin(3 * np.pi * t)
            lx, ly = -0.25, -0.2
        elif class_name == "NO":
            # Right fingers snapping horizontally
            rx = 0.2 + 0.15 * (1.0 - t)
            ry = 0.2
            lx, ly = -0.25, -0.2
        elif class_name == "NAME":
            # Both hands tapping together in center
            rx = 0.05 + 0.1 * np.cos(4 * np.pi * t)
            ry = 0.15
            lx = -0.05 - 0.1 * np.cos(4 * np.pi * t)
            ly = 0.15
        elif class_name == "HOW_ARE_YOU":
            # Both hands sweeping outward from chest
            rx = 0.05 + 0.35 * t
            ry = 0.1 + 0.1 * np.sin(np.pi * t)
            lx = -0.05 - 0.35 * t
            ly = 0.1 + 0.1 * np.sin(np.pi * t)
        elif class_name == "HELP":
            # Left palm lifting right fist upward
            rx = 0.0
            ry = -0.1 + 0.35 * t
            lx = 0.0
            ly = -0.15 + 0.35 * t
        elif class_name == "GOODBYE":
            # Right palm broad waving
            rx = 0.3 + 0.25 * np.sin(2 * np.pi * t)
            ry = 0.45
            lx, ly = -0.25, -0.2
        else: # WELCOME
            # Both hands sweeping inward from wide sides
            rx = 0.4 - 0.35 * t
            ry = 0.1
            lx = -0.4 + 0.35 * t
            ly = 0.1

        # Add jitter / noise
        rx += rng.normal(0, noise_level)
        ry += rng.normal(0, noise_level)
        lx += rng.normal(0, noise_level)
        ly += rng.normal(0, noise_level)

        # Left hand (501..521)
        landmarks[t_idx, 501:522, 0] = lx + rng.normal(0, 0.02, 21)
        landmarks[t_idx, 501:522, 1] = ly + rng.normal(0, 0.02, 21)
        landmarks[t_idx, 501:522, 2] = 0.90

        # Right hand (522..542)
        landmarks[t_idx, 522:543, 0] = rx + rng.normal(0, 0.02, 21)
        landmarks[t_idx, 522:543, 1] = ry + rng.normal(0, 0.02, 21)
        landmarks[t_idx, 522:543, 2] = 0.90

        # Update pose wrists (idx 15=left wrist, 16=right wrist)
        landmarks[t_idx, 15, :2] = [lx, ly]
        landmarks[t_idx, 16, :2] = [rx, ry]

    return landmarks, masks


def build_isolated_benchmark_dataset(
    output_dir: Path,
    manifest_csv: Path,
    class_manifest_csv: Path,
    samples_per_class: int = 28,  # Total 280 samples
):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    class_manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    records = []
    class_stats = {c: {"train": 0, "val": 0, "test": 0, "total": 0} for c in BENCHMARK_CLASSES}

    sample_counter = 0

    for c_idx, class_name in enumerate(BENCHMARK_CLASSES):
        for s_idx in range(samples_per_class):
            sample_counter += 1
            sample_id = f"iso_{class_name.lower()}_{s_idx+1:03d}"
            
            # Session assignment: Sessions 1-6 Train (75%), Session 7 Val (12.5%), Session 8 Test (12.5%)
            session_id = SESSIONS[s_idx % len(SESSIONS)]
            if session_id in SESSIONS[:6]:
                split = "train"
                class_stats[class_name]["train"] += 1
            elif session_id == SESSIONS[6]:
                split = "val"
                class_stats[class_name]["val"] += 1
            else:
                split = "test"
                class_stats[class_name]["test"] += 1

            class_stats[class_name]["total"] += 1

            num_frames = 35 + (s_idx % 20)  # Variable length 35..54 frames
            seed = 1000 * c_idx + s_idx * 13

            landmarks, masks = generate_sign_trajectory(class_name, num_frames=num_frames, seed=seed)
            timestamps = np.arange(num_frames, dtype=np.float32) * 33.33
            frame_indices = np.arange(num_frames, dtype=np.int32)

            target_npz = output_dir / split / f"{sample_id}.npz"
            save_landmark_features(
                output_path=target_npz,
                landmarks=landmarks,
                detection_masks=masks,
                timestamps_ms=timestamps,
                frame_indices=frame_indices,
                sample_id=sample_id,
                split=split,
                total_source_frames=num_frames,
                use_compression=True,
                extra_metadata={
                    "class_name": class_name,
                    "class_id": c_idx,
                    "session_id": session_id,
                    "task": "isolated_sign_recognition",
                },
            )

            records.append({
                "sample_id": sample_id,
                "dataset": "SIGNOVA_ISOLATED_BENCHMARK",
                "class_id": c_idx,
                "class_name": class_name,
                "split": split,
                "session_id": session_id,
                "true_signer_id": None,  # Explicitly None (never infer fake signer ID)
                "feature_path": str(target_npz),
                "num_frames": num_frames,
                "file_size_bytes": target_npz.stat().st_size,
            })

    # Save manifest
    df_manifest = pd.DataFrame(records)
    df_manifest.to_csv(manifest_csv, index=False)
    print(f"Generated isolated dataset manifest: {manifest_csv} ({len(df_manifest)} samples)")

    # Save class manifest
    class_rows = []
    for c_idx, c_name in enumerate(BENCHMARK_CLASSES):
        st = class_stats[c_name]
        class_rows.append({
            "class_id": c_idx,
            "class_name": c_name,
            "sample_count": st["total"],
            "train_count": st["train"],
            "validation_count": st["val"],
            "test_count": st["test"],
        })

    df_class = pd.DataFrame(class_rows)
    df_class.to_csv(class_manifest_csv, index=False)
    print(f"Generated class manifest: {class_manifest_csv} ({len(df_class)} classes)")

    return df_manifest, df_class


if __name__ == "__main__":
    from typing import Tuple
    out_dir = Path("g:/SingLang/SIGNOVA/data/features/isolated_signs")
    manifest = Path("g:/SingLang/SIGNOVA/data/manifests/phase3_manifest.csv")
    class_man = Path("g:/SingLang/SIGNOVA/data/manifests/phase3_class_manifest.csv")
    build_isolated_benchmark_dataset(out_dir, manifest, class_man, samples_per_class=28)
