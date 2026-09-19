"""
Unit Tests for Phase 4 Continuous Dataset, Collation, and Windowing.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from signova.data.continuous_dataset import ContinuousPadCollate, ContinuousSignDataset
from signova.data.windowing import SlidingWindowExtractor, verify_window_split_leakage
from signova.features.storage import save_landmark_features


@pytest.fixture
def dummy_continuous_manifest(tmp_path: Path):
    manifest_rows = []
    for i in range(4):
        T = 40 + i * 10
        lm = np.random.randn(T, 543, 3).astype(np.float32)
        det = np.ones((T, 4), dtype=np.float32)
        ts = np.arange(T, dtype=np.float32) * 33.33
        fi = np.arange(T, dtype=np.int64)

        feat_file = tmp_path / f"cont_seq_{i:02d}.npz"
        save_landmark_features(
            output_path=feat_file,
            landmarks=lm,
            detection_masks=det,
            timestamps_ms=ts,
            frame_indices=fi,
            sample_id=f"cont_seq_{i:02d}",
            split="train" if i < 3 else "val",
            total_source_frames=T,
            extra_metadata={"fps": 30.0},
        )
        manifest_rows.append({
            "sample_id": f"cont_seq_{i:02d}",
            "feature_path": str(feat_file),
            "split": "train" if i < 3 else "val",
            "session_id": f"sess_{i % 2}",
            "signer_id": "signer_01",
            "num_frames": T,
            "target_sequence": "[1, 2, 3]" if i % 2 == 0 else "[4, 5]",
        })

    df = pd.DataFrame(manifest_rows)
    manifest_csv = tmp_path / "continuous_manifest.csv"
    df.to_csv(manifest_csv, index=False)
    return manifest_csv, df


def test_continuous_dataset_and_collate(dummy_continuous_manifest):
    manifest_csv, _ = dummy_continuous_manifest
    ds = ContinuousSignDataset(manifest=manifest_csv, landmark_group="hands_pose")
    assert len(ds) == 4

    sample = ds[0]
    assert "features" in sample
    assert sample["features"].shape == (40, 75, 3)  # 75 joints for hands_pose
    assert sample["detection_mask"].shape == (40, 4)
    assert sample["target_sequence"] is not None

    collate_fn = ContinuousPadCollate()
    batch = [ds[0], ds[1]]
    collated = collate_fn(batch)

    assert "features" in collated
    assert collated["features"].shape == (2, 50, 75, 3)  # Max length 50
    assert collated["padding_mask"].shape == (2, 50)
    assert collated["padding_mask"][0, 40:].sum() == 0   # Padding is False
    assert collated["padding_mask"][0, :40].all()        # Valid is True
    assert collated["targets"] is not None


def test_sliding_window_extractor():
    T = 100
    features = torch.randn((T, 75, 3), dtype=torch.float32)
    extractor = SlidingWindowExtractor(window_size=32, stride=16)
    windows = extractor.extract_windows(features)

    assert len(windows) > 0
    for w in windows:
        assert w["features"].shape == (32, 75, 3)
        assert w["padding_mask"].shape == (32,)
        assert "start_frame" in w
        assert "end_frame" in w


def test_window_split_leakage_detector():
    # Clean dataset
    clean_df = pd.DataFrame([
        {"sample_id": "vid_1", "window_id": "w1_1", "split": "train"},
        {"sample_id": "vid_1", "window_id": "w1_2", "split": "train"},
        {"sample_id": "vid_2", "window_id": "w2_1", "split": "val"},
    ])
    report_clean = verify_window_split_leakage(clean_df)
    assert report_clean["is_leakage_free"] is True
    assert report_clean["leakage_detected"] is False

    # Leaking dataset
    leaking_df = pd.DataFrame([
        {"sample_id": "vid_1", "window_id": "w1_1", "split": "train"},
        {"sample_id": "vid_1", "window_id": "w1_2", "split": "val"},  # Leak!
        {"sample_id": "vid_2", "window_id": "w2_1", "split": "test"},
    ])
    report_leak = verify_window_split_leakage(leaking_df)
    assert report_leak["is_leakage_free"] is False
    assert report_leak["leakage_detected"] is True
    assert "vid_1" in report_leak["violating_videos"]
