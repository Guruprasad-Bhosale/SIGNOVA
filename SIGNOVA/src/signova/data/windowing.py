"""
Temporal Windowing and Split Leakage Control for SIGNOVA.

Provides sliding window extraction for continuous sign sequences,
ensuring strict video-level split isolation so that overlapping windows
from a source video never leak across train/val/test splits.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from signova.features.feature_groups import LandmarkGroup, slice_landmark_tensor
from signova.features.storage import load_landmark_features


class SlidingWindowExtractor:
    """
    Extracts sliding temporal windows from continuous sequence tensors.
    """

    def __init__(
        self,
        window_size: int = 64,
        stride: int = 32,
        min_valid_ratio: float = 0.5,
        padding_mode: str = "zero",
    ):
        """
        Args:
            window_size: Number of frames in each window.
            stride: Step size between consecutive windows.
            min_valid_ratio: Minimum fraction of non-empty frames required for window acceptance.
            padding_mode: How to handle the trailing remainder ('zero' or 'drop').
        """
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive integers.")
        self.window_size = window_size
        self.stride = stride
        self.min_valid_ratio = min_valid_ratio
        self.padding_mode = padding_mode

    def extract_windows(
        self,
        features: torch.Tensor,
        detection_mask: Optional[torch.Tensor] = None,
        timestamps_ms: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Extracts windows from a sequence of shape (T, num_joints, 3) or (T, input_dim).

        Returns:
            List of dictionaries containing:
            - 'features': (window_size, ...)
            - 'padding_mask': (window_size,) bool
            - 'detection_mask': (window_size, 4) float
            - 'timestamps_ms': (window_size,) float
            - 'start_frame': int
            - 'end_frame': int
        """
        T = features.shape[0]
        windows = []

        if T == 0:
            return windows

        # If sequence is shorter than window_size
        if T < self.window_size:
            if self.padding_mode == "drop":
                return windows
            # Pad to window_size
            pad_shape = list(features.shape)
            pad_shape[0] = self.window_size
            padded_feat = torch.zeros(pad_shape, dtype=features.dtype)
            padded_feat[:T] = features

            pad_mask = torch.zeros(self.window_size, dtype=torch.bool)
            pad_mask[:T] = True

            if detection_mask is not None:
                pad_det = torch.zeros((self.window_size, detection_mask.shape[1]), dtype=detection_mask.dtype)
                pad_det[:T] = detection_mask
            else:
                pad_det = torch.ones((self.window_size, 4), dtype=torch.float32)

            if timestamps_ms is not None:
                pad_ts = torch.zeros(self.window_size, dtype=timestamps_ms.dtype)
                pad_ts[:T] = timestamps_ms
            else:
                pad_ts = torch.arange(self.window_size, dtype=torch.float32) * 33.33

            windows.append({
                "features": padded_feat,
                "padding_mask": pad_mask,
                "detection_mask": pad_det,
                "timestamps_ms": pad_ts,
                "start_frame": 0,
                "end_frame": T,
                "valid_frames": T,
            })
            return windows

        # Sliding window generation
        start_idx = 0
        while start_idx < T:
            end_idx = min(start_idx + self.window_size, T)
            actual_len = end_idx - start_idx

            if actual_len < self.window_size and self.padding_mode == "drop":
                break

            pad_shape = list(features.shape)
            pad_shape[0] = self.window_size
            win_feat = torch.zeros(pad_shape, dtype=features.dtype)
            win_feat[:actual_len] = features[start_idx:end_idx]

            win_mask = torch.zeros(self.window_size, dtype=torch.bool)
            win_mask[:actual_len] = True

            if detection_mask is not None:
                win_det = torch.zeros((self.window_size, detection_mask.shape[1]), dtype=detection_mask.dtype)
                win_det[:actual_len] = detection_mask[start_idx:end_idx]
            else:
                win_det = torch.ones((self.window_size, 4), dtype=torch.float32)

            if timestamps_ms is not None:
                win_ts = torch.zeros(self.window_size, dtype=timestamps_ms.dtype)
                win_ts[:actual_len] = timestamps_ms[start_idx:end_idx]
            else:
                win_ts = torch.arange(start_idx, start_idx + self.window_size, dtype=torch.float32) * 33.33

            valid_ratio = float(actual_len) / self.window_size
            if valid_ratio >= self.min_valid_ratio:
                windows.append({
                    "features": win_feat,
                    "padding_mask": win_mask,
                    "detection_mask": win_det,
                    "timestamps_ms": win_ts,
                    "start_frame": start_idx,
                    "end_frame": end_idx,
                    "valid_frames": actual_len,
                })

            if end_idx >= T:
                break

            start_idx += self.stride

        return windows


def verify_window_split_leakage(
    manifest_df: pd.DataFrame,
    source_video_col: str = "sample_id",
    split_col: str = "split",
) -> Dict[str, Any]:
    """
    Verifies that no source video has windows allocated across more than one split.

    Returns summary audit dict with status, video counts, and any violating IDs.
    """
    if source_video_col not in manifest_df.columns or split_col not in manifest_df.columns:
        raise ValueError(f"DataFrame must contain '{source_video_col}' and '{split_col}' columns.")

    grouped = manifest_df.groupby(source_video_col)[split_col].nunique()
    violating_videos = grouped[grouped > 1].index.tolist()

    is_clean = len(violating_videos) == 0
    split_counts = manifest_df[split_col].value_counts().to_dict()
    unique_videos = manifest_df[source_video_col].nunique()

    return {
        "leakage_detected": not is_clean,
        "is_leakage_free": is_clean,
        "total_windows": len(manifest_df),
        "total_unique_source_videos": unique_videos,
        "split_distribution": split_counts,
        "violating_video_count": len(violating_videos),
        "violating_videos": violating_videos,
    }
