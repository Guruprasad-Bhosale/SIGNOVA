"""
Temporal sequence sampling, padding, and decimation routines for SIGNOVA.
"""

from typing import List, Optional, Tuple, Union
import numpy as np


def compute_sample_indices(
    total_frames: int,
    target_frames: Optional[int] = None,
    source_fps: float = 30.0,
    target_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> List[int]:
    """
    Compute frame indices to sample based on target FPS and maximum frame limits.
    """
    if total_frames <= 0:
        return []

    # 1. Target FPS Decimation
    if target_fps and target_fps > 0 and target_fps < source_fps:
        stride = max(1, int(round(source_fps / target_fps)))
        indices = list(range(0, total_frames, stride))
    else:
        indices = list(range(total_frames))

    # 2. Maximum / Target Frame Subsampling
    limit = target_frames or max_frames
    if limit and len(indices) > limit:
        sampled_pos = np.linspace(0, len(indices) - 1, limit).astype(int)
        indices = [indices[p] for p in sampled_pos]

    return indices


def sample_sequence_uniformly(
    sequence: np.ndarray,
    target_length: int = 128,
) -> np.ndarray:
    """
    Subsample or interpolate a temporal landmark sequence uniformly to exactly target_length frames.
    """
    T = sequence.shape[0]
    if T == 0:
        return np.zeros((target_length,) + sequence.shape[1:], dtype=sequence.dtype)
    if T == target_length:
        return sequence

    indices = np.linspace(0, T - 1, target_length).astype(int)
    return sequence[indices]


def pad_or_crop_sequence(
    sequence: np.ndarray,
    target_len: int = 128,
    pad_value: float = 0.0,
    crop_mode: str = "uniform",
) -> np.ndarray:
    """
    Adjust sequence length to target_len via zero-padding or uniform temporal cropping.
    """
    T = sequence.shape[0]
    if T == target_len:
        return sequence

    if T < target_len:
        pad_shape = (target_len - T,) + sequence.shape[1:]
        padding = np.full(pad_shape, pad_value, dtype=sequence.dtype)
        return np.concatenate([sequence, padding], axis=0)

    # T > target_len
    if crop_mode == "uniform":
        return sample_sequence_uniformly(sequence, target_length=target_len)
    elif crop_mode == "center":
        start = (T - target_len) // 2
        return sequence[start : start + target_len]
    else:
        return sequence[:target_len]
