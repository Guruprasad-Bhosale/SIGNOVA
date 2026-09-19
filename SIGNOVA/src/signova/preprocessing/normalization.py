"""
Landmark spatial normalization routines for SIGNOVA.
Ensures scale and translation invariance across different camera setups and signer positions.
Includes robust multi-tiered fallbacks, NaN/Inf guards, and safe clipping.
"""

from typing import Optional, Tuple
import numpy as np


def normalize_landmarks_frame(
    landmarks: np.ndarray,
    center_anchor: str = "mid_hip",
    scale_anchor: str = "shoulder_dist",
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Normalize spatial coordinates (x, y) of a (543, 3) landmark array.

    Args:
        landmarks: Array of shape (543, 3) where columns are (x, y, visibility/confidence).
        center_anchor: Anchoring mode ('mid_hip', 'nose', 'bbox_center', or 'origin').
        scale_anchor: Scaling mode ('shoulder_dist', 'bbox', or 'unit').
        eps: Small epsilon to prevent division by zero.

    Returns:
        Normalized array of shape (543, 3). Coordinates are zero-centered and scale-standardized.
        Visibility/confidence column is preserved unchanged.
    """
    if landmarks.ndim != 2 or landmarks.shape[0] != 543 or landmarks.shape[1] < 2:
        raise ValueError(f"Invalid landmark shape: {landmarks.shape}. Expected (543, 3).")

    # Clean any NaN / Inf upfront
    out = np.nan_to_num(landmarks.copy(), nan=0.0, posinf=0.0, neginf=0.0)
    coords = out[:, :2]  # (543, 2)

    # Valid mask for non-zero coordinates
    valid_mask = (np.abs(coords[:, 0]) > eps) | (np.abs(coords[:, 1]) > eps)
    if not np.any(valid_mask):
        # All points are zero / undetected; return as-is
        return out

    # 1. Center Translation with Fallbacks (mid_hip -> nose -> bbox_center -> origin)
    center = np.zeros(2, dtype=np.float32)
    left_hip = coords[23] if valid_mask[23] else None
    right_hip = coords[24] if valid_mask[24] else None

    if center_anchor == "mid_hip" and left_hip is not None and right_hip is not None:
        center = (left_hip + right_hip) / 2.0
    elif (center_anchor in ("mid_hip", "nose")) and valid_mask[0]:
        # Fallback to nose (landmark 0)
        center = coords[0].copy()
    elif center_anchor != "origin":
        # Fallback to mean center of valid landmarks
        valid_coords = coords[valid_mask]
        if len(valid_coords) > 0:
            center = np.mean(valid_coords, axis=0)

    # Shift valid coordinates
    coords[valid_mask] -= center

    # 2. Scale Normalization with Fallbacks (shoulder_dist -> bbox diagonal -> unit scale)
    scale = 1.0
    left_shoulder = coords[11] if valid_mask[11] else None
    right_shoulder = coords[12] if valid_mask[12] else None

    if scale_anchor == "shoulder_dist" and left_shoulder is not None and right_shoulder is not None:
        dist = float(np.linalg.norm(left_shoulder - right_shoulder))
        if dist > eps:
            scale = dist
        else:
            # Fallback to bbox
            valid_shifted = coords[valid_mask]
            min_c = np.min(valid_shifted, axis=0)
            max_c = np.max(valid_shifted, axis=0)
            bbox_diag = float(np.linalg.norm(max_c - min_c))
            scale = bbox_diag if bbox_diag > eps else 1.0
    elif scale_anchor == "bbox":
        valid_shifted = coords[valid_mask]
        min_c = np.min(valid_shifted, axis=0)
        max_c = np.max(valid_shifted, axis=0)
        bbox_diag = float(np.linalg.norm(max_c - min_c))
        scale = bbox_diag if bbox_diag > eps else 1.0

    coords[valid_mask] /= (scale + eps)
    out[:, :2] = coords

    # Ensure no NaN / Inf in final output
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def normalize_landmark_sequence(
    sequence: np.ndarray,
    center_anchor: str = "mid_hip",
    scale_anchor: str = "shoulder_dist",
) -> np.ndarray:
    """
    Normalize an entire sequence of frames of shape (T, 543, 3).
    """
    if sequence.ndim != 3 or sequence.shape[1] != 543:
        raise ValueError(f"Expected sequence shape (T, 543, 3), got {sequence.shape}")

    T = sequence.shape[0]
    normalized = np.zeros_like(sequence)
    for t in range(T):
        normalized[t] = normalize_landmarks_frame(
            sequence[t],
            center_anchor=center_anchor,
            scale_anchor=scale_anchor,
        )
    return normalized

