"""
Unit tests for landmark normalization and temporal sampling.
"""

import numpy as np
import pytest
from signova.preprocessing.normalization import normalize_landmark_sequence, normalize_landmarks_frame
from signova.preprocessing.sampling import pad_or_crop_sequence


def test_normalization_shape_and_range():
    raw_landmarks = np.random.uniform(0.1, 0.9, (543, 3)).astype(np.float32)
    # Set hips and shoulders to test mid-hip and shoulder distance
    raw_landmarks[23] = [0.45, 0.6, 0.9]  # Left hip
    raw_landmarks[24] = [0.55, 0.6, 0.9]  # Right hip
    raw_landmarks[11] = [0.4, 0.3, 0.9]   # Left shoulder
    raw_landmarks[12] = [0.6, 0.3, 0.9]   # Right shoulder

    norm = normalize_landmarks_frame(raw_landmarks)
    assert norm.shape == (543, 3)
    # Confidence column remains unchanged
    assert np.allclose(norm[:, 2], raw_landmarks[:, 2])


def test_pad_or_crop_sequence():
    # Test padding
    short_seq = np.random.randn(30, 543, 3)
    padded = pad_or_crop_sequence(short_seq, target_len=60)
    assert padded.shape == (60, 543, 3)
    assert np.allclose(padded[30:], 0.0)

    # Test uniform cropping
    long_seq = np.random.randn(100, 543, 3)
    cropped = pad_or_crop_sequence(long_seq, target_len=50, crop_mode="uniform")
    assert cropped.shape == (50, 543, 3)
