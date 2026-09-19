"""
Unit tests for landmark representations and tensor transformations.
"""

import numpy as np
import pytest
from signova.preprocessing.landmarks import HolisticLandmarkFrame


def test_landmark_frame_flatten_and_reconstruct():
    pose = np.random.uniform(0, 1, (33, 3)).astype(np.float32)
    face = np.random.uniform(0, 1, (468, 3)).astype(np.float32)
    lh = np.random.uniform(0, 1, (21, 3)).astype(np.float32)
    rh = np.random.uniform(0, 1, (21, 3)).astype(np.float32)

    frame = HolisticLandmarkFrame(pose=pose, face=face, left_hand=lh, right_hand=rh, timestamp_ms=100.0)
    flat = frame.to_flat_array()

    assert flat.shape == (543, 3)
    assert np.allclose(flat[0:33], pose)
    assert np.allclose(flat[33:501], face)
    assert np.allclose(flat[501:522], lh)
    assert np.allclose(flat[522:543], rh)

    reconstructed = HolisticLandmarkFrame.from_flat_array(flat, timestamp_ms=100.0)
    assert np.allclose(reconstructed.pose, pose)
    assert np.allclose(reconstructed.right_hand, rh)
