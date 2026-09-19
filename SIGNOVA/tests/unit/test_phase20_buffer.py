"""
Unit tests for Phase 20 Temporal Frame Buffer and SIGNAL_RESET.
"""

import numpy as np
import pytest
import torch

from signova.features.feature_groups import LandmarkGroup
from signova.live.buffer import TemporalFrameBuffer


def test_buffer_push_and_capacity():
    buf = TemporalFrameBuffer(buffer_size=64, stride=16, min_frames_for_inference=64)
    assert buf.current_frames_count == 0

    lm = np.zeros((543, 3), dtype=np.float32)
    mask = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # Push 30 frames
    for i in range(30):
        buf.push(lm, mask, frame_index=i)
    assert buf.current_frames_count == 30
    assert buf.is_ready_for_inference() is False

    # Push up to 70 frames (exceeding buffer capacity 64)
    for i in range(30, 70):
        buf.push(lm, mask, frame_index=i)
    assert buf.current_frames_count == 64
    assert buf.is_ready_for_inference() is True


def test_buffer_feature_tensor_slicing():
    buf = TemporalFrameBuffer(buffer_size=64, stride=16, landmark_group=LandmarkGroup.HANDS_POSE)
    lm = np.zeros((543, 3), dtype=np.float32)
    mask = np.ones(4, dtype=np.float32)

    for i in range(20):
        buf.push(lm, mask, frame_index=i)

    feat_tensor, pad_mask, valid_T = buf.get_feature_tensor(device="cpu")
    assert valid_T == 20
    assert feat_tensor.dim() == 4  # (1, T, num_joints, 3)
    assert feat_tensor.shape[0] == 1
    assert feat_tensor.shape[1] == 20
    # HANDS_POSE has 75 joints (33 pose + 21 LH + 21 RH)
    assert feat_tensor.shape[2] == 75
    assert pad_mask.shape == (1, 20)


def test_buffer_signal_reset():
    buf = TemporalFrameBuffer(buffer_size=64, stride=16)
    lm = np.zeros((543, 3), dtype=np.float32)
    mask = np.ones(4, dtype=np.float32)

    for i in range(40):
        buf.push(lm, mask, frame_index=i)
    assert buf.current_frames_count == 40

    buf.reset()
    assert buf.current_frames_count == 0
    assert buf.is_ready_for_inference() is False


def test_buffer_configure_from_model_spec():
    buf = TemporalFrameBuffer(buffer_size=64, stride=16)
    buf.configure_from_model_spec(temporal_window=32, temporal_stride=8, landmark_group="FULL")
    assert buf.buffer_size == 32
    assert buf.stride == 8
    assert buf.min_frames_for_inference == 32
    assert buf.landmark_group == LandmarkGroup.FULL
