"""
Unit tests for Phase 20 Camera Abstraction and Frame Validation.
"""

import numpy as np
import pytest

from signova.live.camera import OpenCVCameraCapture
from signova.live.frame import decode_frame, validate_frame


def test_frame_validation():
    # Valid RGB/BGR frame
    valid_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert validate_frame(valid_frame) is True

    # Invalid: empty frame
    empty_frame = np.array([])
    assert validate_frame(empty_frame) is False

    # Invalid: 2D frame (missing channel dim)
    frame_2d = np.zeros((480, 640), dtype=np.uint8)
    assert validate_frame(frame_2d) is False

    # Invalid: too small
    tiny_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert validate_frame(tiny_frame) is False

    # Invalid: contains NaN
    nan_frame = np.zeros((480, 640, 3), dtype=np.float32)
    nan_frame[0, 0, 0] = np.nan
    assert validate_frame(nan_frame) is False


def test_decode_frame_numpy():
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    decoded, t = decode_frame(raw)
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (480, 640, 3)
    assert t > 0


def test_camera_headless_graceful_fallback():
    # Attempting to open a non-existent high index camera should gracefully return False
    camera = OpenCVCameraCapture(camera_index=999)
    res = camera.open()
    assert res is False
    assert camera.is_opened is False
    # read_frame should return False without crashing
    ret, frame, t_cap = camera.read_frame()
    assert ret is False
    assert frame is None
    camera.release()
