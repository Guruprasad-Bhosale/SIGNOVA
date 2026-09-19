"""
Unit tests for Phase 20 ActivityStatus and TrackingStatus separation.
"""

import numpy as np
import pytest

from signova.live.runtime import SignovaLiveRuntime
from signova.live.status import ActivityStatus, TrackingStatus


def test_tracking_status_evaluation():
    runtime = SignovaLiveRuntime()

    # 1. Pose + Left Hand -> GOOD
    mask_good = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    assert runtime._evaluate_tracking_status(mask_good) == TrackingStatus.GOOD

    # 2. Pose only -> DEGRADED
    mask_degraded = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert runtime._evaluate_tracking_status(mask_degraded) == TrackingStatus.DEGRADED

    # 3. No detections -> LOST
    mask_lost = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert runtime._evaluate_tracking_status(mask_lost) == TrackingStatus.LOST


def test_activity_status_evaluation():
    runtime = SignovaLiveRuntime(activity_threshold=0.01)

    lm1 = np.zeros((543, 3), dtype=np.float32)
    mask_active = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # First frame establishes baseline -> ACTIVE
    act1 = runtime._compute_activity_level(lm1, mask_active)
    assert act1 == ActivityStatus.ACTIVE

    # Stationary frame (identical coordinates) -> LOW_ACTIVITY
    lm2 = lm1.copy()
    act2 = runtime._compute_activity_level(lm2, mask_active)
    assert act2 == ActivityStatus.LOW_ACTIVITY

    # Moving frame (significant coordinate displacement on hands) -> ACTIVE
    lm3 = lm1.copy()
    lm3[501:521, :2] += 0.08  # Left hand shift
    act3 = runtime._compute_activity_level(lm3, mask_active)
    assert act3 == ActivityStatus.ACTIVE

    # Untracked frame (zero detections) -> UNKNOWN
    mask_zero = np.zeros(4, dtype=np.float32)
    act4 = runtime._compute_activity_level(lm3, mask_zero)
    assert act4 == ActivityStatus.UNKNOWN
