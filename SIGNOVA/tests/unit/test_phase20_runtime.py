"""
Unit tests for Phase 20 Unified SignovaLiveRuntime.
"""

import numpy as np
import pytest

from signova.live.runtime import SignovaLiveRuntime
from signova.live.status import CameraStatus, ModelStatus, TranslationStatus


def test_runtime_frame_processing_and_snapshot():
    runtime = SignovaLiveRuntime()
    runtime.set_camera_status(CameraStatus.CONNECTED)

    # Process simulated video frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    snapshot = runtime.process_frame(frame)

    assert snapshot.camera_status == CameraStatus.CONNECTED
    assert snapshot.buffer_frames == 1
    assert snapshot.buffer_capacity == 64
    assert snapshot.model_status == ModelStatus.UNAVAILABLE
    assert snapshot.translation_status == TranslationStatus.BLOCKED
    assert snapshot.current_gloss == "—"
    assert snapshot.current_translation == "—"
    assert snapshot.scientific_state == "STATE_B"


def test_runtime_signal_reset():
    runtime = SignovaLiveRuntime()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for _ in range(10):
        runtime.process_frame(frame)
    assert runtime.buffer.current_frames_count == 10

    snap = runtime.signal_reset()
    assert snap.buffer_frames == 0
    assert snap.committed_glosses == []
    assert snap.current_gloss == "—"
    assert snap.current_translation == "—"


def test_runtime_session_metrics():
    runtime = SignovaLiveRuntime()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for _ in range(5):
        runtime.process_frame(frame)

    sess = runtime.session_manager.get_active_session()
    assert sess.total_frames_received == 5
    assert sess.frames_processed == 5
    assert sess.dropped_frames == 0
