"""
Scientific Safety and Zero-Fabrication Tests for Phase 20 Live Runtime.
"""

from pathlib import Path
import numpy as np
import pytest

from signova.live.errors import ModelUnauthorizedError
from signova.live.model_registry import LiveModelRegistry
from signova.live.runtime import SignovaLiveRuntime
from signova.live.status import ModelStatus, TranslationStatus


def test_zero_hallucination_under_state_b():
    runtime = SignovaLiveRuntime()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Process multiple frames
    for _ in range(80):
        snap = runtime.process_frame(frame)

    assert snap.model_status == ModelStatus.UNAVAILABLE
    assert snap.translation_status == TranslationStatus.BLOCKED
    assert snap.current_gloss == "—"
    assert snap.current_translation == "—"
    assert snap.committed_glosses == []
    assert "not currently available" in snap.status_message


def test_rejection_of_synthetic_fixtures_as_real_model():
    registry = LiveModelRegistry()
    # Ensure synthetic test fixtures cannot bypass Phase 19 gate
    status, meta, reason = registry.inspect_model_availability()
    assert status == ModelStatus.UNAVAILABLE
    assert meta is None

    with pytest.raises(ModelUnauthorizedError):
        registry.load_authorized_model()


def test_no_camera_frame_disk_persistence():
    # Verify no video/image files are written to disk during live streaming
    base_data = Path("data")
    existing_files = set(base_data.rglob("*.mp4")) | set(base_data.rglob("*.jpg")) | set(base_data.rglob("*.png"))

    runtime = SignovaLiveRuntime()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(20):
        runtime.process_frame(frame)

    post_files = set(base_data.rglob("*.mp4")) | set(base_data.rglob("*.jpg")) | set(base_data.rglob("*.png"))
    new_files = post_files - existing_files
    assert len(new_files) == 0, f"Live streaming must not write raw video/frame files to disk! Found: {new_files}"
