"""
Phase 21 Live Integration and Confidence Abstention Tests.
"""

import numpy as np
from signova.live.model_registry import LiveModelRegistry
from signova.live.runtime import SignovaLiveRuntime
from signova.live.status import ActivityStatus, ModelStatus, TranslationStatus


def test_live_runtime_state_b_abstention():
    runtime = SignovaLiveRuntime()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    snap = runtime.process_frame(frame)
    assert snap.model_status == ModelStatus.UNAVAILABLE
    assert snap.translation_status == TranslationStatus.BLOCKED
    assert snap.current_gloss == "—"
    assert snap.current_translation == "—"


def test_live_registry_detects_unauthorized_state():
    registry = LiveModelRegistry()
    status, meta, reason = registry.inspect_model_availability()

    assert status == ModelStatus.UNAVAILABLE
    assert meta is None
    assert "Supervision gate STATE_B" in reason or "Real CTC model training is blocked" in reason
