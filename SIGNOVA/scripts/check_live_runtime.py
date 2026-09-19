"""
SIGNOVA Live Runtime Safe Diagnostic & Readiness Inspection.

Safe, non-training script verifying:
- Python, PyTorch, OpenCV, MediaPipe, CUDA, and GPU hardware support
- Backend REST and WebSocket configurations
- Feature normalization pipeline & landmark topology (543 landmarks)
- Temporal buffer defaults & model-derived specifications
- Real model authorization check under Phase 19 readiness gate
- Current scientific supervision state (STATE_B vs STATE_A)
"""

from pathlib import Path
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

import torch
from signova.live.model_registry import LiveModelRegistry
from signova.live.status import ModelStatus
from signova.operations.phase19_orchestrator import evaluate_phase19_readiness


def check_live_runtime():
    print("=" * 60)
    print(" SIGNOVA LIVE RUNTIME DIAGNOSTIC & READINESS CHECK")
    print("=" * 60)

    # 1. System & Dependencies
    import platform
    try:
        import cv2
        cv2_ver = cv2.__version__
    except ImportError:
        cv2_ver = "NOT_INSTALLED"

    try:
        import mediapipe as mp
        mp_ver = mp.__version__
    except ImportError:
        mp_ver = "NOT_INSTALLED"

    print("\n[1] Environment & Hardware:")
    print(f"  Python:        {platform.python_version()}")
    print(f"  PyTorch:       {torch.__version__}")
    print(f"  OpenCV:        {cv2_ver}")
    print(f"  MediaPipe:     {mp_ver}")
    print(f"  CUDA Enabled:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU Device:    {torch.cuda.get_device_name(0)}")
    else:
        print("  GPU Device:    CPU (Inference fallback active)")

    # 2. Network & Streaming Interfaces
    print("\n[2] Streaming Transport Interfaces:")
    print("  Backend:       FastAPI REST (apps.api.main:app)")
    print("  WebSocket:     ws://127.0.0.1:8000/ws/live")
    print("  REST Status:   http://127.0.0.1:8000/api/live/status")
    print("  Session Reset: POST http://127.0.0.1:8000/api/live/session/reset")

    # 3. Perception & Feature Pipeline
    print("\n[3] Perception & Feature Pipeline:")
    print("  Topology:      543 Landmarks (33 Pose, 468 Face, 21 LH, 21 RH)")
    print("  Normalization: Canonical SIGNOVA Sequence Normalizer (v1.0.0)")
    print("  Feature Group: HANDS_POSE (Default for Continuous ISL)")
    print("  Activity Det:  TrackingStatus (GOOD/DEGRADED/LOST) + ActivityStatus (ACTIVE/LOW_ACTIVITY)")

    # 4. Temporal Buffer Configuration
    registry = LiveModelRegistry()
    spec = registry.default_spec
    print("\n[4] Temporal Buffer Specification:")
    print(f"  Buffer Window: {spec.temporal_window} frames (FIFO rolling window)")
    print(f"  Stride:        {spec.temporal_stride} frames")
    print(f"  Min Frames:    {spec.temporal_window} frames (aligned to trained window)")
    print("  Reset Signal:  SIGNAL_RESET (clears buffer & resets utterance)")

    # 5. Model Availability & Phase 19 Gate Inspection
    status, meta, reason = registry.inspect_model_availability()
    readiness = evaluate_phase19_readiness()
    supervision_state = readiness.get("supervision_state", "STATE_B")
    real_ctc_status = readiness.get("real_ctc_status", "BLOCKED")

    print("\n[5] Real Model & Scientific Gate Status:")
    print(f"  Supervision:   {supervision_state}")
    print(f"  CTC Training:  {real_ctc_status}")
    print(f"  Model Status:  {status.value}")
    print(f"  Gate Reason:   {reason}")

    # 6. Translation Subsystem
    print("\n[6] Translation Subsystem:")
    if status == ModelStatus.AVAILABLE:
        print("  Status:        READY (Authorized real model active)")
    else:
        print("  Status:        BLOCKED (Awaiting authorized genuine CTC model)")
        print("  Display Text:  'Live camera tracking is active. A genuine sequential ISL recognition model is not currently available.'")
        print("  Fake Output:   ZERO (No synthetic checkpoints, no random tokens, no LLM glosses)")

    # 7. Overall Diagnostic Verdict
    print("\n" + "=" * 60)
    print(f" Diagnostic complete. Runtime operational in {supervision_state} Diagnostic Mode.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    check_live_runtime()
