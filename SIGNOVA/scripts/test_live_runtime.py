"""
SIGNOVA Live Runtime Integration Smoke Test.

Validates the full live streaming pipeline in:
1. DIAGNOSTIC MODE (Standard under current STATE_B):
   Frame Capture/Decoding -> MediaPipe Extraction -> Normalization -> Activity Status ->
   Temporal Buffer (FIFO 64 frames) -> Phase 19 Gate Check -> Honest Diagnostic Telemetry ->
   Explicit SIGNAL_RESET -> Buffer Clean Reset.
2. REAL MODEL MODE (When authorized by Phase 19 gate in future STATE_A):
   Frame -> Buffer -> Genuine CTC Recognizer -> Bridge -> Translator -> Live English.
"""

from pathlib import Path
import sys
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.live.camera import OpenCVCameraCapture
from signova.live.runtime import SignovaLiveRuntime
from signova.live.status import ActivityStatus, CameraStatus, ModelStatus, TrackingStatus


def run_live_runtime_smoke_test():
    print("=" * 60)
    print(" SIGNOVA LIVE RUNTIME INTEGRATION SMOKE TEST")
    print("=" * 60)

    runtime = SignovaLiveRuntime()

    # Step 1: Camera Hardware Test
    print("\n[Step 1] Camera Hardware Device Check:")
    camera = OpenCVCameraCapture(camera_index=0)
    camera_available = camera.open()

    if camera_available:
        print("  Status: HARDWARE_CAMERA_AVAILABLE (Index 0 opened)")
        ret, frame, t_cap = camera.read_frame()
        camera.release()
        if ret and frame is not None:
            print(f"  Frame capture verified: Shape {frame.shape}, dtype {frame.dtype}")
        else:
            print("  Frame capture yielded empty buffer; using synthetic stream.")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        print("  Status: SKIPPED_NO_CAMERA (Physical camera absent; using synthetic frame stream)")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Step 2: Stream Simulated Frames into Unified Runtime
    print("\n[Step 2] Processing Frame Stream through Unified Runtime:")
    runtime.set_camera_status(CameraStatus.CONNECTED)

    test_frames = 10
    snapshots = []
    for i in range(test_frames):
        # Create subtle movement in synthetic frame to exercise motion pipeline
        f = frame.copy()
        f[100:150, 100 + i * 5: 150 + i * 5] = 200
        snap = runtime.process_frame(f)
        snapshots.append(snap)

    last_snap = snapshots[-1]
    print(f"  Pushed {test_frames} frames successfully.")
    print(f"  Buffer Frame Count:       {last_snap.buffer_frames} / {last_snap.buffer_capacity}")
    print(f"  Tracking Status:          {last_snap.tracking_status.value}")
    print(f"  Activity Status:          {last_snap.activity_status.value}")
    print(f"  Model Availability:       {last_snap.model_status.value}")
    print(f"  Translation Status:       {last_snap.translation_status.value}")
    print(f"  Processed FPS:            {last_snap.processed_fps:.1f}")
    print(f"  Total Stage Latency:      {last_snap.latencies.total_pipeline_latency_ms:.2f} ms")

    # Step 3: Model Mode Verification
    print("\n[Step 3] Operating Mode Verification:")
    if last_snap.model_status == ModelStatus.AVAILABLE:
        print("  Active Mode: REAL MODEL MODE (Authorized genuine CTC model loaded)")
        print(f"  Current Gloss:       {last_snap.current_gloss}")
        print(f"  Committed Glosses:   {last_snap.committed_glosses}")
        print(f"  English Translation: {last_snap.current_translation}")
    else:
        print("  Active Mode: DIAGNOSTIC MODE (Current STATE_B)")
        print(f"  Status Message:      \"{last_snap.status_message}\"")
        assert last_snap.current_gloss == "—", "Diagnostic mode must not output fake glosses."
        assert last_snap.current_translation == "—", "Diagnostic mode must not output fake translation."
        print("  Zero-Fabrication Invariant: PASSED (No fake glosses or translations produced)")

    # Step 4: Explicit SIGNAL_RESET Verification
    print("\n[Step 4] Explicit SIGNAL_RESET Verification:")
    reset_snap = runtime.signal_reset()
    print(f"  Buffer frames after reset: {reset_snap.buffer_frames}")
    assert reset_snap.buffer_frames == 0, "Buffer frames must return to 0 after SIGNAL_RESET."
    assert reset_snap.committed_glosses == [], "Committed glosses must be empty after SIGNAL_RESET."
    print("  SIGNAL_RESET Behavior: PASSED (Buffer cleanly cleared, utterance state reset)")

    print("\n" + "=" * 60)
    print(" LIVE RUNTIME INTEGRATION SMOKE TEST: PASSED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_live_runtime_smoke_test()
