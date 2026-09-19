"""
SIGNOVA Standalone Live Camera Runner.

Runs real-time camera capture using OpenCV and SignovaLiveRuntime directly
in a local Python terminal/window.
"""

from pathlib import Path
import sys
import time
from typing import Optional

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from signova.live.camera import OpenCVCameraCapture
from signova.live.runtime import SignovaLiveRuntime
from signova.live.status import CameraStatus


def run_live_camera(camera_index: int = 0, max_frames: Optional[int] = None):
    print("=" * 60)
    print(" SIGNOVA STANDALONE LIVE CAMERA RUNNER")
    print("=" * 60)

    if not HAS_CV2:
        print("[!] OpenCV (cv2) is not installed. Live camera window unavailable.")
        return

    camera = OpenCVCameraCapture(camera_index=camera_index)
    if not camera.open():
        print(f"[!] Unable to open camera device at index {camera_index}.")
        print("[!] Ensure a webcam is connected and not in use by another application.")
        return

    runtime = SignovaLiveRuntime()
    runtime.set_camera_status(CameraStatus.CONNECTED)
    print(f"[+] Camera index {camera_index} opened successfully.")
    print("[+] Press 'q' in the camera window or Ctrl+C in terminal to exit.")
    print("[+] Press 'r' to trigger SIGNAL_RESET.\n")

    frame_count = 0
    try:
        while camera.is_opened:
            ret, frame, t_cap = camera.read_frame()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            snapshot = runtime.process_frame(frame, capture_timestamp_ms=t_cap * 1000.0)

            # Display telemetry on frame if GUI available
            display_frame = frame.copy()
            h, w, _ = display_frame.shape

            # Overlay status box
            status_text = (
                f"Track: {snapshot.tracking_status.value} | Act: {snapshot.activity_status.value} | "
                f"Buf: {snapshot.buffer_frames}/{snapshot.buffer_capacity} | "
                f"FPS: {snapshot.processed_fps:.1f} | Lat: {snapshot.latencies.total_pipeline_latency_ms:.1f}ms"
            )
            cv2.rectangle(display_frame, (10, 10), (w - 10, 50), (20, 20, 20), -1)
            cv2.putText(
                display_frame,
                status_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 128),
                1,
                cv2.LINE_AA,
            )

            # Model & Translation banner
            banner_text = f"Model: {snapshot.model_status.value} (STATE_B Diagnostic Mode)"
            cv2.putText(
                display_frame,
                banner_text,
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 180, 255),
                1,
                cv2.LINE_AA,
            )

            try:
                cv2.imshow("SIGNOVA Live Camera Feed (Press 'q' to exit, 'r' to reset)", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    print("\n[*] Manual SIGNAL_RESET triggered via 'r' key.")
                    runtime.signal_reset()
            except Exception:
                # Running headless without display
                if frame_count % 30 == 0:
                    print(
                        f"Frame {frame_count:04d} | "
                        f"Tracking: {snapshot.tracking_status.value:8s} | "
                        f"Activity: {snapshot.activity_status.value:12s} | "
                        f"Buffer: {snapshot.buffer_frames:2d}/64 | "
                        f"FPS: {snapshot.processed_fps:4.1f} | "
                        f"Lat: {snapshot.latencies.total_pipeline_latency_ms:5.1f}ms"
                    )

            if max_frames and frame_count >= max_frames:
                break

    except KeyboardInterrupt:
        print("\n[+] Live camera stopped by user.")
    finally:
        camera.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print(f"[+] Total frames processed: {frame_count}")
        print("[+] Camera hardware released cleanly.\n")


if __name__ == "__main__":
    run_live_camera()
