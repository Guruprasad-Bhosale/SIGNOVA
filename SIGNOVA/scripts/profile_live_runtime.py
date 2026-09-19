"""
SIGNOVA Live Runtime Latency & FPS Performance Profiler.

Measures actual stage-by-stage latencies (Capture, MediaPipe, Normalization, Buffer, Total)
and processed FPS over realistic continuous streams without fabrication.
"""

from pathlib import Path
import sys
import time
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from signova.live.runtime import SignovaLiveRuntime
from signova.live.status import CameraStatus


def profile_live_runtime(num_frames: int = 64):
    print("=" * 60)
    print(" SIGNOVA LIVE RUNTIME PERFORMANCE PROFILER")
    print("=" * 60)
    print(f"Profiling {num_frames} frames through SignovaLiveRuntime...\n")

    runtime = SignovaLiveRuntime()
    runtime.set_camera_status(CameraStatus.CONNECTED)

    # Generate synthetic camera stream
    sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    stage_records = {
        "decode": [],
        "mediapipe": [],
        "normalization": [],
        "inference": [],
        "total": [],
    }

    t_prof_start = time.perf_counter()

    for i in range(num_frames):
        # Add dynamic variation
        f = sample_frame.copy()
        f[120:180, 120 + (i % 50): 180 + (i % 50)] = 220

        t0 = time.perf_counter()
        snap = runtime.process_frame(f)
        t_elapsed = (time.perf_counter() - t0) * 1000.0

        stage_records["decode"].append(snap.latencies.frame_decode_latency_ms)
        stage_records["mediapipe"].append(snap.latencies.mediapipe_latency_ms)
        stage_records["normalization"].append(snap.latencies.normalization_latency_ms)
        stage_records["inference"].append(snap.latencies.model_inference_latency_ms)
        stage_records["total"].append(t_elapsed)

    t_prof_total = time.perf_counter() - t_prof_start
    overall_fps = num_frames / t_prof_total if t_prof_total > 0 else 0.0

    print("--- Stage Latency Profile (Averages across frames) ---")
    print(f"  Frame Decode Latency:         {np.mean(stage_records['decode']):6.2f} ms (std: {np.std(stage_records['decode']):.2f} ms)")
    print(f"  MediaPipe Extraction Latency: {np.mean(stage_records['mediapipe']):6.2f} ms (std: {np.std(stage_records['mediapipe']):.2f} ms)")
    print(f"  Feature Normalization Latency:{np.mean(stage_records['normalization']):6.2f} ms (std: {np.std(stage_records['normalization']):.2f} ms)")
    print(f"  Model Inference Latency:      {np.mean(stage_records['inference']):6.2f} ms")
    print(f"  Total Pipeline End-to-End:    {np.mean(stage_records['total']):6.2f} ms (std: {np.std(stage_records['total']):.2f} ms)")
    print("------------------------------------------------------")
    print(f"  Total Test Duration:          {t_prof_total:6.2f} s")
    print(f"  Measured Processed FPS:       {overall_fps:6.1f} FPS")
    print(f"  Buffer Fill Status:           {snap.buffer_frames} / {snap.buffer_capacity} frames")
    print(f"  Device:                       {snap.device}")
    print(f"  Scientific State:             {snap.scientific_state} (Diagnostic Mode)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    profile_live_runtime(num_frames=64)
