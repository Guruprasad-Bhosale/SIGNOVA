"""
Live Metrics and Latency Tracker for SIGNOVA.
"""

from collections import deque
import time
from typing import Dict, List, Optional
from signova.live.status import StageLatencies


class LiveMetricsTracker:
    """
    Monotonic timer performance and latency instrumentation.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._input_timestamps: deque[float] = deque(maxlen=window_size)
        self._processed_timestamps: deque[float] = deque(maxlen=window_size)
        self._stage_latencies = StageLatencies()

    def record_input_frame(self):
        """Records an incoming frame arrival timestamp."""
        self._input_timestamps.append(time.monotonic())

    def record_processed_frame(self):
        """Records a successfully processed frame timestamp."""
        self._processed_timestamps.append(time.monotonic())

    def set_latencies(
        self,
        capture_ms: float = 0.0,
        frame_decode_ms: float = 0.0,
        mediapipe_ms: float = 0.0,
        normalization_ms: float = 0.0,
        model_inference_ms: float = 0.0,
        ctc_decoding_ms: float = 0.0,
        translation_ms: float = 0.0,
        total_pipeline_ms: float = 0.0,
    ):
        self._stage_latencies.capture_latency_ms = capture_ms
        self._stage_latencies.frame_decode_latency_ms = frame_decode_ms
        self._stage_latencies.mediapipe_latency_ms = mediapipe_ms
        self._stage_latencies.normalization_latency_ms = normalization_ms
        self._stage_latencies.model_inference_latency_ms = model_inference_ms
        self._stage_latencies.ctc_decoding_latency_ms = ctc_decoding_ms
        self._stage_latencies.translation_latency_ms = translation_ms
        self._stage_latencies.total_pipeline_latency_ms = total_pipeline_ms

    @property
    def stage_latencies(self) -> StageLatencies:
        return self._stage_latencies

    @property
    def input_fps(self) -> float:
        """Computes true arrival frame rate over rolling window."""
        if len(self._input_timestamps) < 2:
            return 0.0
        delta = self._input_timestamps[-1] - self._input_timestamps[0]
        if delta <= 0:
            return 0.0
        return (len(self._input_timestamps) - 1) / delta

    @property
    def processed_fps(self) -> float:
        """Computes true processed throughput frame rate over rolling window."""
        if len(self._processed_timestamps) < 2:
            return 0.0
        delta = self._processed_timestamps[-1] - self._processed_timestamps[0]
        if delta <= 0:
            return 0.0
        return (len(self._processed_timestamps) - 1) / delta

    def reset(self):
        """Resets performance history."""
        self._input_timestamps.clear()
        self._processed_timestamps.clear()
        self._stage_latencies = StageLatencies()
