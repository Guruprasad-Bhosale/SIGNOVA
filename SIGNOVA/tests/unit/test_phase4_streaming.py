"""
Unit Tests for Phase 4 Streaming and Windowed Offline Inference.
"""

import numpy as np
import pytest
import torch

from signova.inference.streaming import ExperimentalRollingBufferStream, WindowedOfflineInference
from signova.models.continuous_encoder import ContinuousTemporalEncoder


def test_windowed_offline_inference_sequence_processing():
    model = ContinuousTemporalEncoder(backbone="gru", num_landmarks=75)
    pipeline = WindowedOfflineInference(model=model, window_size=32, stride=16, landmark_group="hands_pose")

    T = 80
    landmarks = np.random.randn(T, 543, 3).astype(np.float32)
    result = pipeline.process_sequence(landmarks, fps=30.0)

    assert "merged_embeddings" in result
    assert result["merged_embeddings"].shape == (T, 512)
    assert result["total_frames"] == T
    assert result["window_count"] > 0
    assert result["mode"] == "WINDOWED_OFFLINE_INFERENCE"


def test_experimental_rolling_buffer_stream():
    model = ContinuousTemporalEncoder(backbone="gru", num_landmarks=75)
    stream = ExperimentalRollingBufferStream(model=model, buffer_size=32, step_interval=4)

    # Push 20 frames into buffer
    updates = []
    for t in range(20):
        frame = torch.randn((75, 3), dtype=torch.float32)
        up = stream.push_frame(frame)
        if up is not None:
            updates.append(up)

    assert len(updates) > 0
    assert "latest_embedding" in updates[0]
    assert len(updates[0]["latest_embedding"]) == 512

    # Test buffer reset
    stream.reset()
    assert len(stream.buffer) == 0
    assert stream.frame_counter == 0
