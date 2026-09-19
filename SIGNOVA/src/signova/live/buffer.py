"""
Temporal Frame Buffer for SIGNOVA Live Streaming.
"""

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from signova.features.feature_groups import LandmarkGroup, slice_landmark_tensor


@dataclass
class BufferedFrame:
    frame_index: int
    timestamp_ms: float
    landmarks: np.ndarray        # Shape: (543, 3) [x, y, confidence/visibility]
    detection_mask: np.ndarray   # Shape: (4,) [pose, face, lh, rh]


class TemporalFrameBuffer:
    """
    FIFO rolling buffer preserving chronological sequence context for continuous ISL recognition.
    """

    def __init__(
        self,
        buffer_size: int = 64,
        stride: int = 16,
        min_frames_for_inference: int = 64,
        landmark_group: Union[str, LandmarkGroup] = LandmarkGroup.HANDS_POSE,
    ):
        self.buffer_size = buffer_size
        self.stride = stride
        self.min_frames_for_inference = min_frames_for_inference
        if isinstance(landmark_group, str):
            self.landmark_group = LandmarkGroup(landmark_group.lower())
        else:
            self.landmark_group = landmark_group
        self._buffer: deque[BufferedFrame] = deque(maxlen=buffer_size)
        self._total_pushed = 0
        self._last_inference_frame = -1

    def configure_from_model_spec(
        self,
        temporal_window: int,
        temporal_stride: int,
        landmark_group: Union[str, LandmarkGroup],
    ):
        """Reconfigures buffer parameters to match trained model input specification."""
        self.buffer_size = temporal_window
        self.stride = temporal_stride
        self.min_frames_for_inference = temporal_window
        if isinstance(landmark_group, str):
            self.landmark_group = LandmarkGroup(landmark_group.lower())
        else:
            self.landmark_group = landmark_group
        # Rebuild deque with new maxlen
        current_frames = list(self._buffer)
        self._buffer = deque(current_frames[-temporal_window:], maxlen=temporal_window)

    def push(
        self,
        landmarks: np.ndarray,
        detection_mask: np.ndarray,
        frame_index: Optional[int] = None,
        timestamp_ms: Optional[float] = None,
    ):
        """
        Pushes a single validated frame into the FIFO buffer.
        """
        if frame_index is None:
            frame_index = self._total_pushed
        if timestamp_ms is None:
            timestamp_ms = time.monotonic() * 1000.0

        if np.isnan(landmarks).any() or np.isinf(landmarks).any():
            # Zero out corrupted coordinates to prevent exploding gradients/inf
            landmarks = np.nan_to_num(landmarks, nan=0.0, posinf=0.0, neginf=0.0)

        frame = BufferedFrame(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            landmarks=landmarks.astype(np.float32),
            detection_mask=detection_mask.astype(np.float32),
        )
        self._buffer.append(frame)
        self._total_pushed += 1

    def reset(self):
        """Clears all buffered frames on explicit SIGNAL_RESET."""
        self._buffer.clear()
        self._total_pushed = 0
        self._last_inference_frame = -1

    @property
    def current_frames_count(self) -> int:
        return len(self._buffer)

    def is_ready_for_inference(self) -> bool:
        """
        Determines whether the buffer has reached the required temporal window
        and respects the configured temporal stride.
        """
        count = len(self._buffer)
        if count < self.min_frames_for_inference:
            return False

        if self._last_inference_frame < 0:
            return True

        frames_since_last = self._total_pushed - self._last_inference_frame
        return frames_since_last >= self.stride

    def mark_inference_executed(self):
        """Records the frame counter when inference was triggered."""
        self._last_inference_frame = self._total_pushed

    def get_feature_tensor(
        self,
        device: Union[str, torch.device] = "cpu",
        group_override: Optional[Union[str, LandmarkGroup]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Extracts current buffer frames as a batched tensor ready for the temporal model.

        Returns:
            features: Tensor of shape (1, T, num_landmarks, 3) sliced to requested feature group.
            padding_mask: Tensor of shape (1, T) indicating valid frame positions.
            valid_length: Integer frame count T.
        """
        frames_list = list(self._buffer)
        T = len(frames_list)
        if T == 0:
            empty_feat = torch.zeros((1, 0, 543, 3), dtype=torch.float32, device=device)
            empty_mask = torch.zeros((1, 0), dtype=torch.bool, device=device)
            return empty_feat, empty_mask, 0

        # Stack landmarks -> (T, 543, 3)
        raw_landmarks = np.stack([f.landmarks for f in frames_list], axis=0)
        landmarks_tensor = torch.from_numpy(raw_landmarks).float().unsqueeze(0)  # (1, T, 543, 3)

        group = group_override or self.landmark_group
        sliced = slice_landmark_tensor(landmarks_tensor, group_name=group)
        sliced = sliced.to(device)

        padding_mask = torch.ones((1, T), dtype=torch.bool, device=device)
        return sliced, padding_mask, T

    def get_detection_summary(self) -> Dict[str, float]:
        """Computes recent detection presence rates (pose, face, lh, rh)."""
        if not self._buffer:
            return {"pose": 0.0, "face": 0.0, "left_hand": 0.0, "right_hand": 0.0}

        masks = np.stack([f.detection_mask for f in self._buffer], axis=0)  # (T, 4)
        mean_rates = masks.mean(axis=0)
        return {
            "pose": float(mean_rates[0]),
            "face": float(mean_rates[1]),
            "left_hand": float(mean_rates[2]),
            "right_hand": float(mean_rates[3]),
        }
