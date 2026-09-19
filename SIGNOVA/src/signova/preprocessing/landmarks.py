"""
Landmark data structures and representations for SIGNOVA.

Handles 543 Holistic keypoints (Pose 33, Face 468, Left Hand 21, Right Hand 21).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class HolisticLandmarkFrame:
    """
    Holds structured landmarks for a single video frame.
    All coordinates are normalized in range [0, 1] with confidence/visibility.
    """
    pose: Optional[np.ndarray] = None        # Shape: (33, 3) [x, y, visibility]
    face: Optional[np.ndarray] = None        # Shape: (468, 3) [x, y, visibility]
    left_hand: Optional[np.ndarray] = None   # Shape: (21, 3) [x, y, confidence]
    right_hand: Optional[np.ndarray] = None  # Shape: (21, 3) [x, y, confidence]
    timestamp_ms: Optional[float] = None

    NUM_POSE = 33
    NUM_FACE = 468
    NUM_HAND = 21
    TOTAL_LANDMARKS = 543

    def to_flat_array(self) -> np.ndarray:
        """
        Flatten all landmarks into a single (543, 3) numpy array.
        Missing landmarks are zero-filled.
        """
        arr = np.zeros((self.TOTAL_LANDMARKS, 3), dtype=np.float32)

        offset = 0
        if self.pose is not None and self.pose.shape == (self.NUM_POSE, 3):
            arr[offset : offset + self.NUM_POSE] = self.pose
        offset += self.NUM_POSE

        if self.face is not None and self.face.shape == (self.NUM_FACE, 3):
            arr[offset : offset + self.NUM_FACE] = self.face
        offset += self.NUM_FACE

        if self.left_hand is not None and self.left_hand.shape == (self.NUM_HAND, 3):
            arr[offset : offset + self.NUM_HAND] = self.left_hand
        offset += self.NUM_HAND

        if self.right_hand is not None and self.right_hand.shape == (self.NUM_HAND, 3):
            arr[offset : offset + self.NUM_HAND] = self.right_hand

        return arr

    @classmethod
    def from_flat_array(cls, arr: np.ndarray, timestamp_ms: Optional[float] = None) -> "HolisticLandmarkFrame":
        """Reconstruct structured frame from a (543, 3) numpy array."""
        if arr.shape != (cls.TOTAL_LANDMARKS, 3):
            raise ValueError(f"Expected shape ({cls.TOTAL_LANDMARKS}, 3), got {arr.shape}")

        pose = arr[0:33].copy()
        face = arr[33:501].copy()
        left_hand = arr[501:522].copy()
        right_hand = arr[522:543].copy()

        return cls(
            pose=pose,
            face=face,
            left_hand=left_hand,
            right_hand=right_hand,
            timestamp_ms=timestamp_ms,
        )
