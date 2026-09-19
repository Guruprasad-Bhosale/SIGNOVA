"""
Landmark Quality and Extraction Health Analytics for SIGNOVA.

Evaluates component detection rates (pose, face, hands), tracking continuity,
confidence distributions, and detects anomalous frames or missing signer features.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class QualityMetrics:
    total_frames: int
    pose_detection_rate: float
    face_detection_rate: float
    left_hand_detection_rate: float
    right_hand_detection_rate: float
    any_hand_detection_rate: float
    both_hands_detection_rate: float
    valid_frame_ratio: float
    mean_confidence: float
    zero_frame_count: int
    jump_anomaly_count: int
    is_usable: bool
    rejection_reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LandmarkQualityEvaluator:
    """
    Evaluates the quality of extracted landmark sequences.
    """

    def __init__(
        self,
        min_pose_rate: float = 0.5,
        min_any_hand_rate: float = 0.3,
        min_valid_ratio: float = 0.4,
        max_jump_threshold: float = 4.0,
    ):
        self.min_pose_rate = min_pose_rate
        self.min_any_hand_rate = min_any_hand_rate
        self.min_valid_ratio = min_valid_ratio
        self.max_jump_threshold = max_jump_threshold

    def evaluate(
        self,
        landmarks: np.ndarray,
        detection_mask: Optional[np.ndarray] = None,
    ) -> QualityMetrics:
        """
        Evaluate a sequence of shape (T, 543, 3) and optional detection mask (T, 4).

        Args:
            landmarks: Array of shape (T, 543, 3).
            detection_mask: Array of shape (T, 4) [pose, face, lh, rh] where 1.0 indicates presence.

        Returns:
            QualityMetrics instance.
        """
        T = landmarks.shape[0]
        rejection_reasons = []

        if T == 0:
            return QualityMetrics(
                total_frames=0,
                pose_detection_rate=0.0,
                face_detection_rate=0.0,
                left_hand_detection_rate=0.0,
                right_hand_detection_rate=0.0,
                any_hand_detection_rate=0.0,
                both_hands_detection_rate=0.0,
                valid_frame_ratio=0.0,
                mean_confidence=0.0,
                zero_frame_count=0,
                jump_anomaly_count=0,
                is_usable=False,
                rejection_reasons=["Empty landmark sequence (0 frames)"],
            )

        # Infer mask if not provided
        if detection_mask is None:
            detection_mask = np.zeros((T, 4), dtype=np.float32)
            for t in range(T):
                pose_active = np.any(landmarks[t, 0:33, :2] != 0)
                face_active = np.any(landmarks[t, 33:501, :2] != 0)
                lh_active = np.any(landmarks[t, 501:522, :2] != 0)
                rh_active = np.any(landmarks[t, 522:543, :2] != 0)
                detection_mask[t] = [
                    1.0 if pose_active else 0.0,
                    1.0 if face_active else 0.0,
                    1.0 if lh_active else 0.0,
                    1.0 if rh_active else 0.0,
                ]

        pose_rate = float(np.mean(detection_mask[:, 0]))
        face_rate = float(np.mean(detection_mask[:, 1]))
        lh_rate = float(np.mean(detection_mask[:, 2]))
        rh_rate = float(np.mean(detection_mask[:, 3]))
        any_hand = (detection_mask[:, 2] > 0) | (detection_mask[:, 3] > 0)
        any_hand_rate = float(np.mean(any_hand))
        both_hands = (detection_mask[:, 2] > 0) & (detection_mask[:, 3] > 0)
        both_hands_rate = float(np.mean(both_hands))

        # Valid frames have pose + at least one hand
        valid_frames = (detection_mask[:, 0] > 0) & any_hand
        valid_ratio = float(np.mean(valid_frames))

        # Check zero-frames (no landmarks detected at all)
        zero_frames = np.sum(np.all(landmarks[:, :, :2] == 0, axis=(1, 2)))

        # Confidence analysis (column index 2)
        conf_mask = landmarks[:, :, 2] > 0
        mean_conf = float(np.mean(landmarks[:, :, 2][conf_mask])) if np.any(conf_mask) else 0.0

        # Jump anomaly detection (inter-frame delta for key pose joints: nose, wrists)
        jump_anomalies = 0
        if T > 1:
            # Check wrist and nose delta
            key_indices = [0, 15, 16]  # nose, left wrist, right wrist
            diffs = np.diff(landmarks[:, key_indices, :2], axis=0)  # (T-1, 3, 2)
            dists = np.linalg.norm(diffs, axis=2)  # (T-1, 3)
            jump_anomalies = int(np.sum(dists > self.max_jump_threshold))

        # Usability gates
        if pose_rate < self.min_pose_rate:
            rejection_reasons.append(
                f"Pose detection rate {pose_rate:.2f} below minimum {self.min_pose_rate:.2f}"
            )
        if any_hand_rate < self.min_any_hand_rate:
            rejection_reasons.append(
                f"Hand detection rate {any_hand_rate:.2f} below minimum {self.min_any_hand_rate:.2f}"
            )
        if valid_ratio < self.min_valid_ratio:
            rejection_reasons.append(
                f"Valid frame ratio {valid_ratio:.2f} below minimum {self.min_valid_ratio:.2f}"
            )

        is_usable = len(rejection_reasons) == 0

        return QualityMetrics(
            total_frames=T,
            pose_detection_rate=round(pose_rate, 4),
            face_detection_rate=round(face_rate, 4),
            left_hand_detection_rate=round(lh_rate, 4),
            right_hand_detection_rate=round(rh_rate, 4),
            any_hand_detection_rate=round(any_hand_rate, 4),
            both_hands_detection_rate=round(both_hands_rate, 4),
            valid_frame_ratio=round(valid_ratio, 4),
            mean_confidence=round(mean_conf, 4),
            zero_frame_count=int(zero_frames),
            jump_anomaly_count=jump_anomalies,
            is_usable=is_usable,
            rejection_reasons=rejection_reasons,
        )
