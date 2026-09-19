"""
Status and Telemetry Models for SIGNOVA Live Camera Runtime.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CameraStatus(str, Enum):
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    ERROR = "ERROR"
    UNAVAILABLE = "UNAVAILABLE"


class TrackingStatus(str, Enum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    LOST = "LOST"


class ActivityStatus(str, Enum):
    ACTIVE = "ACTIVE"
    LOW_ACTIVITY = "LOW_ACTIVITY"
    UNKNOWN = "UNKNOWN"


class ModelStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    BLOCKED = "BLOCKED"


class TranslationStatus(str, Enum):
    READY = "READY"
    BLOCKED = "BLOCKED"


@dataclass
class StageLatencies:
    capture_latency_ms: float = 0.0
    frame_decode_latency_ms: float = 0.0
    mediapipe_latency_ms: float = 0.0
    normalization_latency_ms: float = 0.0
    model_inference_latency_ms: float = 0.0
    ctc_decoding_latency_ms: float = 0.0
    translation_latency_ms: float = 0.0
    total_pipeline_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "capture_latency_ms": round(self.capture_latency_ms, 2),
            "frame_decode_latency_ms": round(self.frame_decode_latency_ms, 2),
            "mediapipe_latency_ms": round(self.mediapipe_latency_ms, 2),
            "normalization_latency_ms": round(self.normalization_latency_ms, 2),
            "model_inference_latency_ms": round(self.model_inference_latency_ms, 2),
            "ctc_decoding_latency_ms": round(self.ctc_decoding_latency_ms, 2),
            "translation_latency_ms": round(self.translation_latency_ms, 2),
            "total_pipeline_latency_ms": round(self.total_pipeline_latency_ms, 2),
        }


@dataclass
class RuntimeStateSnapshot:
    """Complete real-time state emitted by the SIGNOVA live runtime."""
    session_id: str = ""
    frame_index: int = 0
    camera_status: CameraStatus = CameraStatus.DISCONNECTED
    tracking_status: TrackingStatus = TrackingStatus.LOST
    activity_status: ActivityStatus = ActivityStatus.UNKNOWN
    model_status: ModelStatus = ModelStatus.UNAVAILABLE
    translation_status: TranslationStatus = TranslationStatus.BLOCKED
    
    current_gloss: str = "—"
    committed_glosses: List[str] = field(default_factory=list)
    current_translation: str = "—"
    
    buffer_frames: int = 0
    buffer_capacity: int = 64
    min_frames_for_inference: int = 64
    
    input_fps: float = 0.0
    processed_fps: float = 0.0
    
    latencies: StageLatencies = field(default_factory=StageLatencies)
    
    scientific_state: str = "STATE_B"
    supervision_blocker: str = "no_genuine_human_annotations_present"
    status_message: str = (
        "Live camera tracking is active. A genuine sequential ISL recognition model "
        "is not currently available."
    )
    device: str = "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "frame_index": self.frame_index,
            "camera_status": self.camera_status.value,
            "tracking_status": self.tracking_status.value,
            "activity_status": self.activity_status.value,
            "model_status": self.model_status.value,
            "translation_status": self.translation_status.value,
            "current_gloss": self.current_gloss,
            "committed_glosses": self.committed_glosses,
            "current_translation": self.current_translation,
            "buffer_frames": self.buffer_frames,
            "buffer_capacity": self.buffer_capacity,
            "min_frames_for_inference": self.min_frames_for_inference,
            "input_fps": round(self.input_fps, 1),
            "processed_fps": round(self.processed_fps, 1),
            "latencies": self.latencies.to_dict(),
            "scientific_state": self.scientific_state,
            "supervision_blocker": self.supervision_blocker,
            "status_message": self.status_message,
            "device": self.device,
        }
