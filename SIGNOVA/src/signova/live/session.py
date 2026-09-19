"""
Live Session Manager for SIGNOVA Live Streaming.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional
import uuid


@dataclass
class LiveSessionStats:
    session_id: str
    start_time_iso: str
    start_timestamp: float
    end_timestamp: Optional[float] = None
    total_frames_received: int = 0
    frames_processed: int = 0
    dropped_frames: int = 0
    tracking_failures: int = 0
    inferences_attempted: int = 0
    inferences_executed: int = 0
    translations_produced: int = 0
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        duration = (
            (self.end_timestamp or time.monotonic()) - self.start_timestamp
            if self.start_timestamp > 0
            else 0.0
        )
        return {
            "session_id": self.session_id,
            "start_time_iso": self.start_time_iso,
            "duration_sec": round(duration, 2),
            "total_frames_received": self.total_frames_received,
            "frames_processed": self.frames_processed,
            "dropped_frames": self.dropped_frames,
            "tracking_failures": self.tracking_failures,
            "inferences_attempted": self.inferences_attempted,
            "inferences_executed": self.inferences_executed,
            "translations_produced": self.translations_produced,
            "active": self.active,
        }


class LiveSessionManager:
    """
    Manages live session lifecycle without persisting raw camera frames or videos.
    """

    def __init__(self):
        self._current_session: Optional[LiveSessionStats] = None

    def start_session(self) -> LiveSessionStats:
        """Starts a new live inference session."""
        session_id = f"live_{uuid.uuid4().hex[:12]}"
        t_now = time.monotonic()
        iso_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        self._current_session = LiveSessionStats(
            session_id=session_id,
            start_time_iso=iso_str,
            start_timestamp=t_now,
            active=True,
        )
        return self._current_session

    def get_active_session(self) -> LiveSessionStats:
        """Gets active session or creates a default one if none exists."""
        if self._current_session is None or not self._current_session.active:
            return self.start_session()
        return self._current_session

    def end_session(self) -> Optional[LiveSessionStats]:
        """Ends the active session."""
        if self._current_session is not None and self._current_session.active:
            self._current_session.end_timestamp = time.monotonic()
            self._current_session.active = False
            return self._current_session
        return self._current_session

    def record_frame_received(self):
        sess = self.get_active_session()
        sess.total_frames_received += 1

    def record_frame_processed(self):
        sess = self.get_active_session()
        sess.frames_processed += 1

    def record_frame_dropped(self):
        sess = self.get_active_session()
        sess.dropped_frames += 1

    def record_tracking_failure(self):
        sess = self.get_active_session()
        sess.tracking_failures += 1

    def record_inference_executed(self):
        sess = self.get_active_session()
        sess.inferences_executed += 1

    def record_translation_produced(self):
        sess = self.get_active_session()
        sess.translations_produced += 1
