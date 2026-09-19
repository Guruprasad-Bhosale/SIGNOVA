"""
Robust Video Decoder for SIGNOVA.

Provides OpenCV VideoCapture context managers, corrupt video detection,
frame decimation, and non-blocking streaming without loading entire videos into RAM.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class VideoMetadata:
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float
    codec: str
    is_valid: bool
    error: Optional[str] = None


class RobustVideoDecoder:
    """
    Robust frame generator and metadata extractor for local video files.
    """

    def __init__(self, video_path: Union[str, Path]):
        self.video_path = Path(video_path).resolve()
        self._metadata: Optional[VideoMetadata] = None

    def read_metadata(self) -> VideoMetadata:
        """
        Inspect video container metadata safely.
        """
        if self._metadata is not None:
            return self._metadata

        if not self.video_path.is_file() or self.video_path.stat().st_size == 0:
            self._metadata = VideoMetadata(
                width=0,
                height=0,
                fps=0.0,
                total_frames=0,
                duration_sec=0.0,
                codec="UNKNOWN",
                is_valid=False,
                error=f"Video file not found or empty: {self.video_path}",
            )
            return self._metadata

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            self._metadata = VideoMetadata(
                width=0,
                height=0,
                fps=0.0,
                total_frames=0,
                duration_sec=0.0,
                codec="UNREADABLE",
                is_valid=False,
                error="OpenCV failed to open video container",
            )
            return self._metadata

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fps = fps if fps > 0 else 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0

        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()

        # Read first test frame to verify readability
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            self._metadata = VideoMetadata(
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration_sec=duration_sec,
                codec=codec,
                is_valid=False,
                error="Failed to decode initial frame",
            )
        else:
            self._metadata = VideoMetadata(
                width=width,
                height=height,
                fps=fps,
                total_frames=total_frames,
                duration_sec=duration_sec,
                codec=codec,
                is_valid=True,
                error=None,
            )

        return self._metadata

    def stream_frames(
        self,
        target_fps: Optional[float] = None,
        max_frames: Optional[int] = None,
    ) -> Iterator[Tuple[int, float, np.ndarray]]:
        """
        Stream frames from the video container one at a time.
        
        Yields:
            (frame_index: int, timestamp_ms: float, bgr_frame: np.ndarray)
        """
        meta = self.read_metadata()
        if not meta.is_valid:
            return

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            return

        source_fps = meta.fps
        stride = 1
        if target_fps and target_fps > 0 and target_fps < source_fps:
            stride = max(1, int(round(source_fps / target_fps)))

        frame_idx = 0
        yielded_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % stride == 0:
                timestamp_ms = (frame_idx / source_fps) * 1000.0
                yield frame_idx, timestamp_ms, frame
                yielded_count += 1
                if max_frames and yielded_count >= max_frames:
                    break

            frame_idx += 1

        cap.release()


def read_video_metadata(video_path: Union[str, Path]) -> VideoMetadata:
    """Convenience helper to read metadata using RobustVideoDecoder."""
    decoder = RobustVideoDecoder(video_path)
    return decoder.read_metadata()


def get_video_metadata_mock(video_path: Union[str, Path] = "mock_video.mp4") -> VideoMetadata:
    """Generate mock metadata for testing purposes."""
    return VideoMetadata(
        width=1280,
        height=720,
        fps=30.0,
        total_frames=150,
        duration_sec=5.0,
        codec="avc1",
        is_valid=True,
        error=None,
    )

