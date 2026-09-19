"""
OpenCV Camera Capture Interface for SIGNOVA Standalone Python Runtime.
"""

import time
from typing import Generator, Optional, Tuple
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from signova.live.errors import CameraError
from signova.live.status import CameraStatus


class OpenCVCameraCapture:
    """
    Manages local hardware camera capture with graceful fallback for headless environments.
    """

    def __init__(
        self,
        camera_index: int = 0,
        target_width: int = 640,
        target_height: int = 480,
        target_fps: int = 30,
    ):
        self.camera_index = camera_index
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps
        self._cap = None
        self._is_opened = False

    def open(self) -> bool:
        """Attempts to open the camera device."""
        if not HAS_CV2:
            self._is_opened = False
            return False

        try:
            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                self._is_opened = False
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self._is_opened = True
            return True
        except Exception:
            self._is_opened = False
            return False

    @property
    def is_opened(self) -> bool:
        return self._is_opened and (self._cap is not None and self._cap.isOpened())

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Captures a single BGR frame.

        Returns:
            (success: bool, frame: Optional[np.ndarray], capture_timestamp: float)
        """
        t_capture = time.monotonic()
        if not self.is_opened:
            return False, None, t_capture

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None, t_capture

        return True, frame, t_capture

    def stream_frames(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """Generator yielding live camera frames."""
        if not self.is_opened and not self.open():
            raise CameraError(f"Could not open camera device at index {self.camera_index}.")

        try:
            while self.is_opened:
                ret, frame, t_cap = self.read_frame()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                yield frame, t_cap
        finally:
            self.release()

    def release(self):
        """Releases the camera hardware device cleanly."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._is_opened = False
