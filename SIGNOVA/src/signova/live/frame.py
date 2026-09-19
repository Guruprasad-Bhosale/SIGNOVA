"""
Frame Decoding and Validation for SIGNOVA Live Camera Runtime.
"""

import base64
import io
import time
from typing import Optional, Tuple, Union
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def decode_frame(
    raw_input: Union[np.ndarray, bytes, str],
) -> Tuple[np.ndarray, float]:
    """
    Decodes incoming frame from raw ndarray, binary bytes, or base64 data URL string.

    Returns:
        bgr_frame: Decoded frame in BGR format (H, W, 3) as uint8.
        timestamp: Decode timestamp in monotonic seconds.
    """
    t_now = time.monotonic()

    if isinstance(raw_input, np.ndarray):
        frame = raw_input
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            if HAS_CV2:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                frame = frame[:, :, :3]
        return frame.astype(np.uint8), t_now

    if isinstance(raw_input, str):
        # Handle data:image/...;base64,...
        if "," in raw_input:
            raw_input = raw_input.split(",", 1)[1]
        byte_data = base64.b64decode(raw_input)
    elif isinstance(raw_input, (bytes, bytearray)):
        byte_data = bytes(raw_input)
    else:
        raise ValueError(f"Unsupported frame input type: {type(raw_input)}")

    if HAS_CV2:
        nparr = np.frombuffer(byte_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("cv2.imdecode failed to decode image buffer.")
        return frame, t_now
    else:
        # Fallback using PIL
        from PIL import Image
        img = Image.open(io.BytesIO(byte_data)).convert("RGB")
        rgb = np.array(img, dtype=np.uint8)
        bgr = rgb[:, :, ::-1]  # Convert RGB to BGR
        return bgr, t_now


def validate_frame(frame: np.ndarray, min_width: int = 64, min_height: int = 64) -> bool:
    """Validates frame dimensions, non-emptiness, and channel format."""
    if not isinstance(frame, np.ndarray):
        return False
    if frame.size == 0 or frame.ndim != 3:
        return False
    h, w, c = frame.shape
    if h < min_height or w < min_width or c != 3:
        return False
    if np.isnan(frame).any() or np.isinf(frame).any():
        return False
    return True
