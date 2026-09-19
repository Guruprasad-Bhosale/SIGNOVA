"""
Landmark Overlay and Skeleton Visualization for SIGNOVA.

Renders 543-keypoint skeletal overlays on raw video frames or dark canvas,
including pose linkages, hand skeletons, and facial contours for visual validation.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np


# Standard MediaPipe Pose Links (indices 0..32)
POSE_CONNECTIONS = [
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Torso
    (23, 24),  # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (0, 1), (1, 2), (2, 3), (3, 7),  # Left eye/ear
    (0, 4), (4, 5), (5, 6), (6, 8),  # Right eye/ear
    (9, 10),  # Mouth
]

# Standard Hand Links (indices 0..20 relative to hand offset)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (5, 9), (9, 13), (13, 17),            # Palm base
]


def render_landmarks_on_image(
    image: Optional[np.ndarray],
    landmarks: np.ndarray,  # (543, 3)
    canvas_size: Tuple[int, int] = (720, 1280),
    title: Optional[str] = None,
) -> np.ndarray:
    """
    Render 543-keypoint skeleton overlay on an existing BGR frame or a dark canvas.

    Args:
        image: Optional BGR image. If None, a dark background is generated.
        landmarks: Array of shape (543, 3) [x, y, confidence]. Coordinates are normalized [0, 1] or centered.
        canvas_size: (H, W) if generating canvas.
        title: Optional title overlay text.

    Returns:
        Rendered BGR image.
    """
    h, w = image.shape[:2] if image is not None else canvas_size
    canvas = image.copy() if image is not None else np.zeros((h, w, 3), dtype=np.uint8)

    # Detect if coordinates are normalized [0, 1] or centered [-1, 1]
    coords = landmarks[:, :2].copy()
    confs = landmarks[:, 2]

    # If coordinates are centered (e.g. min < 0), remap to canvas pixel space
    if np.min(coords) < 0:
        # Centered normalized coordinates [-1, 1] -> map to [0.1, 0.9] of canvas
        px = ((coords[:, 0] * 0.4 + 0.5) * w).astype(np.int32)
        py = ((coords[:, 1] * 0.4 + 0.5) * h).astype(np.int32)
    else:
        # Standard normalized [0, 1]
        px = (coords[:, 0] * w).astype(np.int32)
        py = (coords[:, 1] * h).astype(np.int32)

    # 1. Draw Face (indices 33..500)
    for i in range(33, 501):
        if confs[i] > 0.1 and 0 <= px[i] < w and 0 <= py[i] < h:
            cv2.circle(canvas, (px[i], py[i]), 1, (180, 180, 180), -1)

    # 2. Draw Pose Connections (indices 0..32)
    for i, j in POSE_CONNECTIONS:
        if confs[i] > 0.2 and confs[j] > 0.2:
            pt1 = (px[i], py[i])
            pt2 = (px[j], py[j])
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv2.line(canvas, pt1, pt2, (255, 200, 0), 2)  # Cyan/Blue

    # Draw Pose Joint Points
    for i in range(33):
        if confs[i] > 0.2 and 0 <= px[i] < w and 0 <= py[i] < h:
            cv2.circle(canvas, (px[i], py[i]), 4, (0, 165, 255), -1)  # Orange joints

    # 3. Draw Left Hand (indices 501..521)
    lh_offset = 501
    for i, j in HAND_CONNECTIONS:
        idx1 = lh_offset + i
        idx2 = lh_offset + j
        if confs[idx1] > 0.1 and confs[idx2] > 0.1:
            pt1 = (px[idx1], py[idx1])
            pt2 = (px[idx2], py[idx2])
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv2.line(canvas, pt1, pt2, (0, 215, 255), 2)  # Gold/Yellow

    for i in range(21):
        idx = lh_offset + i
        if confs[idx] > 0.1 and 0 <= px[idx] < w and 0 <= py[idx] < h:
            cv2.circle(canvas, (px[idx], py[idx]), 3, (0, 255, 255), -1)

    # 4. Draw Right Hand (indices 522..542)
    rh_offset = 522
    for i, j in HAND_CONNECTIONS:
        idx1 = rh_offset + i
        idx2 = rh_offset + j
        if confs[idx1] > 0.1 and confs[idx2] > 0.1:
            pt1 = (px[idx1], py[idx1])
            pt2 = (px[idx2], py[idx2])
            if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
                0 <= pt2[0] < w and 0 <= pt2[1] < h):
                cv2.line(canvas, pt1, pt2, (50, 205, 50), 2)  # Lime Green

    for i in range(21):
        idx = rh_offset + i
        if confs[idx] > 0.1 and 0 <= px[idx] < w and 0 <= py[idx] < h:
            cv2.circle(canvas, (px[idx], py[idx]), 3, (0, 255, 128), -1)

    # Title / Legend
    if title:
        cv2.putText(
            canvas,
            title,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def create_pilot_visualization_grid(
    sequence: np.ndarray,  # (T, 543, 3)
    sample_id: str,
    output_path: Union[str, Path],
    num_key_frames: int = 6,
) -> Path:
    """
    Create a multi-frame visualization grid of landmark progression across time.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    T = sequence.shape[0]
    if T == 0:
        return out_p

    # Pick evenly spaced frame indices
    step = max(1, T // num_key_frames)
    indices = [min(i * step, T - 1) for i in range(num_key_frames)]

    rendered_frames = []
    for idx in indices:
        lm = sequence[idx]
        title = f"{sample_id} | Frame {idx+1}/{T}"
        rendered = render_landmarks_on_image(None, lm, canvas_size=(480, 640), title=title)
        rendered_frames.append(rendered)

    # Arrange into a 2x3 or 1xN grid
    if num_key_frames == 6:
        row1 = np.hstack(rendered_frames[:3])
        row2 = np.hstack(rendered_frames[3:6])
        grid = np.vstack([row1, row2])
    else:
        grid = np.hstack(rendered_frames)

    cv2.imwrite(str(out_p), grid)
    return out_p
