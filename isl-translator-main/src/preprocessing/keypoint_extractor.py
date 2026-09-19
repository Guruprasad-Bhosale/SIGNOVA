"""
MediaPipe Holistic Keypoint Extractor for ISL Videos.

Extracts 543 landmarks per frame:
- 33 pose landmarks (body)
- 468 face landmarks (expressions)
- 21 × 2 hand landmarks (left + right)
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KeypointExtractor:
    """
    Extract pose, face, and hand keypoints from video frames using MediaPipe Holistic.
    
    Attributes:
        num_pose: Number of pose landmarks (33)
        num_face: Number of face landmarks (468)
        num_hand: Number of hand landmarks per hand (21)
        total_landmarks: Total landmarks per frame (543)
    """
    
    NUM_POSE_LANDMARKS = 33
    NUM_FACE_LANDMARKS = 468
    NUM_HAND_LANDMARKS = 21
    TOTAL_LANDMARKS = 543  # 33 + 468 + 21 + 21
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ):
        """
        Initialize the MediaPipe Holistic model.
        
        Args:
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            static_image_mode: Whether to treat each frame independently
        """
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=2,  # 0, 1, or 2 (higher = more accurate)
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        logger.info(f"Initialized KeypointExtractor with {self.TOTAL_LANDMARKS} landmarks per frame")
    
    def extract_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Extract keypoints from a single frame.
        
        Args:
            frame: BGR image from OpenCV (H, W, 3)
            
        Returns:
            keypoints: Array of shape (543, 3) with (x, y, visibility/confidence)
            metadata: Dictionary with detection success for each component
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        
        # Initialize output array
        keypoints = np.zeros((self.TOTAL_LANDMARKS, 3), dtype=np.float32)
        
        metadata = {
            "pose_detected": False,
            "face_detected": False,
            "left_hand_detected": False,
            "right_hand_detected": False,
        }
        
        idx = 0
        
        # Extract pose landmarks (33 points)
        if results.pose_landmarks:
            metadata["pose_detected"] = True
            for lm in results.pose_landmarks.landmark:
                keypoints[idx] = [lm.x, lm.y, lm.visibility]
                idx += 1
        else:
            idx += self.NUM_POSE_LANDMARKS
        
        # Extract face landmarks (468 points)
        if results.face_landmarks:
            metadata["face_detected"] = True
            for lm in results.face_landmarks.landmark:
                keypoints[idx] = [lm.x, lm.y, lm.z]  # Face uses z instead of visibility
                idx += 1
        else:
            idx += self.NUM_FACE_LANDMARKS
        
        # Extract left hand landmarks (21 points)
        if results.left_hand_landmarks:
            metadata["left_hand_detected"] = True
            for lm in results.left_hand_landmarks.landmark:
                keypoints[idx] = [lm.x, lm.y, lm.z]
                idx += 1
        else:
            idx += self.NUM_HAND_LANDMARKS
        
        # Extract right hand landmarks (21 points)
        if results.right_hand_landmarks:
            metadata["right_hand_detected"] = True
            for lm in results.right_hand_landmarks.landmark:
                keypoints[idx] = [lm.x, lm.y, lm.z]
                idx += 1
        
        return keypoints, metadata
    
    def extract_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        skip_frames: int = 1,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Extract keypoints from all frames of a video.
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to process (None = all)
            skip_frames: Process every Nth frame (1 = all frames)
            show_progress: Whether to show progress bar
            
        Returns:
            keypoints: Array of shape (T, 543, 3)
            metadata: List of metadata dicts for each frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Total frames: {total_frames}, FPS: {fps:.2f}")
        
        all_keypoints = []
        all_metadata = []
        frame_idx = 0
        
        pbar = tqdm(total=total_frames, disable=not show_progress, desc="Extracting keypoints")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            if frame_idx % skip_frames == 0:
                keypoints, metadata = self.extract_frame(frame)
                all_keypoints.append(keypoints)
                all_metadata.append(metadata)
            
            frame_idx += 1
            pbar.update(1)
            
            if max_frames and len(all_keypoints) >= max_frames:
                break
        
        pbar.close()
        cap.release()
        
        keypoints_array = np.array(all_keypoints, dtype=np.float32)
        logger.info(f"Extracted keypoints shape: {keypoints_array.shape}")
        
        return keypoints_array, all_metadata
    
    def interpolate_missing(
        self,
        keypoints: np.ndarray,
        metadata: List[Dict],
        component: str = "pose",
    ) -> np.ndarray:
        """
        Interpolate missing landmarks for frames where detection failed.
        
        Args:
            keypoints: Array of shape (T, 543, 3)
            metadata: List of metadata dicts
            component: Which component to interpolate ("pose", "face", "left_hand", "right_hand")
            
        Returns:
            Interpolated keypoints array
        """
        keypoints = keypoints.copy()
        key = f"{component}_detected"
        
        # Find detected frame indices
        detected_indices = [i for i, m in enumerate(metadata) if m[key]]
        
        if len(detected_indices) < 2:
            logger.warning(f"Not enough detected frames for {component} interpolation")
            return keypoints
        
        # Get landmark range for component
        if component == "pose":
            start, end = 0, self.NUM_POSE_LANDMARKS
        elif component == "face":
            start = self.NUM_POSE_LANDMARKS
            end = start + self.NUM_FACE_LANDMARKS
        elif component == "left_hand":
            start = self.NUM_POSE_LANDMARKS + self.NUM_FACE_LANDMARKS
            end = start + self.NUM_HAND_LANDMARKS
        elif component == "right_hand":
            start = self.NUM_POSE_LANDMARKS + self.NUM_FACE_LANDMARKS + self.NUM_HAND_LANDMARKS
            end = start + self.NUM_HAND_LANDMARKS
        else:
            raise ValueError(f"Unknown component: {component}")
        
        # Linear interpolation for missing frames
        for i in range(len(keypoints)):
            if not metadata[i][key]:
                # Find nearest detected frames
                prev_idx = max([j for j in detected_indices if j < i], default=None)
                next_idx = min([j for j in detected_indices if j > i], default=None)
                
                if prev_idx is not None and next_idx is not None:
                    # Interpolate
                    alpha = (i - prev_idx) / (next_idx - prev_idx)
                    keypoints[i, start:end] = (
                        (1 - alpha) * keypoints[prev_idx, start:end] +
                        alpha * keypoints[next_idx, start:end]
                    )
                elif prev_idx is not None:
                    keypoints[i, start:end] = keypoints[prev_idx, start:end]
                elif next_idx is not None:
                    keypoints[i, start:end] = keypoints[next_idx, start:end]
        
        return keypoints
    
    def save_keypoints(
        self,
        keypoints: np.ndarray,
        output_path: str,
        metadata: Optional[List[Dict]] = None,
    ):
        """
        Save extracted keypoints to a .npz file.
        
        Args:
            keypoints: Array of shape (T, 543, 3)
            output_path: Path to save the .npz file
            metadata: Optional metadata to save
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {"keypoints": keypoints}
        if metadata:
            # Convert metadata to arrays
            data["pose_detected"] = np.array([m["pose_detected"] for m in metadata])
            data["face_detected"] = np.array([m["face_detected"] for m in metadata])
            data["left_hand_detected"] = np.array([m["left_hand_detected"] for m in metadata])
            data["right_hand_detected"] = np.array([m["right_hand_detected"] for m in metadata])
        
        np.savez_compressed(output_path, **data)
        logger.info(f"Saved keypoints to: {output_path}")
    
    @staticmethod
    def load_keypoints(path: str) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Load keypoints from a .npz file.
        
        Args:
            path: Path to the .npz file
            
        Returns:
            keypoints: Array of shape (T, 543, 3)
            metadata: Dictionary of detection flags (or None)
        """
        data = np.load(path)
        keypoints = data["keypoints"]
        
        metadata = None
        if "pose_detected" in data:
            metadata = {
                "pose_detected": data["pose_detected"],
                "face_detected": data["face_detected"],
                "left_hand_detected": data["left_hand_detected"],
                "right_hand_detected": data["right_hand_detected"],
            }
        
        return keypoints, metadata
    
    def visualize_frame(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        draw_pose: bool = True,
        draw_face: bool = False,  # Often too dense
        draw_hands: bool = True,
    ) -> np.ndarray:
        """
        Draw keypoints on a frame for visualization.
        
        Args:
            frame: BGR image (H, W, 3)
            keypoints: Array of shape (543, 3)
            draw_pose: Whether to draw pose landmarks
            draw_face: Whether to draw face landmarks
            draw_hands: Whether to draw hand landmarks
            
        Returns:
            Frame with drawn keypoints
        """
        vis_frame = frame.copy()
        h, w = frame.shape[:2]
        
        idx = 0
        
        # Draw pose
        if draw_pose:
            for i in range(self.NUM_POSE_LANDMARKS):
                x, y, conf = keypoints[idx + i]
                if conf > 0.5:
                    cx, cy = int(x * w), int(y * h)
                    cv2.circle(vis_frame, (cx, cy), 3, (0, 255, 0), -1)
        idx += self.NUM_POSE_LANDMARKS
        
        # Draw face
        if draw_face:
            for i in range(self.NUM_FACE_LANDMARKS):
                x, y, _ = keypoints[idx + i]
                if x > 0 or y > 0:
                    cx, cy = int(x * w), int(y * h)
                    cv2.circle(vis_frame, (cx, cy), 1, (255, 0, 0), -1)
        idx += self.NUM_FACE_LANDMARKS
        
        # Draw left hand
        if draw_hands:
            for i in range(self.NUM_HAND_LANDMARKS):
                x, y, _ = keypoints[idx + i]
                if x > 0 or y > 0:
                    cx, cy = int(x * w), int(y * h)
                    cv2.circle(vis_frame, (cx, cy), 4, (0, 0, 255), -1)
        idx += self.NUM_HAND_LANDMARKS
        
        # Draw right hand
        if draw_hands:
            for i in range(self.NUM_HAND_LANDMARKS):
                x, y, _ = keypoints[idx + i]
                if x > 0 or y > 0:
                    cx, cy = int(x * w), int(y * h)
                    cv2.circle(vis_frame, (cx, cy), 4, (255, 255, 0), -1)
        
        return vis_frame
    
    def close(self):
        """Release MediaPipe resources."""
        self.holistic.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def batch_extract_keypoints(
    video_dir: str,
    output_dir: str,
    extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov"),
    **kwargs,
):
    """
    Extract keypoints from all videos in a directory.
    
    Args:
        video_dir: Directory containing videos
        output_dir: Directory to save keypoint files
        extensions: Video file extensions to process
        **kwargs: Arguments passed to KeypointExtractor.extract_video
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_files = []
    for ext in extensions:
        video_files.extend(video_dir.glob(f"**/*{ext}"))
    
    logger.info(f"Found {len(video_files)} videos to process")
    
    with KeypointExtractor() as extractor:
        for video_path in tqdm(video_files, desc="Processing videos"):
            # Create output path preserving directory structure
            relative_path = video_path.relative_to(video_dir)
            output_path = output_dir / relative_path.with_suffix(".npz")
            
            # Skip if already processed
            if output_path.exists():
                logger.info(f"Skipping (already exists): {output_path}")
                continue
            
            try:
                keypoints, metadata = extractor.extract_video(str(video_path), **kwargs)
                
                # Interpolate missing landmarks
                for component in ["pose", "left_hand", "right_hand"]:
                    keypoints = extractor.interpolate_missing(keypoints, metadata, component)
                
                extractor.save_keypoints(keypoints, str(output_path), metadata)
                
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract keypoints from ISL videos")
    parser.add_argument("--video", type=str, help="Path to single video")
    parser.add_argument("--video_dir", type=str, help="Directory of videos")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Show visualization")
    
    args = parser.parse_args()
    
    if args.video:
        with KeypointExtractor() as extractor:
            keypoints, metadata = extractor.extract_video(args.video)
            extractor.save_keypoints(
                keypoints,
                str(Path(args.output_dir) / Path(args.video).stem) + ".npz",
                metadata,
            )
    elif args.video_dir:
        batch_extract_keypoints(args.video_dir, args.output_dir)
