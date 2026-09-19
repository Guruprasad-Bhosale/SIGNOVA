#!/usr/bin/env python
"""
Real-time webcam demo for ISL translation.

Usage:
    python src/inference/demo.py --checkpoint checkpoints/best.pt
"""

import argparse
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
import time
import threading
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.isl_model import ISLTranslator


class CameraStream:
    """Threaded camera capture to avoid blocking the main loop."""

    def __init__(self, src: int = 0, width: int = 1280, height: int = 720):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            # Fallback to DirectShow if default backend fails
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret = False
        self.frame = None
        self._lock = threading.Lock()
        self._stopped = False
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            with self._lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self._lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self._stopped = True
        self.cap.release()


class FastKeypointExtractor:
    """MediaPipe Holistic keypoint extractor (lite model for speed)."""

    def __init__(self, complexity: int = 0, min_det: float = 0.5, min_track: float = 0.5):
        import mediapipe as mp
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            model_complexity=complexity,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_track,
            smooth_landmarks=True,
        )
        self.NUM_POSE = 33
        self.NUM_FACE = 468
        self.NUM_HAND = 21

    def extract(self, frame_rgb: np.ndarray):
        """Extract 543 keypoints. Returns (543, 3) array and detection metadata."""
        results = self.holistic.process(frame_rgb)

        total = self.NUM_POSE + self.NUM_FACE + self.NUM_HAND * 2
        keypoints = np.zeros((total, 3), dtype=np.float32)
        meta = {"pose": False, "face": False, "lh": False, "rh": False}
        idx = 0

        if results.pose_landmarks:
            meta["pose"] = True
            for lm in results.pose_landmarks.landmark:
                keypoints[idx] = [lm.x, lm.y, lm.visibility]
                idx += 1
        else:
            idx += self.NUM_POSE

        if results.face_landmarks:
            meta["face"] = True
            for lm in results.face_landmarks.landmark:
                keypoints[idx] = [lm.x, lm.y, lm.z]
                idx += 1
        else:
            idx += self.NUM_FACE

        if results.left_hand_landmarks:
            meta["lh"] = True
            for lm in results.left_hand_landmarks.landmark:
                keypoints[idx] = [lm.x, lm.y, lm.z]
                idx += 1
        else:
            idx += self.NUM_HAND

        if results.right_hand_landmarks:
            meta["rh"] = True
            for lm in results.right_hand_landmarks.landmark:
                keypoints[idx] = [lm.x, lm.y, lm.z]
                idx += 1

        return keypoints, meta

    def close(self):
        self.holistic.close()


# Skeleton connection indices

_POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28),
]

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# Face mesh contour indices
_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
]
_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
_LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]


# Color palette (BGR)

COL_POSE_JOINT = (100, 255, 180)
COL_POSE_BONE = (60, 200, 140)
COL_LH_JOINT = (120, 140, 255)
COL_LH_BONE = (90, 110, 230)
COL_RH_JOINT = (255, 210, 80)
COL_RH_BONE = (230, 185, 50)
COL_FACE = (200, 200, 200)
COL_FACE_FEAT = (180, 220, 255)
COL_HUD_BG = (15, 15, 15)
COL_HUD_TEXT = (240, 240, 240)
COL_HUD_ACCENT = (0, 220, 180)
COL_HUD_DIM = (100, 100, 100)
COL_CONF_HIGH = (0, 255, 160)
COL_CONF_LOW = (80, 80, 80)


# Drawing helpers

def _draw_chain(frame, keypoints, indices, offset, w, h, color, thickness=1, closed=False):
    """Draw connected chain of keypoints."""
    pts = []
    for i in indices:
        x, y = keypoints[offset + i, 0], keypoints[offset + i, 1]
        if x > 0 or y > 0:
            pts.append((int(x * w), int(y * h)))
        else:
            pts.append(None)
    for j in range(len(pts) - 1):
        if pts[j] is not None and pts[j + 1] is not None:
            cv2.line(frame, pts[j], pts[j + 1], color, thickness, cv2.LINE_AA)


def _draw_joints(frame, keypoints, n, offset, w, h, color, radius, conf_thresh=0.0):
    """Draw joint circles."""
    pts = {}
    for i in range(n):
        x, y, c = keypoints[offset + i]
        if (x > 0 or y > 0) and c > conf_thresh:
            px, py = int(x * w), int(y * h)
            pts[i] = (px, py)
            cv2.circle(frame, (px, py), radius, color, -1, cv2.LINE_AA)
            # Outer ring for visibility
            cv2.circle(frame, (px, py), radius + 2, (*color[:2], color[2] // 3), 1, cv2.LINE_AA)
    return pts


def _draw_bones(frame, pts, connections, color, thickness=2):
    """Draw bone connections between joints."""
    for a, b in connections:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], color, thickness, cv2.LINE_AA)


def draw_skeleton(frame, keypoints, meta):
    """Draw skeleton overlay on the frame."""
    h, w = frame.shape[:2]
    POSE_OFF = 0
    FACE_OFF = 33
    LH_OFF = 33 + 468
    RH_OFF = 33 + 468 + 21

    # Face contour
    if meta["face"]:
        _draw_chain(frame, keypoints, _FACE_OVAL, FACE_OFF, w, h, COL_FACE, 1)
        _draw_chain(frame, keypoints, _LEFT_EYE, FACE_OFF, w, h, COL_FACE_FEAT, 1)
        _draw_chain(frame, keypoints, _RIGHT_EYE, FACE_OFF, w, h, COL_FACE_FEAT, 1)
        _draw_chain(frame, keypoints, _LIPS_OUTER, FACE_OFF, w, h, COL_FACE_FEAT, 1)

    # Pose
    if meta["pose"]:
        pts = _draw_joints(frame, keypoints, 33, POSE_OFF, w, h, COL_POSE_JOINT, 4, conf_thresh=0.3)
        _draw_bones(frame, pts, _POSE_CONNECTIONS, COL_POSE_BONE, 2)

    # Left hand
    if meta["lh"]:
        pts = _draw_joints(frame, keypoints, 21, LH_OFF, w, h, COL_LH_JOINT, 5)
        _draw_bones(frame, pts, _HAND_CONNECTIONS, COL_LH_BONE, 2)

    # Right hand
    if meta["rh"]:
        pts = _draw_joints(frame, keypoints, 21, RH_OFF, w, h, COL_RH_JOINT, 5)
        _draw_bones(frame, pts, _HAND_CONNECTIONS, COL_RH_BONE, 2)

    return frame


def draw_hud(frame, glosses, confidence, fps, buf_len, is_signing):
    """Draw the overlay HUD (FPS, confidence, predictions)."""
    h, w = frame.shape[:2]

    # Top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), COL_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Status indicator
    status_color = COL_CONF_HIGH if is_signing else COL_CONF_LOW
    cv2.circle(frame, (20, 25), 6, status_color, -1, cv2.LINE_AA)
    status_text = "DETECTING" if is_signing else "IDLE"
    cv2.putText(frame, status_text, (34, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)

    # FPS counter
    fps_text = f"{fps:.0f} FPS"
    cv2.putText(frame, fps_text, (w - 110, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_HUD_ACCENT, 1, cv2.LINE_AA)

    # Confidence bar
    bar_x, bar_y, bar_w, bar_h = w - 140, 40, 120, 8
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    fill = min(confidence, 1.0)
    bar_color = COL_CONF_HIGH if fill > 0.3 else COL_CONF_LOW
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * fill), bar_y + bar_h), bar_color, -1)
    cv2.putText(frame, f"Conf: {confidence:.0%}", (bar_x, bar_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_HUD_DIM, 1, cv2.LINE_AA)

    # Buffer indicator
    buf_fill = min(buf_len / 64, 1.0)
    cv2.rectangle(frame, (bar_x, bar_y + 30), (bar_x + bar_w, bar_y + 38), (40, 40, 40), -1)
    cv2.rectangle(frame, (bar_x, bar_y + 30), (bar_x + int(bar_w * buf_fill), bar_y + 38),
                  COL_HUD_ACCENT, -1)
    cv2.putText(frame, f"Buffer: {buf_len}/64", (bar_x, bar_y + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_HUD_DIM, 1, cv2.LINE_AA)

    # Gloss output
    if is_signing and glosses:
        gloss_text = "  ".join(glosses)
    else:
        gloss_text = ""

    if gloss_text:
        text_size = cv2.getTextSize(gloss_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)[0]
        text_x = max(15, (w - text_size[0]) // 2)

        # Bottom banner
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 65), (w, h), COL_HUD_BG, -1)
        cv2.addWeighted(overlay2, 0.70, frame, 0.30, 0, frame)

        cv2.putText(frame, gloss_text, (text_x, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "PREDICTED GLOSSES", (text_x, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_HUD_ACCENT, 1, cv2.LINE_AA)

    # Controls
    cv2.putText(frame, "Q Quit  |  C Clear", (15, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1, cv2.LINE_AA)

    return frame


# Decoding with confidence filtering

def decode_with_confidence(model, keypoints_tensor, lengths, device, conf_threshold=0.35):
    """
    Decode glosses and compute average prediction confidence.
    Returns (glosses, confidence_score, is_signing).
    """
    with torch.no_grad():
        output = model.forward(keypoints_tensor, lengths)
        logits = output["logits"]
        enc_lengths = output["lengths"]

        # Softmax probabilities
        probs = F.softmax(logits, dim=-1)                # (1, T, V)
        max_probs, predictions = probs.max(dim=-1)       # (1, T)

        length = enc_lengths[0].item()
        pred_seq = predictions[0, :length].cpu().tolist()
        conf_seq = max_probs[0, :length].cpu().tolist()

        # CTC collapse: remove blanks and consecutive duplicates
        blank = 0
        decoded = []
        decoded_confs = []
        prev = blank
        for p, c in zip(pred_seq, conf_seq):
            if p != blank and p != prev:
                decoded.append(p)
                decoded_confs.append(c)
            prev = p

        # Average confidence of non-blank predictions
        avg_conf = float(np.mean(decoded_confs)) if decoded_confs else 0.0

        # Convert to gloss strings
        idx_to_gloss = model.idx_to_gloss
        glosses = [idx_to_gloss.get(i, "") for i in decoded]
        glosses = [g for g in glosses if g and g not in ("<blank>", "<sos>", "<eos>", "<unk>")]

        # Only show predictions if confidence is above threshold
        is_signing = avg_conf > conf_threshold and len(glosses) > 0

        return glosses if is_signing else [], avg_conf, is_signing


# Main

def main():
    parser = argparse.ArgumentParser(description="ISL Translator demo")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (0-1)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load model from config
    import yaml
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_cfg = config.get("model", {})
    trans_cfg = model_cfg.get("transformer", {})
    stgcn_cfg = model_cfg.get("stgcn", {})

    print("[1/3] Loading model weights...")
    model = ISLTranslator.load(
        args.checkpoint,
        use_translation=False,
        d_model=trans_cfg.get("d_model", 256),
        nhead=trans_cfg.get("nhead", 8),
        num_encoder_layers=trans_cfg.get("num_encoder_layers", 4),
        stgcn_hidden=stgcn_cfg.get("hidden_channels", 64),
        stgcn_layers=stgcn_cfg.get("num_layers", 4),
    )
    model = model.to(device).eval()

    # Camera and extractor setup
    print("[2/3] Starting camera stream...")
    cam = CameraStream(src=args.camera, width=args.width, height=args.height)
    time.sleep(0.5)

    print("[3/3] Initializing MediaPipe...")
    extractor = FastKeypointExtractor(complexity=0)

    keypoint_buffer = deque(maxlen=128)
    current_glosses = []
    current_conf = 0.0
    is_signing = False
    last_inference_time = 0.0
    inference_interval = 0.8

    # FPS tracking
    frame_times = deque(maxlen=30)

    print(f"Demo running! Confidence threshold: {args.conf:.0%}")
    print("Press Q to quit, C to clear buffer.")

    while True:
        t0 = time.perf_counter()

        ret, frame = cam.read()
        if not ret or frame is None:
            continue

        # Extract keypoints
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kps, meta = extractor.extract(rgb)
        keypoint_buffer.append(kps)

        # Run model inference at fixed intervals
        now = time.perf_counter()
        if now - last_inference_time > inference_interval and len(keypoint_buffer) >= 16:
            last_inference_time = now
            buf = np.array(list(keypoint_buffer))[-64:]
            inp = torch.tensor(buf, dtype=torch.float32).unsqueeze(0).to(device)
            lengths = torch.tensor([len(buf)]).to(device)
            current_glosses, current_conf, is_signing = decode_with_confidence(
                model, inp, lengths, device, conf_threshold=args.conf
            )

        # Render overlay
        draw_skeleton(frame, kps, meta)

        # FPS
        frame_times.append(time.perf_counter() - t0)
        fps = len(frame_times) / max(sum(frame_times), 1e-9)

        draw_hud(frame, current_glosses, current_conf, fps, len(keypoint_buffer), is_signing)

        cv2.imshow("ISL Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            keypoint_buffer.clear()
            current_glosses = []
            current_conf = 0.0
            is_signing = False

    cam.stop()
    extractor.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
