"""
Continuous and Streaming Inference Interfaces for SIGNOVA.

MODES:
1. Windowed Offline Inference (Primary):
   Processes continuous video through overlapping temporal windows, merging per-frame
   representations across the complete video.
2. Experimental Rolling Buffer Stream (Optional):
   Simulates streaming input frames through a FIFO feature buffer.

IMPORTANT ARCHITECTURAL NOTE:
Bidirectional recurrent models (BiGRU) and non-causal TCNs observe both past and future
context within a window. Therefore, real-time zero-latency causal inference is not claimed.
Windowed offline inference is the standard operational mode.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from signova.data.windowing import SlidingWindowExtractor
from signova.features.feature_groups import LandmarkGroup, slice_landmark_tensor
from signova.models.boundary_head import HeuristicBoundaryDetector


class WindowedOfflineInference:
    """
    Standard windowed inference pipeline for continuous video landmark sequences.
    """

    def __init__(
        self,
        model: nn.Module,
        window_size: int = 64,
        stride: int = 32,
        device: Optional[torch.device] = None,
        landmark_group: Union[str, LandmarkGroup] = LandmarkGroup.FULL,
    ):
        self.model = model
        self.window_size = window_size
        self.stride = stride
        self.landmark_group = landmark_group
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.extractor = SlidingWindowExtractor(window_size=window_size, stride=stride)
        self.boundary_detector = HeuristicBoundaryDetector()

    def process_sequence(
        self,
        landmarks: Union[np.ndarray, torch.Tensor],
        fps: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Processes full continuous sequence through sliding windows and merges representations.

        Args:
            landmarks: Array/tensor of shape (T, 543, 3) or (T, num_joints, 3).
            fps: Frame rate.

        Returns:
            Dict containing per-frame embeddings, candidate boundaries, and diagnostic summary.
        """
        if isinstance(landmarks, np.ndarray):
            landmarks = slice_landmark_tensor(landmarks, group_name=self.landmark_group)
            feat_tensor = torch.from_numpy(landmarks.astype(np.float32))
        else:
            landmarks = slice_landmark_tensor(landmarks, group_name=self.landmark_group)
            feat_tensor = landmarks.float()

        T = feat_tensor.shape[0]
        if T == 0:
            return {"error": "Empty sequence provided."}

        # Extract sliding windows
        windows = self.extractor.extract_windows(feat_tensor)
        if not windows:
            return {"error": "No valid windows extracted from sequence."}

        # Process each window through model
        window_embeddings = []
        with torch.no_grad():
            for win in windows:
                w_feat = win["features"].unsqueeze(0).to(self.device) # (1, W, joints, 3)
                w_mask = win["padding_mask"].unsqueeze(0).to(self.device) # (1, W)
                w_len = torch.tensor([win["valid_frames"]], dtype=torch.long, device=self.device)

                if hasattr(self.model, "encoder"):
                    emb = self.model.encoder(w_feat, padding_mask=w_mask, lengths=w_len)
                else:
                    emb = self.model(w_feat, padding_mask=w_mask, lengths=w_len)

                window_embeddings.append(emb.squeeze(0).cpu().numpy()) # (W, D)

        # Merge overlapping window representations into sequence timeline (T, D)
        out_dim = window_embeddings[0].shape[-1]
        merged_emb = np.zeros((T, out_dim), dtype=np.float32)
        count_map = np.zeros((T, 1), dtype=np.float32)

        for win, w_emb in zip(windows, window_embeddings):
            s = win["start_frame"]
            e = win["end_frame"]
            valid_w = e - s
            merged_emb[s:e] += w_emb[:valid_w]
            count_map[s:e] += 1.0

        count_map = np.maximum(count_map, 1.0)
        merged_emb = merged_emb / count_map

        # Heuristic candidate boundary proposals
        proposals = self.boundary_detector.compute_candidate_proposals(feat_tensor, fps=fps)

        return {
            "total_frames": T,
            "duration_sec": round(T / fps, 3),
            "window_count": len(windows),
            "merged_embeddings": merged_emb,
            "embedding_dimension": out_dim,
            "candidate_boundaries": proposals["candidate_boundaries"],
            "candidate_segments": proposals["candidate_segments"],
            "energy_profile": proposals["energy_profile"],
            "mode": "WINDOWED_OFFLINE_INFERENCE",
        }


class ExperimentalRollingBufferStream:
    """
    Experimental FIFO rolling buffer simulating streaming frames.
    """

    def __init__(
        self,
        model: nn.Module,
        buffer_size: int = 64,
        step_interval: int = 8,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.buffer_size = buffer_size
        self.step_interval = step_interval
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.buffer: List[torch.Tensor] = []
        self.frame_counter = 0

    def reset(self):
        """Resets the streaming buffer state."""
        self.buffer.clear()
        self.frame_counter = 0

    def push_frame(self, frame_tensor: torch.Tensor) -> Optional[Dict[str, Any]]:
        """
        Pushes a single frame of shape (num_joints, 3) or (input_dim).
        Returns prediction dictionary every `step_interval` frames once buffer has sufficient context.
        """
        self.buffer.append(frame_tensor.cpu().clone())
        self.frame_counter += 1

        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        # Trigger prediction on step interval
        if len(self.buffer) >= 16 and (self.frame_counter % self.step_interval == 0):
            current_len = len(self.buffer)
            buf_tensor = torch.stack(self.buffer, dim=0).unsqueeze(0).to(self.device) # (1, T_buf, ...)
            mask = torch.ones((1, current_len), dtype=torch.bool, device=self.device)
            lengths = torch.tensor([current_len], dtype=torch.long, device=self.device)

            with torch.no_grad():
                if hasattr(self.model, "encoder"):
                    emb = self.model.encoder(buf_tensor, padding_mask=mask, lengths=lengths)
                else:
                    emb = self.model(buf_tensor, padding_mask=mask, lengths=lengths)

            return {
                "frame_index": self.frame_counter,
                "buffer_length": current_len,
                "latest_embedding": emb[0, -1].cpu().numpy().tolist(),
                "status": "STREAM_UPDATE",
            }

        return None
