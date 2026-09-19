"""
Temporal Boundary Modeling and Heuristic Candidate Proposals for SIGNOVA.

IMPORTANT SCIENTIFIC CONSTRAINT:
- If ground-truth temporal boundaries are unavailable, supervised boundary training is NOT conducted.
- Unsupervised kinematic energy / change-point detections are strictly designated as
  "HEURISTIC CANDIDATE BOUNDARY PROPOSALS" for visualization and research exploration only.
- Heuristic proposals are NEVER used as ground-truth targets for CTC or supervised segmentation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


class HeuristicBoundaryDetector:
    """
    Computes heuristic candidate boundary proposals from landmark motion/kinematic energy.
    Used exclusively for exploratory visualization and temporal diagnostics.
    """

    def __init__(
        self,
        energy_threshold: float = 0.05,
        smooth_window: int = 5,
        min_segment_length: int = 8,
    ):
        """
        Args:
            energy_threshold: Relative velocity magnitude threshold for movement onset/offset.
            smooth_window: Moving average smoothing kernel size.
            min_segment_length: Minimum duration in frames for a candidate sign unit.
        """
        self.energy_threshold = energy_threshold
        self.smooth_window = smooth_window
        self.min_segment_length = min_segment_length

    def compute_candidate_proposals(
        self,
        features: Union[np.ndarray, torch.Tensor],
        fps: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Extracts candidate boundary intervals and per-frame motion energy.

        Args:
            features: Tensor or array of shape (T, num_joints, 3) or (T, input_dim).
            fps: Frame rate of the video.

        Returns:
            Dictionary containing:
            - 'energy_profile': List of frame-level kinematic energies.
            - 'candidate_boundaries': List of candidate frame indices where transitions occur.
            - 'candidate_segments': List of dicts with (start_frame, end_frame, start_sec, end_sec).
            - 'mode': Explicitly marked 'HEURISTIC_PROPOSAL_ONLY'
        """
        if isinstance(features, torch.Tensor):
            feat_np = features.detach().cpu().numpy()
        else:
            feat_np = np.asarray(features)

        T = feat_np.shape[0]
        if T < 2:
            return {
                "energy_profile": [0.0] * T,
                "candidate_boundaries": [],
                "candidate_segments": [],
                "mode": "HEURISTIC_PROPOSAL_ONLY",
            }

        # Reshape to (T, -1)
        feat_flat = feat_np.reshape(T, -1)

        # 1. Compute frame-to-frame velocity norm (kinematic energy)
        velocity = np.diff(feat_flat, axis=0) # (T-1, dim)
        energy = np.linalg.norm(velocity, axis=1) # (T-1,)
        energy = np.pad(energy, (1, 0), mode="edge") # (T,)

        # 2. Temporal Smoothing
        if self.smooth_window > 1 and T >= self.smooth_window:
            kernel = np.ones(self.smooth_window) / self.smooth_window
            smoothed_energy = np.convolve(energy, kernel, mode="same")
        else:
            smoothed_energy = energy

        # Normalize energy to [0, 1] relative scale
        e_max = np.max(smoothed_energy)
        e_norm = (smoothed_energy / e_max) if e_max > 1e-6 else smoothed_energy

        # 3. Detect energy minima / transitions
        is_active = e_norm >= self.energy_threshold

        segments = []
        in_segment = False
        seg_start = 0

        for t in range(T):
            if is_active[t] and not in_segment:
                in_segment = True
                seg_start = t
            elif not is_active[t] and in_segment:
                in_segment = False
                seg_end = t
                if (seg_end - seg_start) >= self.min_segment_length:
                    segments.append({
                        "start_frame": seg_start,
                        "end_frame": seg_end,
                        "start_sec": round(seg_start / fps, 3),
                        "end_sec": round(seg_end / fps, 3),
                        "duration_sec": round((seg_end - seg_start) / fps, 3),
                        "mean_energy": float(np.mean(e_norm[seg_start:seg_end])),
                        "label_status": "PREDICTED_CANDIDATE",
                    })

        if in_segment and (T - seg_start) >= self.min_segment_length:
            segments.append({
                "start_frame": seg_start,
                "end_frame": T,
                "start_sec": round(seg_start / fps, 3),
                "end_sec": round(T / fps, 3),
                "duration_sec": round((T - seg_start) / fps, 3),
                "mean_energy": float(np.mean(e_norm[seg_start:T])),
                "label_status": "PREDICTED_CANDIDATE",
            })

        candidate_boundaries = [s["start_frame"] for s in segments] + [s["end_frame"] for s in segments]
        candidate_boundaries = sorted(list(set(candidate_boundaries)))

        return {
            "energy_profile": [round(float(v), 4) for v in e_norm],
            "candidate_boundaries": candidate_boundaries,
            "candidate_segments": segments,
            "total_candidate_segments": len(segments),
            "mode": "HEURISTIC_PROPOSAL_ONLY",
            "scientific_disclaimer": "Heuristic motion proposals for visualization. Not ground-truth boundaries.",
        }


class TemporalBoundaryHead(nn.Module):
    """
    Modular Neural Head for Frame-Wise Boundary State Prediction.
    Maps per-frame encoder representations (B, T, D) -> (B, T, 3) logits:
    [0: Background/Non-sign, 1: Sign-Active, 2: Boundary-Transition].
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        num_states: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_states),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: Tensor of shape (B, T, D).

        Returns:
            Logits of shape (B, T, num_states).
        """
        return self.head(embeddings)
