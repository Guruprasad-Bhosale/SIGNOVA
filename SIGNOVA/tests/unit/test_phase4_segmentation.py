"""
Unit Tests for Phase 4 Temporal Boundary Modeling and Heuristic Proposals.
"""

import numpy as np
import pytest
import torch

from signova.models.boundary_head import HeuristicBoundaryDetector, TemporalBoundaryHead


def test_heuristic_boundary_detector_proposals():
    detector = HeuristicBoundaryDetector(energy_threshold=0.05, smooth_window=3, min_segment_length=4)

    # Construct synthetic sequence with 2 active motion bursts separated by stationary frames
    T, joints = 30, 75
    seq = np.zeros((T, joints, 3), dtype=np.float32)

    # Burst 1: frames 5..12
    for t in range(5, 13):
        seq[t] = t * 0.1

    # Burst 2: frames 20..27
    for t in range(20, 28):
        seq[t] = (30 - t) * 0.1

    proposals = detector.compute_candidate_proposals(seq, fps=30.0)

    assert "candidate_segments" in proposals
    assert "candidate_boundaries" in proposals
    assert proposals["mode"] == "HEURISTIC_PROPOSAL_ONLY"
    assert len(proposals["candidate_segments"]) >= 1

    for seg in proposals["candidate_segments"]:
        assert seg["label_status"] == "PREDICTED_CANDIDATE"
        assert seg["duration_sec"] > 0


def test_temporal_boundary_head_forward():
    B, T, D = 2, 50, 512
    head = TemporalBoundaryHead(input_dim=D, hidden_dim=128, num_states=3)
    embeddings = torch.randn((B, T, D), dtype=torch.float32)

    logits = head(embeddings)
    assert logits.shape == (B, T, 3)  # [Background, Active, Boundary]
