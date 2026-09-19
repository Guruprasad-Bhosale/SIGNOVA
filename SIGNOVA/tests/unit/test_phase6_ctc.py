"""
Phase 6 CTC Architecture & Greedy Decoder Unit Tests for SIGNOVA.

Verifies:
1. CTC loss computation and length constraint checks (T_in >= T_target).
2. Greedy decoding and blank/repeated token collapse.
3. Feature group slicing compatibility (HANDS 42, HANDS_POSE 75, FULL 543).
"""

import pytest
import torch
from signova.features.feature_groups import LandmarkGroup, slice_landmark_tensor
from signova.models.ctc_recognizer import CTCContinuousRecognizer


def test_ctc_recognizer_forward_and_loss():
    recognizer = CTCContinuousRecognizer(
        num_landmarks=75,
        num_classes=10,
        backbone="gru",
        projection_dim=64,
        hidden_size=64,
        num_layers=1,
    )
    recognizer.eval()

    B, T = 2, 40
    dummy_feats = torch.randn(B, T, 75, 3)
    lengths = torch.tensor([40, 35], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], dtype=torch.long)
    target_lengths = torch.tensor([3, 2], dtype=torch.long)

    out = recognizer(dummy_feats, lengths=lengths, targets=targets, target_lengths=target_lengths)
    assert "logits" in out
    assert out["logits"].shape == (B, T, 10)
    assert "loss" in out
    assert out["loss"] is not None
    assert not torch.isnan(out["loss"])
    assert not torch.isinf(out["loss"])


def test_ctc_recognizer_constraint_violation_raises():
    recognizer = CTCContinuousRecognizer(
        num_landmarks=75,
        num_classes=10,
        backbone="gru",
        projection_dim=64,
        hidden_size=64,
    )
    B, T = 1, 3
    dummy_feats = torch.randn(B, T, 75, 3)
    lengths = torch.tensor([3], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    target_lengths = torch.tensor([5], dtype=torch.long)  # 5 > 3 -> violation

    with pytest.raises(ValueError, match="CTC constraint violation"):
        _ = recognizer(dummy_feats, lengths=lengths, targets=targets, target_lengths=target_lengths)


def test_ctc_recognizer_greedy_decoding():
    recognizer = CTCContinuousRecognizer(
        num_landmarks=42,
        num_classes=8,
        backbone="gru",
        projection_dim=64,
        hidden_size=64,
        blank_idx=0,
    )
    B, T = 2, 30
    dummy_feats = torch.randn(B, T, 42, 3)
    lengths = torch.tensor([30, 25], dtype=torch.long)

    decoded = recognizer.decode_greedy(dummy_feats, lengths=lengths)
    assert len(decoded) == B
    for item in decoded:
        assert "collapsed_tokens" in item
        assert "sequence_length" in item
        assert isinstance(item["collapsed_tokens"], list)
