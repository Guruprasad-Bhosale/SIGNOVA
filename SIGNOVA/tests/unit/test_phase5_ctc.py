"""
Unit Tests for Phase 5 CTC Architecture, Gradients, and Feasibility.
"""

import pytest
import torch

from signova.models.ctc_recognizer import CTCContinuousRecognizer


def test_ctc_recognizer_synthetic_training_step():
    B, T, num_joints, vocab_size = 2, 50, 75, 12
    model = CTCContinuousRecognizer(num_landmarks=num_joints, num_classes=vocab_size, backbone="gru")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    feat = torch.randn((B, T, num_joints, 3), dtype=torch.float32, requires_grad=True)
    mask = torch.ones((B, T), dtype=torch.bool)
    lens = torch.tensor([T, T], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4], [5, 6, 0]], dtype=torch.long)
    tgt_lens = torch.tensor([3, 2], dtype=torch.long)

    model.train()
    optimizer.zero_grad()
    out = model(feat, padding_mask=mask, lengths=lens, targets=targets, target_lengths=tgt_lens)

    loss = out["loss"]
    assert loss is not None
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    optimizer.step()

    # Verify gradients computed
    assert model.ctc_head[-1].weight.grad is not None


def test_ctc_recognizer_tcn_backbone():
    B, T, num_joints, vocab_size = 2, 40, 75, 10
    model = CTCContinuousRecognizer(num_landmarks=num_joints, num_classes=vocab_size, backbone="tcn")
    model.eval()

    feat = torch.randn((B, T, num_joints, 3), dtype=torch.float32)
    mask = torch.ones((B, T), dtype=torch.bool)
    lens = torch.tensor([T, T], dtype=torch.long)

    out = model(feat, padding_mask=mask, lengths=lens)
    assert out["logits"].shape == (B, T, vocab_size)
