"""
Phase 6 Sequential Training & Optimization Tests for SIGNOVA.

Verifies:
1. Optimization step and gradient backpropagation.
2. Gradient clipping safety.
3. Checkpoint saving and reloading.
"""

from pathlib import Path
import pytest
import torch
from signova.models.ctc_recognizer import CTCContinuousRecognizer


def test_sequential_ctc_training_optimization_step(tmp_path: Path):
    model = CTCContinuousRecognizer(
        num_landmarks=42,
        num_classes=6,
        backbone="gru",
        projection_dim=32,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    B, T = 2, 20
    feats = torch.randn(B, T, 42, 3)
    f_lens = torch.tensor([20, 18], dtype=torch.long)
    targets = torch.tensor([[1, 2, 0], [3, 4, 1]], dtype=torch.long)
    t_lens = torch.tensor([2, 3], dtype=torch.long)

    model.train()
    optimizer.zero_grad()
    out = model(feats, lengths=f_lens, targets=targets, target_lengths=t_lens)
    loss = out["loss"]
    assert loss is not None
    loss.backward()

    # Check gradients exist
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Checkpoint test
    ckpt_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt_path)
    assert ckpt_path.is_file()

    new_model = CTCContinuousRecognizer(
        num_landmarks=42,
        num_classes=6,
        backbone="gru",
        projection_dim=32,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
    )
    new_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    new_model.eval()

    with torch.no_grad():
        out_orig = model(feats, lengths=f_lens)
        out_new = new_model(feats, lengths=f_lens)
        assert torch.allclose(out_orig["logits"], out_new["logits"], atol=1e-4)
