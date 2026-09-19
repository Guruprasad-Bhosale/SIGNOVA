"""
Unit tests for SIGNOVA Phase 3 Feature Groups, Variable-Length Collation, and Model Backbones.
"""

import numpy as np
import pytest
import torch

from signova.data.collate import PadCollate
from signova.features.feature_groups import (
    LandmarkGroup,
    get_landmark_group_indices,
    get_landmark_group_size,
    slice_landmark_tensor,
)
from signova.models.baseline_pooled import StaticPooledMLP
from signova.models.baseline_rnn import BaselineRNN
from signova.models.baseline_tcn import BaselineTCN


def test_feature_groups_slicing():
    assert get_landmark_group_size(LandmarkGroup.FULL) == 543
    assert get_landmark_group_size(LandmarkGroup.HANDS_POSE) == 75
    assert get_landmark_group_size(LandmarkGroup.HANDS) == 42
    assert get_landmark_group_size(LandmarkGroup.POSE) == 33
    assert get_landmark_group_size(LandmarkGroup.FACE) == 468

    # Test slicing on (T, 543, 3) NumPy array
    arr = np.random.uniform(0, 1, (20, 543, 3)).astype(np.float32)
    sliced_hands = slice_landmark_tensor(arr, group_name=LandmarkGroup.HANDS)
    assert sliced_hands.shape == (20, 42, 3)

    sliced_hp = slice_landmark_tensor(arr, group_name=LandmarkGroup.HANDS_POSE)
    assert sliced_hp.shape == (20, 75, 3)

    # Test slicing on (B, T, 543, 3) PyTorch tensor
    t_tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 20, 543, 3)
    t_sliced = slice_landmark_tensor(t_tensor, group_name=LandmarkGroup.HANDS)
    assert t_sliced.shape == (1, 20, 42, 3)


def test_pad_collate_variable_length():
    collate_fn = PadCollate()

    # 3 samples with variable lengths 10, 15, 20
    batch = [
        {
            "features": torch.zeros((10, 42, 3)),
            "detection_mask": torch.ones((10, 4)),
            "label": torch.tensor(1),
            "sample_id": "s1",
            "session_id": "sess_1",
            "class_name": "HELLO",
        },
        {
            "features": torch.zeros((15, 42, 3)),
            "detection_mask": torch.ones((15, 4)),
            "label": torch.tensor(2),
            "sample_id": "s2",
            "session_id": "sess_1",
            "class_name": "THANK_YOU",
        },
        {
            "features": torch.zeros((20, 42, 3)),
            "detection_mask": torch.ones((20, 4)),
            "label": torch.tensor(3),
            "sample_id": "s3",
            "session_id": "sess_2",
            "class_name": "PLEASE",
        },
    ]

    collated = collate_fn(batch)
    assert collated["features"].shape == (3, 20, 42, 3)
    assert collated["padding_mask"].shape == (3, 20)
    assert collated["detection_mask"].shape == (3, 20, 4)
    assert collated["labels"].shape == (3,)
    assert collated["lengths"].tolist() == [10, 15, 20]

    # Check validity mask values
    assert collated["padding_mask"][0, 9].item() is True
    assert collated["padding_mask"][0, 10].item() is False


def test_static_pooled_mlp_forward():
    model = StaticPooledMLP(num_landmarks=42, num_classes=10, projection_dim=64, hidden_dim=64)
    x = torch.randn(4, 30, 42, 3)
    mask = torch.ones((4, 30), dtype=torch.bool)
    mask[0, 20:] = False

    out = model(x, padding_mask=mask)
    assert out.shape == (4, 10)


def test_baseline_rnn_forward_and_backward():
    model = BaselineRNN(num_landmarks=42, num_classes=10, projection_dim=64, hidden_size=64, num_layers=1)
    x = torch.randn(4, 25, 42, 3, requires_grad=True)
    mask = torch.ones((4, 25), dtype=torch.bool)
    lengths = torch.tensor([25, 20, 15, 10], dtype=torch.long)

    out = model(x, padding_mask=mask, lengths=lengths)
    assert out.shape == (4, 10)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_baseline_tcn_forward_and_backward():
    model = BaselineTCN(num_landmarks=42, num_classes=10, projection_dim=64, channels=[64, 64])
    x = torch.randn(4, 25, 42, 3, requires_grad=True)
    mask = torch.ones((4, 25), dtype=torch.bool)

    out = model(x, padding_mask=mask)
    assert out.shape == (4, 10)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
