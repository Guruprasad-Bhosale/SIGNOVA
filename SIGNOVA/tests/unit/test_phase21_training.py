"""
Phase 21 BiGRU CTC Architecture and Synthetic Path Unit Tests.
"""

import numpy as np
import torch
from signova.operations.phase21_orchestrator import (
    Phase21BiGRUCTCModel,
    Phase21TorchDataset,
    collate_phase21_batch,
)


def test_bigru_ctc_model_forward_pass():
    model = Phase21BiGRUCTCModel(
        input_dim=150,
        hidden_dim=64,
        num_layers=2,
        num_classes=10,
        dropout=0.1,
    )
    # Batch=2, Time=30 frames, Dim=150
    x = torch.randn(2, 30, 150)
    lengths = torch.tensor([30, 25], dtype=torch.long)
    logits = model(x, lengths)

    assert logits.shape == (2, 30, 10)


def test_torch_dataset_and_collate():
    sample_data = [
        (np.random.randn(20, 150).astype(np.float32), [2, 3], "s1"),
        (np.random.randn(25, 150).astype(np.float32), [4, 5, 6], "s2"),
    ]
    ds = Phase21TorchDataset(sample_data)
    assert len(ds) == 2

    batch = [ds[0], ds[1]]
    padded_feats, padded_targets, input_lengths, target_lengths, sids = collate_phase21_batch(batch)

    assert padded_feats.shape == (2, 25, 150)
    assert padded_targets.shape == (2, 3)
    assert input_lengths.tolist() == [20, 25]
    assert target_lengths.tolist() == [2, 3]
    assert sids == ["s1", "s2"]
