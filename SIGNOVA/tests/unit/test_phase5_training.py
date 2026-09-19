"""
Unit Tests for Phase 5 Sequential CTC Trainer.
"""

from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from signova.models.ctc_recognizer import CTCContinuousRecognizer
from signova.recognition.sequential_trainer import SequentialCTCTrainer


class DummySequentialDataset(Dataset):
    def __init__(self, size: int = 4, seq_len: int = 40):
        self.size = size
        self.seq_len = seq_len

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "features": torch.randn((self.seq_len, 75, 3), dtype=torch.float32),
            "detection_mask": torch.ones((self.seq_len, 4), dtype=torch.float32),
            "timestamps_ms": torch.arange(self.seq_len, dtype=torch.float32) * 33.33,
            "target_sequence": torch.tensor([2, 3], dtype=torch.long),
            "sample_id": f"syn_{idx}",
            "session_id": "sess_01",
            "signer_id": "signer_01",
        }


def dummy_collate(batch):
    B = len(batch)
    T = batch[0]["features"].shape[0]
    feat = torch.stack([item["features"] for item in batch], dim=0)
    mask = torch.ones((B, T), dtype=torch.bool)
    lens = torch.tensor([T] * B, dtype=torch.long)
    targets = torch.stack([item["target_sequence"] for item in batch], dim=0)
    tgt_lens = torch.tensor([2] * B, dtype=torch.long)
    return {
        "features": feat,
        "padding_mask": mask,
        "lengths": lens,
        "targets": targets,
        "target_lengths": tgt_lens,
        "sample_ids": [item["sample_id"] for item in batch],
    }


def test_sequential_ctc_trainer_mini_run_and_checkpointing(tmp_path: Path):
    ds_train = DummySequentialDataset(size=4)
    ds_val = DummySequentialDataset(size=2)

    loader_train = DataLoader(ds_train, batch_size=2, collate_fn=dummy_collate)
    loader_val = DataLoader(ds_val, batch_size=2, collate_fn=dummy_collate)

    model = CTCContinuousRecognizer(num_landmarks=75, num_classes=10, backbone="gru")
    exp_dir = tmp_path / "test_exp"

    trainer = SequentialCTCTrainer(
        model=model,
        train_loader=loader_train,
        val_loader=loader_val,
        epochs=2,
        experiment_dir=exp_dir,
        device="cpu",
    )

    summary = trainer.train()
    assert summary["best_epoch"] >= 1
    assert (exp_dir / "best.pt").is_file()
    assert (exp_dir / "metrics.json").is_file()
