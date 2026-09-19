"""
Unit tests for SIGNOVA Phase 3 Trainer, Checkpointing, and Evaluation Metrics.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from signova.data.collate import PadCollate
from signova.data.feature_dataset import SignLanguageFeatureDataset
from signova.models.baseline_pooled import StaticPooledMLP
from signova.models.baseline_rnn import BaselineRNN
from signova.recognition.trainer import SignLanguageTrainer


def test_sign_language_feature_dataset_and_collate():
    manifest_path = Path("data/manifests/phase3_manifest.csv")
    if not manifest_path.exists():
        pytest.skip("phase3_manifest.csv not yet generated")

    ds = SignLanguageFeatureDataset(manifest_path, split="train", landmark_group="hands")
    assert len(ds) > 0

    item = ds[0]
    assert "features" in item
    assert "detection_mask" in item
    assert "label" in item
    assert item["features"].shape[1] == 42  # Hands only

    collate_fn = PadCollate()
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert batch["features"].shape[0] == 4
    assert batch["features"].shape[2] == 42
    assert batch["padding_mask"].shape[0] == 4


def test_trainer_mini_run_and_checkpoint_roundtrip():
    manifest_path = Path("data/manifests/phase3_manifest.csv")
    if not manifest_path.exists():
        pytest.skip("phase3_manifest.csv not yet generated")

    with tempfile.TemporaryDirectory() as tmp_dir:
        ds_train = SignLanguageFeatureDataset(manifest_path, split="train", landmark_group="hands")
        ds_val = SignLanguageFeatureDataset(manifest_path, split="val", landmark_group="hands")

        collate_fn = PadCollate()
        train_loader = DataLoader(ds_train, batch_size=4, collate_fn=collate_fn)
        val_loader = DataLoader(ds_val, batch_size=4, collate_fn=collate_fn)

        model = BaselineRNN(num_landmarks=42, num_classes=10, projection_dim=32, hidden_size=32, num_layers=1)
        trainer = SignLanguageTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,
            device="cpu",
            experiment_dir=Path(tmp_dir),
        )

        summary = trainer.train()
        assert "best_epoch" in summary
        assert (Path(tmp_dir) / "best.pt").exists()
        assert (Path(tmp_dir) / "last.pt").exists()

        # Test Checkpoint Loading
        model_eval = BaselineRNN(num_landmarks=42, num_classes=10, projection_dim=32, hidden_size=32, num_layers=1)
        ckpt = torch.load(str(Path(tmp_dir) / "best.pt"), map_location="cpu")
        model_eval.load_state_dict(ckpt["model_state_dict"])
        model_eval.eval()

        dummy_x = torch.randn(2, 10, 42, 3)
        with torch.no_grad():
            out = model_eval(dummy_x)
        assert out.shape == (2, 10)
        trainer.close()
