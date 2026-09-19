#!/usr/bin/env python3
"""
Training & Experiment CLI for SIGNOVA Isolated Sign Recognition.

Supports data-only smoke tests, full baseline training, class weighting,
mixed precision, early stopping, and reproducible experiment tracking.
"""

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signova.data.collate import PadCollate
from signova.data.feature_dataset import SignLanguageFeatureDataset
from signova.features.feature_groups import LandmarkGroup, get_landmark_group_size
from signova.models.baseline_pooled import StaticPooledMLP
from signova.models.baseline_rnn import BaselineRNN
from signova.models.baseline_tcn import BaselineTCN
from signova.recognition.trainer import SignLanguageTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train SIGNOVA isolated sign recognition models.")
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(PROJECT_ROOT / "data" / "manifests" / "phase3_manifest.csv"),
        help="Path to feature manifest CSV.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="baseline_rnn",
        choices=["baseline_rnn", "baseline_tcn", "baseline_pooled"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--landmark-group",
        type=str,
        default="full",
        choices=["full", "hands_pose", "hands", "pose", "face"],
        help="Semantic landmark group (full=543, hands_pose=75, hands=42).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per step (default 4 for RTX 3050 safety).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Total training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to train on.",
    )
    parser.add_argument(
        "--data-smoke-test",
        action="store_true",
        help="Run 16-sample data-only verification test before/without model training.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 2-epoch tiny smoke training test.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision.",
    )
    parser.add_argument(
        "--exp-id",
        type=str,
        default=None,
        help="Custom experiment ID string.",
    )
    return parser.parse_args()


def run_data_smoke_test(manifest_path: Path, landmark_group: str) -> bool:
    print("\n" + "=" * 70)
    print("RUNNING DATA-ONLY SMOKE TEST (16 SAMPLES VERIFICATION)")
    print("=" * 70)

    ds_train = SignLanguageFeatureDataset(manifest_path, split="train", landmark_group=landmark_group)
    ds_val = SignLanguageFeatureDataset(manifest_path, split="val", landmark_group=landmark_group)
    ds_test = SignLanguageFeatureDataset(manifest_path, split="test", landmark_group=landmark_group)

    print(f"Train Dataset Samples: {len(ds_train)}")
    print(f"Val Dataset Samples:   {len(ds_val)}")
    print(f"Test Dataset Samples:  {len(ds_test)}")

    if len(ds_train) == 0:
        print("[FAIL] Train dataset is empty!")
        return False

    collate_fn = PadCollate()
    loader = DataLoader(ds_train, batch_size=min(16, len(ds_train)), shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))

    feat = batch["features"]
    p_mask = batch["padding_mask"]
    det_mask = batch["detection_mask"]
    labels = batch["labels"]
    lengths = batch["lengths"]

    print(f"Batch Features Shape:    {feat.shape} (dtype={feat.dtype})")
    print(f"Batch Padding Mask:      {p_mask.shape} (dtype={p_mask.dtype})")
    print(f"Batch Detection Mask:    {det_mask.shape} (dtype={det_mask.dtype})")
    print(f"Batch Labels:            {labels.tolist()}")
    print(f"Batch Lengths:           {lengths.tolist()}")

    # Assertions
    has_nan = torch.isnan(feat).any().item()
    has_inf = torch.isinf(feat).any().item()
    print(f"Has NaN Values:          {has_nan}")
    print(f"Has Inf Values:          {has_inf}")

    if has_nan or has_inf:
        print("[FAIL] NaN/Inf corruption detected in features batch!")
        return False

    expected_joints = get_landmark_group_size(landmark_group)
    if feat.shape[2] != expected_joints:
        print(f"[FAIL] Expected {expected_joints} joints for group '{landmark_group}', got {feat.shape[2]}")
        return False

    # Check zero session leakage between train and test
    train_sessions = set(ds_train.df["session_id"].unique())
    test_sessions = set(ds_test.df["session_id"].unique())
    overlap = train_sessions.intersection(test_sessions)
    print(f"Session Overlap (Train vs Test): {len(overlap)} overlapping session(s)")

    print("[SUCCESS] Data-only smoke test passed cleanly!\n" + "=" * 70)
    return True


def compute_class_weights(df_train: pd.DataFrame, num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float32)
    for c_id in df_train["class_id"]:
        counts[int(c_id)] += 1.0
    total = np.sum(counts)
    # Inverse frequency weighting
    weights = total / (num_classes * np.maximum(counts, 1.0))
    weights = weights / np.mean(weights)
    return torch.from_numpy(weights.astype(np.float32))


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)

    if not manifest_path.is_file():
        print(f"[ERROR] Manifest file not found: {manifest_path}")
        sys.exit(1)

    # 1. Run Data-Only Smoke Test
    passed_data_test = run_data_smoke_test(manifest_path, args.landmark_group)
    if not passed_data_test:
        sys.exit(1)

    if args.data_smoke_test:
        print("Data smoke test completed. Exiting as requested by --data-smoke-test.")
        sys.exit(0)

    # 2. Build Datasets & Loaders
    num_joints = get_landmark_group_size(args.landmark_group)
    collate_fn = PadCollate()

    ds_train = SignLanguageFeatureDataset(manifest_path, split="train", landmark_group=args.landmark_group)
    ds_val = SignLanguageFeatureDataset(manifest_path, split="val", landmark_group=args.landmark_group)
    ds_test = SignLanguageFeatureDataset(manifest_path, split="test", landmark_group=args.landmark_group)

    num_classes = len(set(ds_train.df["class_id"].unique()).union(set(ds_val.df["class_id"].unique())))
    num_classes = max(num_classes, 10)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 3. Class Weighting
    class_weights = compute_class_weights(ds_train.df, num_classes)
    print("Class Distribution & Inverse Weights:")
    for c_id, w in enumerate(class_weights.tolist()):
        c_count = int(np.sum(ds_train.df["class_id"] == c_id))
        print(f"  Class {c_id:02d}: {c_count:02d} samples | Weight: {w:.3f}")

    # 4. Instantiate Model
    if args.model == "baseline_rnn":
        model = BaselineRNN(num_landmarks=num_joints, num_classes=num_classes)
    elif args.model == "baseline_tcn":
        model = BaselineTCN(num_landmarks=num_joints, num_classes=num_classes)
    elif args.model == "baseline_pooled":
        model = StaticPooledMLP(num_landmarks=num_joints, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # 5. Setup Experiment Directory
    exp_id = args.exp_id or f"{args.model}_{args.landmark_group}_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = PROJECT_ROOT / "models" / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    epochs = 2 if args.smoke_test else args.epochs

    # Save Experiment Config
    config_dict = {
        "model": args.model,
        "landmark_group": args.landmark_group,
        "num_landmarks": num_joints,
        "num_classes": num_classes,
        "batch_size": args.batch_size,
        "epochs": epochs,
        "lr": args.lr,
        "grad_accum": args.grad_accum,
        "device": args.device,
        "mixed_precision": not args.no_amp,
        "manifest": str(manifest_path),
        "smoke_test": args.smoke_test,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(exp_dir / "config.yaml", "w") as f:
        import yaml
        yaml.dump(config_dict, f)

    # 6. Initialize Trainer & Train
    trainer = SignLanguageTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.lr,
        epochs=epochs,
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=not args.no_amp,
        device=args.device,
        class_weights=class_weights,
        early_stopping_patience=5,
        experiment_dir=exp_dir,
        config_dict=config_dict,
    )

    summary = trainer.train()
    print(f"\nExperiment artifacts successfully recorded to {exp_dir}")


if __name__ == "__main__":
    main()
