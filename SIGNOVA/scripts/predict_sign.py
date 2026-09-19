#!/usr/bin/env python3
"""
Offline Sign Recognition Inference CLI for SIGNOVA.

Loads a trained model checkpoint and performs single-sample inference on an extracted
.npz skeletal landmark sequence, outputting predicted class, confidence, and Top-5 distributions.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signova.features.feature_groups import LandmarkGroup, slice_landmark_tensor
from signova.features.storage import load_landmark_features
from signova.models.baseline_pooled import StaticPooledMLP
from signova.models.baseline_rnn import BaselineRNN
from signova.models.baseline_tcn import BaselineTCN


def parse_args():
    parser = argparse.ArgumentParser(description="Predict sign class from extracted .npz feature file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model .pt checkpoint.",
    )
    parser.add_argument(
        "--feature",
        type=str,
        required=True,
        help="Path to .npz feature file.",
    )
    parser.add_argument(
        "--class-manifest",
        type=str,
        default=str(PROJECT_ROOT / "data" / "manifests" / "phase3_class_manifest.csv"),
        help="Path to class manifest CSV for label mapping.",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path):
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    model_class = ckpt.get("model_class", "BaselineRNN")
    cfg = ckpt.get("config", {})
    num_landmarks = cfg.get("num_landmarks", 543)
    num_classes = cfg.get("num_classes", 10)
    landmark_group = cfg.get("landmark_group", "full")

    if model_class == "BaselineRNN":
        model = BaselineRNN(num_landmarks=num_landmarks, num_classes=num_classes)
    elif model_class == "BaselineTCN":
        model = BaselineTCN(num_landmarks=num_landmarks, num_classes=num_classes)
    elif model_class == "StaticPooledMLP":
        model = StaticPooledMLP(num_landmarks=num_landmarks, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, landmark_group, num_classes


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    feat_path = Path(args.feature)

    if not ckpt_path.is_file():
        print(f"[ERROR] Checkpoint file not found: {ckpt_path}")
        sys.exit(1)
    if not feat_path.is_file():
        print(f"[ERROR] Feature file not found: {feat_path}")
        sys.exit(1)

    # 1. Load Class Mapping
    class_map = {}
    class_man_path = Path(args.class_manifest)
    if class_man_path.is_file():
        df_c = pd.read_csv(class_man_path)
        for _, row in df_c.iterrows():
            class_map[int(row["class_id"])] = str(row["class_name"])

    # 2. Load Model & Feature
    model, landmark_group, num_classes = load_model(ckpt_path)
    raw_lm, det_mask, timestamps, frame_indices, meta = load_landmark_features(feat_path)

    # 3. Prepare Input Tensor
    sliced_lm = slice_landmark_tensor(raw_lm, group_name=landmark_group)
    T = sliced_lm.shape[0]

    input_tensor = torch.from_numpy(sliced_lm.astype(np.float32)).unsqueeze(0) # (1, T, num_joints, 3)
    padding_mask = torch.ones((1, T), dtype=torch.bool)
    lengths = torch.tensor([T], dtype=torch.long)

    # 4. Forward Inference
    with torch.no_grad():
        logits = model(input_tensor, padding_mask=padding_mask, lengths=lengths)
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

    pred_id = int(np.argmax(probs))
    pred_conf = float(probs[pred_id])
    pred_name = class_map.get(pred_id, f"CLASS_{pred_id}")

    # Top-5
    top5_indices = np.argsort(probs)[::-1][:min(5, num_classes)]
    top5_list = [(class_map.get(i, f"CLASS_{i}"), round(float(probs[i]), 4)) for i in top5_indices]

    print("=" * 60)
    print("SIGNOVA OFFLINE SIGN PREDICTION")
    print(f"Sample:           {feat_path.stem} ({T} frames)")
    print(f"Predicted class:  {pred_name} (ID: {pred_id})")
    print(f"Confidence:       {pred_conf * 100:.2f}%")
    print(f"Top-5:            {top5_list}")
    print("=" * 60)


if __name__ == "__main__":
    main()
