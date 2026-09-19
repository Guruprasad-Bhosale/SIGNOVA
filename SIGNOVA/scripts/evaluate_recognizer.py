#!/usr/bin/env python3
"""
Evaluation & Diagnostic CLI for SIGNOVA Isolated Sign Recognition Checkpoints.

Evaluates trained model weights on validation and test splits, calculates multi-class
metrics (Accuracy, Balanced Accuracy, Macro F1, Top-1/5), generates confusion matrices,
and exports sample-level error logs for misclassification analysis.
"""

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SIGNOVA sign recognition model checkpoints.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt model checkpoint file.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(PROJECT_ROOT / "data" / "manifests" / "phase3_manifest.csv"),
        help="Path to feature manifest CSV.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Split to evaluate on.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation reports and error CSVs.",
    )
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: Path):
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    model_class = ckpt.get("model_class", "BaselineRNN")
    cfg = ckpt.get("config", {})
    num_landmarks = cfg.get("num_landmarks", 543)
    num_classes = cfg.get("num_classes", 10)

    if model_class == "BaselineRNN":
        model = BaselineRNN(num_landmarks=num_landmarks, num_classes=num_classes)
    elif model_class == "BaselineTCN":
        model = BaselineTCN(num_landmarks=num_landmarks, num_classes=num_classes)
    elif model_class == "StaticPooledMLP":
        model = StaticPooledMLP(num_landmarks=num_landmarks, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model class in checkpoint: {model_class}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def evaluate_split(model, dataset, batch_size=8):
    collate_fn = PadCollate()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_preds = []
    all_targets = []
    all_probs = []
    all_sample_ids = []
    all_session_ids = []
    all_class_names = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"]
            padding_mask = batch["padding_mask"]
            lengths = batch["lengths"]

            logits = model(features, padding_mask=padding_mask, lengths=lengths)
            probs = torch.softmax(logits, dim=-1).numpy()
            preds = np.argmax(probs, axis=-1)

            all_probs.append(probs)
            all_preds.extend(preds.tolist())
            all_targets.extend(batch["labels"].tolist())
            all_sample_ids.extend(batch["sample_ids"])
            all_session_ids.extend(batch["session_ids"])
            all_class_names.extend(batch["class_names"])

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Class labels
    unique_classes = sorted(list(set(all_targets).union(set(all_preds))))

    report = classification_report(all_targets, all_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=unique_classes)

    # Error analysis records
    error_records = []
    for i in range(len(all_sample_ids)):
        true_lbl = all_targets[i]
        pred_lbl = all_preds[i]
        is_correct = (true_lbl == pred_lbl)
        conf = float(all_probs[i, pred_lbl])

        if not is_correct:
            error_records.append({
                "sample_id": all_sample_ids[i],
                "session_id": all_session_ids[i],
                "true_class_id": int(true_lbl),
                "true_class_name": all_class_names[i],
                "predicted_class_id": int(pred_lbl),
                "prediction_confidence": round(conf, 4),
            })

    return {
        "classification_report": report,
        "confusion_matrix": cm,
        "error_records": error_records,
        "total_samples": len(all_sample_ids),
        "correct_samples": int(np.sum(all_preds == all_targets)),
        "accuracy": round(float(np.mean(all_preds == all_targets)), 4),
    }


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.is_file():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    model, cfg = load_model_from_checkpoint(ckpt_path)
    landmark_group = cfg.get("landmark_group", "full")

    manifest_path = Path(args.manifest)
    dataset = SignLanguageFeatureDataset(manifest_path, split=args.split, landmark_group=landmark_group)

    print("=" * 70)
    print(f"EVALUATING MODEL: {model.__class__.__name__} ({landmark_group.upper()})")
    print(f"Checkpoint:       {ckpt_path.name}")
    print(f"Evaluating Split: {args.split.upper()} ({len(dataset)} samples)")
    print("=" * 70)

    res = evaluate_split(model, dataset)

    print(f"\nAccuracy: {res['accuracy'] * 100:.2f}% ({res['correct_samples']}/{res['total_samples']} correct)")
    print(f"Macro F1: {res['classification_report']['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {res['classification_report']['weighted avg']['f1-score']:.4f}")
    print(f"Total Errors: {len(res['error_records'])}")

    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save Classification Report JSON
    with open(out_dir / f"eval_{args.split}_report.json", "w") as f:
        json.dump(res["classification_report"], f, indent=2)

    # Save Confusion Matrix CSV
    df_cm = pd.DataFrame(res["confusion_matrix"])
    df_cm.to_csv(out_dir / f"eval_{args.split}_confusion_matrix.csv", index=False)

    # Save Error Analysis CSV
    if res["error_records"]:
        df_errors = pd.DataFrame(res["error_records"])
        df_errors.to_csv(out_dir / f"eval_{args.split}_error_analysis.csv", index=False)
        print(f"Saved error analysis to {out_dir / f'eval_{args.split}_error_analysis.csv'}")

    print(f"Saved evaluation metrics to {out_dir}")


if __name__ == "__main__":
    main()
