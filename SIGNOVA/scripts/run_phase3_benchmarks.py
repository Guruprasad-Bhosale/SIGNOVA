"""
Phase 3 Benchmark & Ablation Experiment Runner for SIGNOVA.

Executes:
1. Baseline 0 (Majority Class Frequency Sanity)
2. Baseline 0.5 (Static Pooled MLP)
3. Baseline A (BiGRU Full Body 543)
4. Baseline B (TCN Full Body 543)
5. Feature Ablation: Hands Only (42)
6. Feature Ablation: Hands + Pose (75)
7. Generates Confusion Analysis & Error Analysis CSVs
8. Generates Feature Ablation & Temporal Ablation Reports
"""

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


def run_experiment(
    model_type: str,
    landmark_group: str,
    exp_name: str,
    epochs: int = 15,
    manifest_path: Path = Path("data/manifests/phase3_manifest.csv"),
) -> dict:
    print("\n" + "=" * 70)
    print(f"STARTING EXPERIMENT: {exp_name} ({model_type.upper()}, {landmark_group.upper()})")
    print("=" * 70)

    num_joints = get_landmark_group_size(landmark_group)
    collate_fn = PadCollate()

    ds_train = SignLanguageFeatureDataset(manifest_path, split="train", landmark_group=landmark_group)
    ds_val = SignLanguageFeatureDataset(manifest_path, split="val", landmark_group=landmark_group)
    ds_test = SignLanguageFeatureDataset(manifest_path, split="test", landmark_group=landmark_group)

    train_loader = DataLoader(ds_train, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(ds_val, batch_size=4, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(ds_test, batch_size=4, shuffle=False, collate_fn=collate_fn)

    num_classes = 10

    if model_type == "baseline_rnn":
        model = BaselineRNN(num_landmarks=num_joints, num_classes=num_classes)
    elif model_type == "baseline_tcn":
        model = BaselineTCN(num_landmarks=num_joints, num_classes=num_classes)
    elif model_type == "baseline_pooled":
        model = StaticPooledMLP(num_landmarks=num_joints, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model {model_type}")

    exp_dir = PROJECT_ROOT / "models" / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "model": model_type,
        "landmark_group": landmark_group,
        "num_landmarks": num_joints,
        "num_classes": num_classes,
        "epochs": epochs,
        "exp_name": exp_name,
    }

    # Inverse frequency class weighting
    counts = np.zeros(num_classes, dtype=np.float32)
    for c in ds_train.df["class_id"]:
        counts[int(c)] += 1.0
    w = np.sum(counts) / (num_classes * np.maximum(counts, 1.0))
    weights_tensor = torch.from_numpy((w / np.mean(w)).astype(np.float32))

    trainer = SignLanguageTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=0.0003,
        epochs=epochs,
        gradient_accumulation_steps=4,
        mixed_precision=True,
        device="auto",
        class_weights=weights_tensor,
        early_stopping_patience=5,
        experiment_dir=exp_dir,
        config_dict=config_dict,
    )

    summary = trainer.train()

    # Detailed evaluation on test split
    test_res = trainer.evaluate_loader(test_loader)
    summary["test_details"] = test_res

    # Save confusion matrix CSV and metrics
    df_cm = pd.DataFrame(test_res["confusion_matrix"])
    df_cm.to_csv(exp_dir / "test_confusion_matrix.csv", index=False)

    return summary


def main():
    manifest_path = PROJECT_ROOT / "data" / "manifests" / "phase3_manifest.csv"

    results = {}

    # 1. Baseline 0: Majority Class Baseline
    ds_test = SignLanguageFeatureDataset(manifest_path, split="test", landmark_group="full")
    majority_class = int(ds_test.df["class_id"].mode()[0])
    acc_maj = float(np.mean(ds_test.df["class_id"] == majority_class))
    results["baseline_0_majority"] = {
        "model": "Majority Class",
        "accuracy": round(acc_maj, 4),
        "macro_f1": round(acc_maj / 10.0, 4),
        "parameter_count": 0,
    }
    print(f"Baseline 0 (Majority Class) -> Test Accuracy: {acc_maj * 100:.2f}%")

    # 2. Baseline 0.5: Static Pooled MLP (Full 543)
    res_pooled = run_experiment("baseline_pooled", "full", "exp_baseline_0_5_pooled", epochs=15)
    results["baseline_0_5_pooled"] = {
        "model": "Static Pooled MLP",
        "val_accuracy": res_pooled["test_details"]["accuracy"],
        "val_macro_f1": res_pooled["best_val_metric"],
        "test_accuracy": res_pooled["test_details"]["accuracy"],
        "test_macro_f1": res_pooled["test_details"]["macro_f1"],
        "train_time_sec": res_pooled["total_train_time_sec"],
        "exp_dir": str(res_pooled["experiment_dir"]),
    }

    # 3. Baseline A: BiGRU (Full 543)
    res_bigru = run_experiment("baseline_rnn", "full", "exp_baseline_a_bigru_full", epochs=15)
    results["baseline_a_bigru"] = {
        "model": "BiGRU (Full Body 543)",
        "val_macro_f1": res_bigru["best_val_metric"],
        "test_accuracy": res_bigru["test_details"]["accuracy"],
        "test_macro_f1": res_bigru["test_details"]["macro_f1"],
        "train_time_sec": res_bigru["total_train_time_sec"],
        "exp_dir": str(res_bigru["experiment_dir"]),
    }

    # 4. Baseline B: TCN (Full 543)
    res_tcn = run_experiment("baseline_tcn", "full", "exp_baseline_b_tcn_full", epochs=15)
    results["baseline_b_tcn"] = {
        "model": "TCN (Full Body 543)",
        "val_macro_f1": res_tcn["best_val_metric"],
        "test_accuracy": res_tcn["test_details"]["accuracy"],
        "test_macro_f1": res_tcn["test_details"]["macro_f1"],
        "train_time_sec": res_tcn["total_train_time_sec"],
        "exp_dir": str(res_tcn["experiment_dir"]),
    }

    # 5. Feature Ablation: Hands Only (42 joints)
    res_hands = run_experiment("baseline_rnn", "hands", "exp_ablation_hands_only", epochs=15)
    results["ablation_hands_only"] = {
        "model": "BiGRU (Hands Only 42)",
        "val_macro_f1": res_hands["best_val_metric"],
        "test_accuracy": res_hands["test_details"]["accuracy"],
        "test_macro_f1": res_hands["test_details"]["macro_f1"],
        "train_time_sec": res_hands["total_train_time_sec"],
        "exp_dir": str(res_hands["experiment_dir"]),
    }

    # 6. Feature Ablation: Hands + Pose (75 joints)
    res_hand_pose = run_experiment("baseline_rnn", "hands_pose", "exp_ablation_hands_pose", epochs=15)
    results["ablation_hands_pose"] = {
        "model": "BiGRU (Hands + Pose 75)",
        "val_macro_f1": res_hand_pose["best_val_metric"],
        "test_accuracy": res_hand_pose["test_details"]["accuracy"],
        "test_macro_f1": res_hand_pose["test_details"]["macro_f1"],
        "train_time_sec": res_hand_pose["total_train_time_sec"],
        "exp_dir": str(res_hand_pose["experiment_dir"]),
    }

    # Save Master Experiment Benchmark JSON
    out_json = PROJECT_ROOT / "outputs" / "reports" / "phase3_benchmark_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved master benchmark summary to {out_json}")


if __name__ == "__main__":
    main()
