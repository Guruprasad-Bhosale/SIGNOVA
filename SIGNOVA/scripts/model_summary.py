#!/usr/bin/env python3
"""
Model Architecture & Parameter Profiling CLI for SIGNOVA.

Calculates trainable parameters, non-trainable parameters, total parameter count,
forward pass latency, throughput, and approximate GPU memory footprint across
isolated and continuous architectures at sequence lengths (32, 64, 128, 256).
"""

import argparse
import json
from pathlib import Path
import sys
import time
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from signova.models.baseline_pooled import StaticPooledMLP
from signova.models.baseline_rnn import BaselineRNN
from signova.models.baseline_tcn import BaselineTCN
from signova.models.boundary_head import TemporalBoundaryHead
from signova.models.continuous_encoder import ContinuousTemporalEncoder
from signova.models.ctc_recognizer import CTCContinuousRecognizer


def parse_args():
    parser = argparse.ArgumentParser(description="Profile parameter count, latency, and memory for SIGNOVA models.")
    parser.add_argument(
        "--model",
        type=str,
        default="continuous_rnn",
        choices=[
            "continuous_rnn",
            "continuous_tcn",
            "ctc_recognizer",
            "boundary_head",
            "baseline_rnn",
            "baseline_tcn",
            "baseline_pooled",
        ],
        help="Model architecture to profile.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classification categories / CTC tokens.",
    )
    parser.add_argument(
        "--num-landmarks",
        type=int,
        default=543,
        help="Number of input landmarks (543=full, 75=hands+pose, 42=hands).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence length for latency benchmark (e.g. 32, 64, 128, 256).",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to export profile JSON.",
    )
    return parser.parse_args()


def profile_model(
    model: nn.Module,
    model_name: str,
    num_landmarks: int,
    num_classes: int,
    test_lengths: list = [32, 64, 128, 256],
) -> dict:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    param_bytes = total_params * 4
    param_mb = param_bytes / (1024 ** 2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    latency_benchmarks = {}

    with torch.no_grad():
        for t in test_lengths:
            b = 4
            if model_name == "boundary_head":
                in_tensor = torch.zeros((b, t, 512), dtype=torch.float32, device=device)
                # Warmup
                for _ in range(5):
                    _ = model(in_tensor)
                t0 = time.time()
                for _ in range(20):
                    out = model(in_tensor)
                elapsed = (time.time() - t0) / 20.0
            elif "continuous" in model_name or model_name == "ctc_recognizer":
                in_tensor = torch.zeros((b, t, num_landmarks, 3), dtype=torch.float32, device=device)
                pad_mask = torch.ones((b, t), dtype=torch.bool, device=device)
                lens = torch.tensor([t] * b, dtype=torch.long, device=device)
                # Warmup
                for _ in range(5):
                    _ = model(in_tensor, padding_mask=pad_mask, lengths=lens)
                t0 = time.time()
                for _ in range(20):
                    out = model(in_tensor, padding_mask=pad_mask, lengths=lens)
                elapsed = (time.time() - t0) / 20.0
            else:
                in_tensor = torch.zeros((b, t, num_landmarks, 3), dtype=torch.float32, device=device)
                pad_mask = torch.ones((b, t), dtype=torch.bool, device=device)
                lens = torch.tensor([t] * b, dtype=torch.long, device=device)
                for _ in range(5):
                    _ = model(in_tensor, padding_mask=pad_mask, lengths=lens)
                t0 = time.time()
                for _ in range(20):
                    out = model(in_tensor, padding_mask=pad_mask, lengths=lens)
                elapsed = (time.time() - t0) / 20.0

            fps = (b * t) / elapsed if elapsed > 0 else 0.0
            out_shape = list(out["logits"].shape) if isinstance(out, dict) and "logits" in out else list(out.shape)

            latency_benchmarks[f"seq_len_{t}"] = {
                "batch_size": b,
                "sequence_length": t,
                "forward_latency_ms": round(elapsed * 1000.0, 3),
                "throughput_frames_per_sec": round(fps, 1),
                "output_shape": out_shape,
            }

    profile = {
        "model_name": model_name,
        "device": device.type.upper(),
        "num_landmarks": num_landmarks,
        "input_dim": num_landmarks * 3 if model_name != "boundary_head" else 512,
        "num_classes": num_classes,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": non_trainable_params,
        "total_parameters": total_params,
        "parameter_memory_mb": round(param_mb, 3),
        "benchmarks": latency_benchmarks,
    }
    return profile


def main():
    args = parse_args()

    if args.model == "continuous_rnn":
        model = ContinuousTemporalEncoder(backbone="gru", num_landmarks=args.num_landmarks)
    elif args.model == "continuous_tcn":
        model = ContinuousTemporalEncoder(backbone="tcn", num_landmarks=args.num_landmarks)
    elif args.model == "ctc_recognizer":
        model = CTCContinuousRecognizer(num_landmarks=args.num_landmarks, num_classes=args.num_classes + 1)
    elif args.model == "boundary_head":
        model = TemporalBoundaryHead(input_dim=512)
    elif args.model == "baseline_rnn":
        model = BaselineRNN(num_landmarks=args.num_landmarks, num_classes=args.num_classes)
    elif args.model == "baseline_tcn":
        model = BaselineTCN(num_landmarks=args.num_landmarks, num_classes=args.num_classes)
    elif args.model == "baseline_pooled":
        model = StaticPooledMLP(num_landmarks=args.num_landmarks, num_classes=args.num_classes)
    else:
        raise ValueError(f"Unknown model {args.model}")

    prof = profile_model(model, args.model, args.num_landmarks, args.num_classes)

    print("=" * 70)
    print(f"SIGNOVA MODEL PROFILE: {args.model.upper()} ({prof['device']})")
    print(f"Input Landmarks:         {prof['num_landmarks']} ({prof['input_dim']} values per frame)")
    print(f"Target Classes:          {prof['num_classes']}")
    print(f"Trainable Parameters:    {prof['trainable_parameters']:,}")
    print(f"Non-Trainable Params:    {prof['non_trainable_parameters']:,}")
    print(f"Total Parameters:        {prof['total_parameters']:,}")
    print(f"Parameter Memory (FP32): {prof['parameter_memory_mb']:.3f} MB")
    print("\nLatency Benchmarks (Batch = 4):")
    for k, v in prof["benchmarks"].items():
        print(
            f"  T={v['sequence_length']:03d} -> Latency: {v['forward_latency_ms']:6.2f} ms | "
            f"Throughput: {v['throughput_frames_per_sec']:7.1f} fps | Out: {v['output_shape']}"
        )
    print("=" * 70)

    if args.json_output:
        out_p = Path(args.json_output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(prof, f, indent=2)
        print(f"Saved summary JSON to {out_p}")


if __name__ == "__main__":
    main()
