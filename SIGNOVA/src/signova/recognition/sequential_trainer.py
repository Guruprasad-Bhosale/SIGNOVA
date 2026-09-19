"""
Sequential CTC Training Engine for SIGNOVA.

Provides training, validation, greedy decoding, and sequence error rate evaluation
for Connectionist Temporal Classification (CTC) sequence recognizers.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from signova.evaluation.sequence_metrics import compute_sequence_metrics
from signova.models.ctc_recognizer import CTCContinuousRecognizer

logger = logging.getLogger("signova.sequential_trainer")


class SequentialCTCTrainer:
    """
    Standardized trainer for CTC-based continuous sign recognizers.
    """

    def __init__(
        self,
        model: CTCContinuousRecognizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        learning_rate: float = 0.0003,
        weight_decay: float = 0.0001,
        epochs: int = 15,
        gradient_accumulation_steps: int = 4,
        mixed_precision: bool = True,
        device: str = "auto",
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 0.005,
        early_stopping_monitor: str = "val_ter",  # lower is better
        experiment_dir: Optional[Union[str, Path]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_monitor = early_stopping_monitor
        self.config_dict = config_dict or {}

        # 1. Device Resolution
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available.")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

        # 2. Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
        )

        # 3. Mixed Precision
        self.use_amp = mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # 4. Experiment Directory
        if experiment_dir:
            self.exp_dir = Path(experiment_dir)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.exp_dir = Path("models") / "experiments" / f"ctc_run_{timestamp}"

        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            features = batch["features"].to(self.device)
            padding_mask = batch["padding_mask"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            targets = batch["targets"].to(self.device) if batch["targets"] is not None else None
            target_lengths = batch["target_lengths"].to(self.device) if batch["target_lengths"] is not None else None

            if targets is None or target_lengths is None:
                raise ValueError("Batch has no target sequences for CTC training.")

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    out = self.model(
                        features,
                        padding_mask=padding_mask,
                        lengths=lengths,
                        targets=targets,
                        target_lengths=target_lengths,
                    )
                    loss = out["loss"] / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                out = self.model(
                    features,
                    padding_mask=padding_mask,
                    lengths=lengths,
                    targets=targets,
                    target_lengths=target_lengths,
                )
                loss = out["loss"] / self.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps

        return total_loss / max(1, len(self.train_loader))

    def evaluate_loader(self, loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        all_refs = []
        all_hyps = []

        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                lengths = batch["lengths"].to(self.device)
                targets = batch["targets"].to(self.device) if batch["targets"] is not None else None
                target_lengths = batch["target_lengths"].to(self.device) if batch["target_lengths"] is not None else None

                if targets is not None and target_lengths is not None:
                    out = self.model(
                        features,
                        padding_mask=padding_mask,
                        lengths=lengths,
                        targets=targets,
                        target_lengths=target_lengths,
                    )
                    if out["loss"] is not None:
                        total_loss += out["loss"].item()

                    # Extract ground truth target lists
                    for b in range(len(target_lengths)):
                        t_len = target_lengths[b].item()
                        ref_toks = targets[b, :t_len].cpu().tolist()
                        all_refs.append(ref_toks)

                decoded = self.model.decode_greedy(features, padding_mask=padding_mask, lengths=lengths)
                for d in decoded:
                    all_hyps.append(d["collapsed_tokens"])

        metrics = compute_sequence_metrics(all_refs, all_hyps) if all_refs else {}
        metrics["loss"] = round(total_loss / max(1, len(loader)), 4)
        return metrics

    def train(self) -> Dict[str, Any]:
        best_ter = float("inf")
        best_epoch = -1
        patience_counter = 0
        history = []
        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            t_ep_start = time.time()
            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate_loader(self.val_loader)
            ep_time = time.time() - t_ep_start

            val_ter = val_metrics.get("token_error_rate", float("inf"))
            self.scheduler.step(val_ter)

            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": val_metrics.get("loss", 0.0),
                "val_ter": val_ter,
                "val_exact_match": val_metrics.get("exact_match_rate", 0.0),
                "epoch_time_sec": round(ep_time, 2),
            }
            history.append(record)

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {record['val_loss']:.4f} | "
                f"Val TER: {val_ter:.4f} | "
                f"Val Exact: {record['val_exact_match'] * 100:.1f}% ({ep_time:.1f}s)"
            )

            # Checkpoint: Save Best
            if val_ter < (best_ter - self.early_stopping_min_delta):
                best_ter = val_ter
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(self.exp_dir / "best.pt", epoch=epoch, val_metrics=val_metrics)
            else:
                patience_counter += 1

            self.save_checkpoint(self.exp_dir / "last.pt", epoch=epoch, val_metrics=val_metrics)

            if patience_counter >= self.early_stopping_patience:
                print(f"[Early Stopping] No improvement in TER for {self.early_stopping_patience} epochs.")
                break

        total_time = round(time.time() - t0, 2)
        summary = {
            "experiment_dir": str(self.exp_dir),
            "best_epoch": best_epoch,
            "best_val_ter": round(best_ter, 4),
            "total_train_time_sec": total_time,
            "training_history": history,
        }

        with open(self.exp_dir / "metrics.json", "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def save_checkpoint(self, path: Path, epoch: int, val_metrics: Dict[str, Any]):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": self.config_dict,
                "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            str(path),
        )
