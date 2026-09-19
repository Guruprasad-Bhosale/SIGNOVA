"""
Neural Training Engine for SIGNOVA Isolated / Dynamic Sign Recognition.

Provides training, validation, multi-metric evaluation, early stopping,
checkpointing, mixed precision, and RTX 3050 memory-safe optimizations.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger("signova.trainer")


class SignLanguageTrainer:
    """
    Standardized trainer for SIGNOVA temporal landmark classifiers.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        learning_rate: float = 0.0003,
        weight_decay: float = 0.0001,
        epochs: int = 20,
        gradient_accumulation_steps: int = 4,
        mixed_precision: bool = True,
        device: str = "auto",
        class_weights: Optional[torch.Tensor] = None,
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 0.001,
        early_stopping_monitor: str = "val_macro_f1",
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

        # 2. Loss & Class Weighting
        if class_weights is not None:
            self.class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.class_weights = None
            self.criterion = nn.CrossEntropyLoss()

        # 3. Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max" if "f1" in early_stopping_monitor or "acc" in early_stopping_monitor else "min",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
        )

        # 4. Mixed Precision
        self.use_amp = mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # 5. Experiment Directory
        if experiment_dir:
            self.exp_dir = Path(experiment_dir)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_name = getattr(model, "__class__", type(model)).__name__
            self.exp_dir = Path("models") / "experiments" / f"{model_name}_{timestamp}"

        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        log_file = self.exp_dir / "training.log"
        self._log_handler = logging.FileHandler(str(log_file))
        self._log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(self._log_handler)
        logger.setLevel(logging.INFO)

    def close(self):
        if hasattr(self, "_log_handler") and self._log_handler is not None:
            self._log_handler.close()
            logger.removeHandler(self._log_handler)
            self._log_handler = None


    def print_hardware_summary(self):
        print("\n" + "=" * 70)
        print("SIGNOVA TRAINING HARDWARE & RUNTIME PROFILE")
        print(f"Device:               {self.device.type.upper()}")
        if self.device.type == "cuda":
            print(f"GPU Name:             {torch.cuda.get_device_name(0)}")
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"Total VRAM:           {total_vram_gb:.2f} GB")
        else:
            print("GPU Name:             N/A (CPU Fallback Execution)")
        print(f"Mixed Precision:      {'ENABLED (FP16/AMP)' if self.use_amp else 'DISABLED (FP32)'}")
        print(f"Batch Size:           {self.train_loader.batch_size}")
        print(f"Grad Accum Steps:     {self.gradient_accumulation_steps}")
        print(f"Effective Batch:      {self.train_loader.batch_size * self.gradient_accumulation_steps}")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable Params:     {trainable_params:,}")
        print(f"Total Params:         {total_params:,}")
        print("=" * 70 + "\n")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            features = batch["features"].to(self.device)
            padding_mask = batch["padding_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            lengths = batch["lengths"].to(self.device)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(features, padding_mask=padding_mask, lengths=lengths)
                    loss = self.criterion(logits, labels)
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits = self.model(features, padding_mask=padding_mask, lengths=lengths)
                loss = self.criterion(logits, labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps

        return total_loss / len(self.train_loader)

    def evaluate_loader(self, loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []
        all_sample_ids = []
        all_session_ids = []

        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self.device)
                padding_mask = batch["padding_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                lengths = batch["lengths"].to(self.device)

                logits = self.model(features, padding_mask=padding_mask, lengths=lengths)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
                all_sample_ids.extend(batch["sample_ids"])
                all_session_ids.extend(batch["session_ids"])

        all_probs = np.concatenate(all_probs, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        num_classes = all_probs.shape[1]
        acc = float(accuracy_score(all_targets, all_preds))
        bal_acc = float(balanced_accuracy_score(all_targets, all_preds))
        macro_prec = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
        macro_rec = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))
        macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))

        top5_acc = None
        if num_classes >= 5:
            try:
                top5_acc = float(top_k_accuracy_score(all_targets, all_probs, k=min(5, num_classes), labels=list(range(num_classes))))
            except Exception:
                pass

        cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))

        return {
            "loss": round(total_loss / len(loader), 4),
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "top5_accuracy": round(top5_acc, 4) if top5_acc is not None else None,
            "confusion_matrix": cm.tolist(),
            "predictions": all_preds.tolist(),
            "targets": all_targets.tolist(),
            "probabilities": all_probs.tolist(),
            "sample_ids": all_sample_ids,
            "session_ids": all_session_ids,
        }

    def train(self) -> Dict[str, Any]:
        self.print_hardware_summary()

        best_metric = -float("inf") if "f1" in self.early_stopping_monitor or "acc" in self.early_stopping_monitor else float("inf")
        best_epoch = -1
        patience_counter = 0
        history = []
        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            t_ep_start = time.time()
            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate_loader(self.val_loader)
            ep_time = time.time() - t_ep_start

            current_val_metric = val_metrics.get(self.early_stopping_monitor.replace("val_", ""), val_metrics["macro_f1"])
            self.scheduler.step(current_val_metric)

            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "epoch_time_sec": round(ep_time, 2),
                "lr": float(self.optimizer.param_groups[0]["lr"]),
            }
            history.append(record)

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['macro_f1']:.4f} "
                f"({ep_time:.1f}s)"
            )

            # Checkpoint: Save Last
            self.save_checkpoint(self.exp_dir / "last.pt", epoch=epoch, val_metrics=val_metrics)

            # Checkpoint: Save Best
            is_better = (current_val_metric > (best_metric + self.early_stopping_min_delta)) if "f1" in self.early_stopping_monitor or "acc" in self.early_stopping_monitor else (current_val_metric < (best_metric - self.early_stopping_min_delta))
            if is_better:
                best_metric = current_val_metric
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(self.exp_dir / "best.pt", epoch=epoch, val_metrics=val_metrics)
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                print(f"\n[Early Stopping Triggered] No improvement in '{self.early_stopping_monitor}' for {self.early_stopping_patience} epochs.")
                break

        total_train_time = round(time.time() - t0, 2)
        print(f"\nTraining completed in {total_train_time}s. Best Epoch: {best_epoch} with {self.early_stopping_monitor} = {best_metric:.4f}")

        # Final Evaluation on Test Split
        test_metrics = None
        if self.test_loader is not None:
            # Load best checkpoint for test evaluation
            best_ckpt_path = self.exp_dir / "best.pt"
            if best_ckpt_path.exists():
                self.load_checkpoint(best_ckpt_path)
            test_metrics = self.evaluate_loader(self.test_loader)
            print(f"Test Split Evaluation -> Accuracy: {test_metrics['accuracy']:.4f} | Macro F1: {test_metrics['macro_f1']:.4f}")

        # Save Metrics & Logs
        summary = {
            "experiment_dir": str(self.exp_dir),
            "best_epoch": best_epoch,
            "best_val_metric": round(best_metric, 4),
            "total_train_time_sec": total_train_time,
            "training_history": history,
            "test_metrics": {k: v for k, v in test_metrics.items() if k not in ("predictions", "probabilities", "sample_ids", "session_ids", "targets", "confusion_matrix")} if test_metrics else None,
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
                "model_class": getattr(self.model, "__class__", type(self.model)).__name__,
                "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            str(path),
        )

    def load_checkpoint(self, path: Path):
        checkpoint = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
