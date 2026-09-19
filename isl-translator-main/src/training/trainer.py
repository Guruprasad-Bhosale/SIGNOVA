"""
Training Loop for ISL Translator.

Handles training, validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from pathlib import Path
from typing import Dict, Optional
import logging
from tqdm import tqdm

from .metrics import MetricsTracker
from ..models.isl_model import ISLTranslator

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_improvement(self, score: float) -> bool:
        assert self.best_score is not None
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta


class Trainer:
    """
    Trainer for ISL Translator model.

    Supports:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Learning rate scheduling with linear warmup
    - Checkpointing
    - Early stopping
    - TensorBoard logging
    - Fine-tuning from pre-trained encoder weights
    """

    def __init__(
        self,
        model: ISLTranslator,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cuda",
        output_dir: str = "./checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Training config
        train_config = config.get("training", {})
        self.epochs = train_config.get("epochs", 100)
        self.use_amp = train_config.get("use_amp", True) and device == "cuda"
        self.grad_accum = train_config.get("gradient_accumulation", 1)
        self.grad_clip = train_config.get("gradient_clip", 5.0)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=train_config.get("learning_rate", 1e-4),
            weight_decay=train_config.get("weight_decay", 0.01),
        )

        # Scheduler
        scheduler_type = train_config.get("scheduler", "cosine")
        warmup_epochs = train_config.get("warmup_epochs", 5)

        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max(self.epochs // 4, 1),
                T_mult=2,
                eta_min=train_config.get("min_lr", 1e-6),
            )
        else:
            steps_per_epoch = max(len(train_loader) // self.grad_accum, 1)
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=train_config.get("learning_rate", 1e-4),
                steps_per_epoch=steps_per_epoch,
                epochs=self.epochs,
                pct_start=min(warmup_epochs / max(self.epochs, 1), 0.3),
            )

        # AMP scaler
        self.scaler: Optional[GradScaler] = GradScaler() if self.use_amp else None

        # Early stopping
        es_config = train_config.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_config.get("patience", 15),
            min_delta=es_config.get("min_delta", 0.001),
            mode="min",
        )

        # Tracking
        self.best_wer = float("inf")
        self.current_epoch = 0

        # TensorBoard
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))
        except ImportError:
            logger.warning("TensorBoard not available")

    def train_epoch(self) -> Dict:
        """Train for one epoch."""
        self.model.train()
        tracker = MetricsTracker()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        self.optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            # Move to device
            keypoints = batch["keypoints"].to(self.device)
            lengths = batch["seq_lengths"].to(self.device)
            glosses = batch["glosses"].to(self.device)
            gloss_lengths = batch["gloss_lengths"].to(self.device)

            # Forward pass
            with autocast(enabled=self.use_amp):
                output = self.model(keypoints, lengths, glosses, gloss_lengths)
                loss = output["loss"] / self.grad_accum

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation step
            if (step + 1) % self.grad_accum == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()

            # Track metrics
            tracker.update(loss=loss.item() * self.grad_accum)
            pbar.set_postfix({"loss": f"{loss.item() * self.grad_accum:.4f}"})

        if not isinstance(self.scheduler, OneCycleLR):
            self.scheduler.step()

        return tracker.compute()

    @torch.no_grad()
    def validate(self) -> Dict:
        """Run validation."""
        self.model.eval()
        tracker = MetricsTracker()

        for batch in tqdm(self.val_loader, desc="Validating"):
            keypoints = batch["keypoints"].to(self.device)
            lengths = batch["seq_lengths"].to(self.device)
            glosses = batch["glosses"].to(self.device)
            gloss_lengths = batch["gloss_lengths"].to(self.device)

            # Forward pass
            with autocast(enabled=self.use_amp):
                output = self.model(keypoints, lengths, glosses, gloss_lengths)

            # Decode predictions
            predictions = self.model.decode(keypoints, lengths, beam_width=5)

            # Get reference glosses
            references = []
            for i in range(len(glosses)):
                ref_len = gloss_lengths[i].item()
                ref_indices = glosses[i, :ref_len].tolist()
                ref_glosses = [self.model.idx_to_gloss.get(idx, "<unk>") for idx in ref_indices]
                ref_glosses = [g for g in ref_glosses if g not in ["<blank>", "<sos>", "<eos>", "<unk>"]]
                references.append(ref_glosses)

            tracker.update(
                loss=output["loss"].item(),
                predictions=predictions,
                references=references,
            )

        return tracker.compute()

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_wer": self.best_wer,
            "vocab": self.model.vocab,
            "encoder_type": self.model.encoder_type,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save latest
        path = self.output_dir / "latest.pt"
        torch.save(checkpoint, path)

        # Save best
        if is_best:
            best_path = self.output_dir / "best.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with WER: {self.best_wer:.2%}")

    def load_checkpoint(self, path: str):
        """Load full checkpoint for resuming training."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_wer = checkpoint.get("best_wer", float("inf"))

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Resumed from checkpoint at epoch {self.current_epoch}")

    def load_pretrained_encoder(self, path: str):
        """Load pre-trained encoder weights only (skip CTC classifier head).

        Used for transfer learning: encoder features transfer to a new
        vocabulary while the CTC head is randomly initialized.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model_state = self.model.state_dict()
        pretrained_state = checkpoint["model_state_dict"]

        # Only keep encoder weights that match current model shape
        transferred, skipped = 0, 0
        for name, param in pretrained_state.items():
            if "ctc" in name:
                skipped += 1
                continue
            if name in model_state and param.shape == model_state[name].shape:
                model_state[name] = param
                transferred += 1
            else:
                skipped += 1

        self.model.load_state_dict(model_state)
        logger.info(f"Loaded pre-trained encoder: {transferred} tensors transferred, {skipped} skipped")

    def train(self, resume_from: Optional[str] = None, finetune_from: Optional[str] = None) -> float:
        """Main training loop.

        Args:
            resume_from: Path to checkpoint for full resumption.
            finetune_from: Path to pre-trained model for transfer learning.

        Returns:
            Best validation WER achieved during training.
        """
        if resume_from:
            self.load_checkpoint(resume_from)
        elif finetune_from:
            self.load_pretrained_encoder(finetune_from)

        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Model parameters: {self.model.get_num_parameters():,}")

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Epoch {epoch} Train - Loss: {train_metrics.get('loss', 0):.4f}")

            # Validate
            val_metrics = self.validate()
            val_wer = val_metrics.get("wer", 1.0)
            logger.info(f"Epoch {epoch} Val - Loss: {val_metrics.get('loss', 0):.4f}, WER: {val_wer:.2%}")

            # Log to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar("train/loss", train_metrics.get("loss", 0), epoch)
                self.writer.add_scalar("val/loss", val_metrics.get("loss", 0), epoch)
                self.writer.add_scalar("val/wer", val_wer, epoch)
                self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)

            # Check if best model
            is_best = val_wer < self.best_wer
            if is_best:
                self.best_wer = val_wer

            # Save checkpoint
            self.save_checkpoint(is_best=is_best)

            # Early stopping
            if self.early_stopping(val_wer):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        logger.info(f"Training complete. Best WER: {self.best_wer:.2%}")

        if self.writer is not None:
            self.writer.close()

        return self.best_wer
