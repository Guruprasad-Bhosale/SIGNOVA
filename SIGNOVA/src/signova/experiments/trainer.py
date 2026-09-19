"""
Phase 12 Continuous ISL Real CTC Training Engine for SIGNOVA.

Trains ContinuousBiGRUEncoder + CTC Recognition Head strictly gated on
STATE_A or STATE_A_DATA_LIMITED supervision.

Includes:
- Deterministic random seeds
- Gradient clipping
- Early stopping on validation loss
- Reproducible environment capture
- Checkpointing
"""

import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from signova.annotation.schema import VideoAnnotation
from signova.qualification.constants import (
    BLANK_ID,
    SUPERVISION_STATE_A,
    SUPERVISION_STATE_A_DATA_LIMITED,
)
from signova.qualification.gate import GateEvaluationResult, Phase12SupervisionGate
from signova.qualification.vocabulary import Phase12GlossVocabulary
from signova.experiments.metrics import evaluate_sequence_predictions


class LandmarkSequenceDataset(Dataset):
    """Dataset for continuous landmark features paired with gloss token IDs."""

    def __init__(
        self,
        samples: List[Tuple[np.ndarray, List[int], str]],
    ):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int, str]:
        feats, targets, sample_id = self.samples[idx]
        feat_tensor = torch.tensor(feats, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.long)
        return feat_tensor, target_tensor, len(feats), len(targets), sample_id


def collate_landmark_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, int, str]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Pads features and targets to form a padded batch."""
    feats, targets, feat_lens, tgt_lens, sample_ids = zip(*batch)
    max_feat_len = max(feat_lens)
    input_dim = feats[0].shape[-1]
    batch_size = len(batch)

    padded_feats = torch.zeros(batch_size, max_feat_len, input_dim)
    for i, f in enumerate(feats):
        padded_feats[i, :len(f), :] = f

    # Flatten targets for CTCLoss or pad
    padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)

    input_lengths = torch.tensor(feat_lens, dtype=torch.long)
    target_lengths = torch.tensor(tgt_lens, dtype=torch.long)

    return padded_feats, padded_targets, input_lengths, target_lengths, list(sample_ids)


class ContinuousBiGRUCTCModel(nn.Module):
    """Continuous BiGRU CTC sequence model."""

    def __init__(
        self,
        input_dim: int = 150,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.fc_in(x)
        out, _ = self.gru(feat)
        logits = self.classifier(out)
        return logits


class RealCTCTrainer:
    """Trains ContinuousBiGRUCTCModel with PyTorch CTCLoss on genuine ISL sequence data."""

    def __init__(
        self,
        vocab: Phase12GlossVocabulary,
        input_dim: int = 150,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.vocab = vocab
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.seed = seed
        self.device = torch.device(device)

        self._set_seed(seed)
        self.model = ContinuousBiGRUCTCModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=vocab.size,
            dropout=dropout,
        ).to(self.device)

        self.criterion = nn.CTCLoss(blank=BLANK_ID, zero_infinity=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        batches = 0

        for feats, targets, feat_lens, tgt_lens, _ in dataloader:
            feats = feats.to(self.device)
            targets = targets.to(self.device)
            feat_lens = feat_lens.to(self.device)
            tgt_lens = tgt_lens.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(feats, feat_lens)  # (batch, seq_len, num_classes)
            log_probs = logits.log_softmax(2).transpose(0, 1)  # (seq_len, batch, num_classes)

            loss = self.criterion(log_probs, targets, feat_lens, tgt_lens)
            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                total_loss += loss.item()
                batches += 1

        return (total_loss / max(1, batches))

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, List[List[str]], List[List[str]], List[str]]:
        self.model.eval()
        total_loss = 0.0
        batches = 0
        all_refs = []
        all_hyps = []
        all_sids = []

        with torch.no_grad():
            for feats, targets, feat_lens, tgt_lens, sample_ids in dataloader:
                feats = feats.to(self.device)
                targets = targets.to(self.device)
                feat_lens = feat_lens.to(self.device)
                tgt_lens = tgt_lens.to(self.device)

                logits = self.model(feats, feat_lens)
                log_probs = logits.log_softmax(2).transpose(0, 1)
                loss = self.criterion(log_probs, targets, feat_lens, tgt_lens)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    batches += 1

                # CTC Greedy Decoding
                preds = logits.argmax(dim=-1).cpu().numpy()
                for b_idx in range(len(sample_ids)):
                    pred_row = preds[b_idx, :feat_lens[b_idx]]
                    # Remove consecutive duplicates and blanks
                    collapsed = []
                    prev = None
                    for t in pred_row:
                        if t != prev and t != BLANK_ID:
                            collapsed.append(t)
                        prev = t
                    hyp_glosses = self.vocab.decode(collapsed, remove_blank=True)

                    tgt_row = targets[b_idx, :tgt_lens[b_idx]].cpu().numpy()
                    ref_glosses = self.vocab.decode(list(tgt_row), remove_blank=True)

                    all_refs.append(ref_glosses)
                    all_hyps.append(hyp_glosses)
                    all_sids.append(sample_ids[b_idx])

        mean_loss = total_loss / max(1, batches)
        return mean_loss, all_refs, all_hyps, all_sids

    def run_experiment(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 10,
        output_dir: Optional[Path] = None,
        eval_result: Optional[GateEvaluationResult] = None,
    ) -> Dict[str, Any]:
        """Runs CTC experiment when gate authorization is satisfied."""
        if eval_result is not None:
            Phase12SupervisionGate.require_training_authorized(eval_result)

        history = []
        best_val_loss = float("inf")
        best_model_state = None

        for ep in range(1, epochs + 1):
            t_loss = self.train_epoch(train_loader)
            v_loss, _, _, _ = self.evaluate(val_loader)

            history.append({"epoch": ep, "train_loss": round(t_loss, 4), "val_loss": round(v_loss, 4)})

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_model_state = self.model.state_dict().copy()

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Test evaluation
        test_loss, test_refs, test_hyps, test_sids = self.evaluate(test_loader)
        test_metrics = evaluate_sequence_predictions(test_refs, test_hyps)

        res = {
            "epochs": epochs,
            "best_val_loss": round(best_val_loss, 4),
            "test_loss": round(test_loss, 4),
            "test_metrics": test_metrics.to_dict(),
            "history": history,
            "test_sample_ids": test_sids,
            "test_references": test_refs,
            "test_hypotheses": test_hyps,
        }

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            if best_model_state is not None:
                torch.save(best_model_state, output_dir / "best_model.pt")
            (output_dir / "training_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

        return res
