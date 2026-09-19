"""
Continuous Sign Recognizer with CTC Architecture for SIGNOVA.

Combines continuous temporal encoders with Connectionist Temporal Classification (CTC)
for sequential sign recognition when aligned gloss sequence targets are available.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from signova.models.continuous_encoder import ContinuousTemporalEncoder


class CTCContinuousRecognizer(nn.Module):
    """
    End-to-End Continuous Sign Language Recognizer with CTC Loss & Greedy Decoding.
    """

    def __init__(
        self,
        num_landmarks: int = 543,
        num_classes: int = 11,  # Includes blank token at index 0
        backbone: str = "gru",
        projection_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        blank_idx: int = 0,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_classes = num_classes
        self.blank_idx = blank_idx

        # 1. Continuous Temporal Backbone
        self.encoder = ContinuousTemporalEncoder(
            backbone=backbone,
            num_landmarks=num_landmarks,
            projection_dim=projection_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )

        # 2. CTC Logit Projection Head
        self.ctc_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.output_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

        # 3. Native PyTorch CTC Loss
        self.ctc_loss_fn = nn.CTCLoss(
            blank=blank_idx,
            reduction="mean",
            zero_infinity=True,
        )

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Args:
            features: Tensor of shape (B, T, num_landmarks, 3) or (B, T, input_dim).
            padding_mask: Boolean tensor (B, T).
            lengths: Long tensor (B,) with input sequence frame counts.
            targets: Long tensor (B, max_target_len) or 1D concatenated targets.
            target_lengths: Long tensor (B,) with target sequence lengths.

        Returns:
            Dict containing 'logits', 'log_probs', and optional 'loss'.
        """
        B, T = features.shape[0], features.shape[1]

        # 1. Extract per-frame representations
        embeddings = self.encoder(features, padding_mask=padding_mask, lengths=lengths) # (B, T, D)

        # 2. Project to CTC logits
        logits = self.ctc_head(embeddings)  # (B, T, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, num_classes)

        result = {
            "embeddings": embeddings,
            "logits": logits,
            "log_probs": log_probs,
            "loss": None,
        }

        # 3. Compute CTC Loss if targets are provided
        if targets is not None and target_lengths is not None:
            input_lens = lengths if lengths is not None else torch.tensor([T] * B, device=features.device, dtype=torch.long)

            # Strict CTC validation: input length must be >= target length
            if torch.any(input_lens < target_lengths):
                violating = torch.where(input_lens < target_lengths)[0]
                raise ValueError(
                    f"CTC constraint violation: Input sequence length {input_lens[violating].tolist()} "
                    f"is shorter than target length {target_lengths[violating].tolist()} for samples {violating.tolist()}."
                )

            # PyTorch CTCLoss expects (T, B, C)
            ctc_input = log_probs.transpose(0, 1)  # (T, B, C)
            loss = self.ctc_loss_fn(ctc_input, targets, input_lens, target_lengths)
            result["loss"] = loss

        return result

    def decode_greedy(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """
        Greedy CTC decoding for batched inputs.
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(features, padding_mask=padding_mask, lengths=lengths)
            log_probs = out["log_probs"] # (B, T, C)
            probs = torch.softmax(out["logits"], dim=-1)
            best_tokens = torch.argmax(log_probs, dim=-1) # (B, T)

        B, T = features.shape[0], features.shape[1]
        decoded_batches = []

        for b in range(B):
            seq_len = lengths[b].item() if lengths is not None else T
            raw_seq = best_tokens[b, :seq_len].tolist()
            prob_seq = probs[b, :seq_len].tolist()

            collapsed = []
            confidences = []
            prev_token = None

            for t, tok in enumerate(raw_seq):
                if tok != self.blank_idx:
                    if tok != prev_token:
                        collapsed.append(tok)
                        confidences.append(prob_seq[t][tok])
                prev_token = tok

            decoded_batches.append({
                "raw_tokens": raw_seq,
                "collapsed_tokens": collapsed,
                "token_confidences": [round(c, 4) for c in confidences],
                "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
                "sequence_length": seq_len,
            })

        return decoded_batches
