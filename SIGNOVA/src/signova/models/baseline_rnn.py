"""
Baseline Recurrent Neural Network (BiGRU / BiLSTM) for SIGNOVA.

Processes 543-keypoint skeletal landmark sequences through a 2-stage feature projection,
bidirectional recurrent temporal backbone, masked temporal pooling, and an MLP classifier.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BaselineRNN(nn.Module):
    """
    Baseline A: Bidirectional GRU / LSTM with Masked Temporal Pooling.
    """

    def __init__(
        self,
        num_landmarks: int = 543,
        coordinates_per_landmark: int = 3,
        num_classes: int = 10,
        projection_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        rnn_type: str = "gru",
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.coordinates_per_landmark = coordinates_per_landmark
        self.input_dim = num_landmarks * coordinates_per_landmark
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.rnn_type = rnn_type.lower()

        # 1. 2-Stage Spatial Feature Projection
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

        # 2. Temporal Recurrent Backbone
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=projection_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            self.rnn = nn.GRU(
                input_size=projection_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        rnn_out_dim = hidden_size * self.num_directions

        # 3. Masked Temporal Pooling & Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_out_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, num_landmarks, 3) or (B, T, input_dim).
            padding_mask: Boolean tensor of shape (B, T) where True indicates a valid frame.
            lengths: Long tensor of shape (B,) with unpadded sequence lengths.

        Returns:
            logits: Tensor of shape (B, num_classes).
        """
        B, T = x.shape[0], x.shape[1]
        if x.ndim == 4:
            x = x.reshape(B, T, -1)  # (B, T, input_dim)


        # Project frame-wise
        proj = self.projection(x)  # (B, T, projection_dim)

        # Recurrent processing with packed sequences if lengths provided
        if lengths is not None and not torch.is_grad_enabled():
            # Evaluation / Inference or unpacked training
            rnn_out, _ = self.rnn(proj)
        elif lengths is not None and min(lengths).item() > 0:
            lengths_cpu = lengths.cpu()
            packed = pack_padded_sequence(proj, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.rnn(packed)
            rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        else:
            rnn_out, _ = self.rnn(proj)

        # Masked Temporal Average Pooling
        if padding_mask is not None:
            mask_expanded = padding_mask.unsqueeze(-1).float()     # (B, T, 1)
            sum_rnn = (rnn_out * mask_expanded).sum(dim=1)         # (B, rnn_out_dim)
            eff_lengths = mask_expanded.sum(dim=1).clamp(min=1.0) # (B, 1)
            pooled = sum_rnn / eff_lengths
        else:
            pooled = rnn_out.mean(dim=1)

        logits = self.classifier(pooled)
        return logits
