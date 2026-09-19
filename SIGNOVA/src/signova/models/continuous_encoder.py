"""
Continuous Temporal Encoders for SIGNOVA.

Preserves per-frame temporal representations (B, T, D) across sequence lengths,
supporting BiGRU and Dilated TCN backbones with seamless Phase 3 weight transfer.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from signova.models.baseline_tcn import TemporalResidualBlock


class ContinuousBiGRUEncoder(nn.Module):
    """
    Continuous Sequence-to-Sequence Recurrent Temporal Backbone.
    Maps (B, T, input_dim) -> (B, T, hidden_size * num_directions).
    """

    def __init__(
        self,
        num_landmarks: int = 543,
        coordinates_per_landmark: int = 3,
        projection_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.coordinates_per_landmark = coordinates_per_landmark
        self.input_dim = num_landmarks * coordinates_per_landmark
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.output_dim = hidden_size * self.num_directions

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

        # 2. Recurrent Temporal Backbone
        self.rnn = nn.GRU(
            input_size=projection_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
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
            padding_mask: Boolean tensor (B, T) where True indicates valid frame.
            lengths: Long tensor (B,) with unpadded sequence lengths.

        Returns:
            Per-frame temporal embeddings of shape (B, T, output_dim).
        """
        B, T = x.shape[0], x.shape[1]
        if x.ndim == 4:
            x = x.reshape(B, T, -1)

        proj = self.projection(x)  # (B, T, projection_dim)

        if lengths is not None and not torch.is_grad_enabled():
            rnn_out, _ = self.rnn(proj)
        elif lengths is not None and min(lengths).item() > 0:
            lengths_cpu = lengths.cpu()
            packed = pack_padded_sequence(proj, lengths_cpu, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.rnn(packed)
            rnn_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        else:
            rnn_out, _ = self.rnn(proj)

        # Mask padding frames if mask is provided
        if padding_mask is not None:
            mask_exp = padding_mask.unsqueeze(-1).float()
            rnn_out = rnn_out * mask_exp

        return rnn_out  # (B, T, output_dim)


class ContinuousTCNEncoder(nn.Module):
    """
    Continuous Dilated Temporal Convolutional Backbone.
    Maps (B, T, input_dim) -> (B, T, channels[-1]).
    """

    def __init__(
        self,
        num_landmarks: int = 543,
        coordinates_per_landmark: int = 3,
        projection_dim: int = 256,
        channels: List[int] = [256, 256, 256],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.coordinates_per_landmark = coordinates_per_landmark
        self.input_dim = num_landmarks * coordinates_per_landmark
        self.output_dim = channels[-1]

        # 1. Spatial Feature Projection
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

        # 2. Stacked Dilated Residual Blocks
        layers = []
        in_ch = projection_dim
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(
                TemporalResidualBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, num_landmarks, 3) or (B, T, input_dim).
            padding_mask: Boolean tensor (B, T).

        Returns:
            Per-frame temporal embeddings of shape (B, T, output_dim).
        """
        B, T = x.shape[0], x.shape[1]
        if x.ndim == 4:
            x = x.reshape(B, T, -1)

        proj = self.projection(x)  # (B, T, projection_dim)
        tcn_in = proj.transpose(1, 2)  # (B, projection_dim, T)
        tcn_out = self.tcn(tcn_in)     # (B, output_dim, T)
        tcn_out = tcn_out.transpose(1, 2)  # (B, T, output_dim)

        if padding_mask is not None:
            mask_exp = padding_mask.unsqueeze(-1).float()
            if mask_exp.shape[1] > tcn_out.shape[1]:
                mask_exp = mask_exp[:, :tcn_out.shape[1], :]
            tcn_out = tcn_out * mask_exp

        return tcn_out


class ContinuousTemporalEncoder(nn.Module):
    """
    Unified Factory for Continuous Temporal Encoders (BiGRU / TCN).
    """

    def __init__(
        self,
        backbone: str = "gru",
        num_landmarks: int = 543,
        coordinates_per_landmark: int = 3,
        projection_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        channels: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone_type = backbone.lower()
        if self.backbone_type in ("gru", "lstm", "rnn", "bigru"):
            self.encoder = ContinuousBiGRUEncoder(
                num_landmarks=num_landmarks,
                coordinates_per_landmark=coordinates_per_landmark,
                projection_dim=projection_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
            )
        elif self.backbone_type == "tcn":
            self.encoder = ContinuousTCNEncoder(
                num_landmarks=num_landmarks,
                coordinates_per_landmark=coordinates_per_landmark,
                projection_dim=projection_dim,
                channels=channels or [256, 256, 256],
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported temporal backbone: {backbone}")

        self.output_dim = self.encoder.output_dim

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.encoder(x, padding_mask=padding_mask, lengths=lengths, **kwargs)

    def load_from_phase3_checkpoint(self, checkpoint_path: Union[str, Path]):
        """
        Transfers projection and temporal backbone weights from a Phase 3 isolated model checkpoint.
        """
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        # Filter out Phase 3 classification head weights
        encoder_weights = {
            k: v for k, v in state_dict.items()
            if not k.startswith("classifier")
        }
        missing, unexpected = self.encoder.load_state_dict(encoder_weights, strict=False)
        return {
            "loaded_weights_count": len(encoder_weights),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }
