"""
Temporal Convolutional Network (TCN) Baseline for SIGNOVA.

Processes 543-keypoint skeletal landmark sequences through 1D dilated residual
temporal convolutions, providing efficient parallel training and dynamic receptive fields.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class TemporalResidualBlock(nn.Module):
    """
    Dilated 1D Temporal Convolutional Block with Residual Connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        # Match dimensions if padding caused 1-frame disparity
        if out.shape[2] != res.shape[2]:
            min_len = min(out.shape[2], res.shape[2])
            out = out[:, :, :min_len]
            res = res[:, :, :min_len]

        return out + res


class BaselineTCN(nn.Module):
    """
    Baseline B: Dilated Temporal Convolutional Network.
    """

    def __init__(
        self,
        num_landmarks: int = 543,
        coordinates_per_landmark: int = 3,
        num_classes: int = 10,
        projection_dim: int = 256,
        channels: List[int] = [256, 256, 256],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.coordinates_per_landmark = coordinates_per_landmark
        self.input_dim = num_landmarks * coordinates_per_landmark
        self.num_classes = num_classes

        # Spatial Feature Projection
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

        # Stacked Temporal Blocks with Exponential Dilations (1, 2, 4)
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
        final_ch = channels[-1]

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(final_ch, final_ch // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_ch // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, num_landmarks, 3) or (B, T, input_dim).
            padding_mask: Boolean tensor of shape (B, T).

        Returns:
            logits: Tensor of shape (B, num_classes).
        """
        B, T = x.shape[0], x.shape[1]
        if x.ndim == 4:
            x = x.reshape(B, T, -1)  # (B, T, input_dim)


        # 1. Project frame-wise
        proj = self.projection(x)  # (B, T, projection_dim)

        # 2. Convolve over temporal dimension: (B, Channels, T)
        tcn_in = proj.transpose(1, 2)
        tcn_out = self.tcn(tcn_in)  # (B, final_ch, T)
        tcn_out = tcn_out.transpose(1, 2)  # (B, T, final_ch)

        # 3. Masked Temporal Pooling
        if padding_mask is not None:
            mask_expanded = padding_mask.unsqueeze(-1).float()
            # Match lengths if any convolution truncation
            if mask_expanded.shape[1] > tcn_out.shape[1]:
                mask_expanded = mask_expanded[:, :tcn_out.shape[1], :]
            sum_tcn = (tcn_out * mask_expanded).sum(dim=1)
            eff_len = mask_expanded.sum(dim=1).clamp(min=1.0)
            pooled = sum_tcn / eff_len
        else:
            pooled = tcn_out.mean(dim=1)

        logits = self.classifier(pooled)
        return logits
