"""
Static Pooled MLP Baseline (Baseline 0.5) for SIGNOVA.

Performs masked spatial-temporal mean pooling across all valid frames,
followed by an MLP classification head. Serves as a critical sanity baseline
to measure how much predictive signal comes from static landmark distributions
versus dynamic temporal sequencing.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


class StaticPooledMLP(nn.Module):
    """
    Baseline 0.5: Masked Mean-Pooled Landmark Classifier.
    """

    def __init__(
        self,
        num_landmarks: int = 543,
        coordinates_per_landmark: int = 3,
        num_classes: int = 10,
        projection_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.coordinates_per_landmark = coordinates_per_landmark
        self.input_dim = num_landmarks * coordinates_per_landmark
        self.num_classes = num_classes

        # Feature Projection
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
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
            padding_mask: Boolean tensor of shape (B, T) where True indicates a valid frame.

        Returns:
            logits: Tensor of shape (B, num_classes).
        """
        B, T = x.shape[0], x.shape[1]
        if x.ndim == 4:
            x = x.reshape(B, T, -1)  # (B, T, input_dim)


        # Frame-wise projection
        proj = self.projection(x)  # (B, T, hidden_dim)

        # Masked Temporal Average Pooling
        if padding_mask is not None:
            mask_expanded = padding_mask.unsqueeze(-1).float()  # (B, T, 1)
            sum_proj = (proj * mask_expanded).sum(dim=1)       # (B, hidden_dim)
            lengths = mask_expanded.sum(dim=1).clamp(min=1.0)  # (B, 1)
            pooled = sum_proj / lengths
        else:
            pooled = proj.mean(dim=1)

        logits = self.classifier(pooled)  # (B, num_classes)
        return logits
