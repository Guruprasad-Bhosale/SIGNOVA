"""
Model architectures and neural network modules for SIGNOVA.
"""

from signova.models.baseline_pooled import StaticPooledMLP
from signova.models.baseline_rnn import BaselineRNN
from signova.models.baseline_tcn import BaselineTCN
from signova.models.ctc import CTCDecoderPlaceholder
from signova.models.encoder import SpatialEncoderPlaceholder
from signova.models.temporal import TemporalModelPlaceholder

__all__ = [
    "SpatialEncoderPlaceholder",
    "TemporalModelPlaceholder",
    "CTCDecoderPlaceholder",
    "StaticPooledMLP",
    "BaselineRNN",
    "BaselineTCN",
]

