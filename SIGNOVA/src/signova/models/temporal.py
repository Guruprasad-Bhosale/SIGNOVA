"""
Temporal Sequence Model interface and placeholder for SIGNOVA.

Planned for Phase 4/5: Multi-layer Transformer Encoder or Conformer to model
temporal dynamics of sign movements across frames.
"""

from typing import Any, Dict, Optional


class TemporalModelPlaceholder:
    """
    Placeholder interface for temporal sequence encoder.

    TODO (Phase 4/5):
    - Implement Transformer Encoder with sinusoidal / learned positional encodings.
    - Input: Tensor of shape (Batch, Frames=T, d_model=256).
    - Output: Sequence representations of shape (Batch, Frames=T, d_model=256).
    - Apply dropout and attention masking for variable sequence lengths.
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 1024):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward

    def forward(self, x: Any, mask: Optional[Any] = None) -> Any:
        raise NotImplementedError(
            "TemporalModel is a Phase 0 placeholder. "
            "Sequence modeling components will be implemented in Phase 4/5."
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "status": "planned_phase_4_5",
        }
