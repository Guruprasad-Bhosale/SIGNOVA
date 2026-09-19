"""
Connectionist Temporal Classification (CTC) Decoder interface and placeholder for SIGNOVA.

Planned for Phase 5/6: Alignment-free continuous gloss decoding with Beam Search
and Language Model rescoring.
"""

from typing import Any, Dict, List, Optional


class CTCDecoderPlaceholder:
    """
    Placeholder interface for CTC Decoder and gloss sequence generator.

    TODO (Phase 5/6):
    - Linear projection head from d_model (256) to vocab_size + 1 (blank token).
    - CTC Loss calculation during training.
    - Greedy and Prefix Beam Search decoding during inference.
    - Gloss sequence post-processing and collapse duplicate tokens.
    """

    def __init__(self, vocab_size: int = 1000, blank_token: int = 0, beam_width: int = 5):
        self.vocab_size = vocab_size
        self.blank_token = blank_token
        self.beam_width = beam_width

    def decode_greedy(self, logits: Any) -> List[List[int]]:
        raise NotImplementedError(
            "CTCDecoder is a Phase 0 placeholder. "
            "Decoding logic will be implemented in Phase 5/6."
        )

    def decode_beam_search(self, logits: Any, beam_width: Optional[int] = None) -> List[List[int]]:
        raise NotImplementedError(
            "Beam search decoding is a Phase 0 placeholder. "
            "Will be implemented in Phase 5/6."
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "blank_token": self.blank_token,
            "beam_width": self.beam_width,
            "status": "planned_phase_5_6",
        }
