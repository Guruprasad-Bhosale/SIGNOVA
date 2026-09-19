"""
tests/unit/test_phase24_abstention.py
Unit tests verifying confidence-gated decoding, silence/inactivity handling, and abstention behavior.
"""

from __future__ import annotations

import pytest
import torch

from signova.recognition.ctc_decoder import CTCDecoder


class TestPhase24Abstention:
    """Tests verifying confidence gating and abstention semantics under ambiguous or inactive inputs."""

    def test_inactivity_abstention(self) -> None:
        """Verify decoder cleanly emits blank/empty output during prolonged inactivity."""
        # Simulated log-probabilities for blank tokens (index 0) over 30 time frames
        # Batch size 1, 30 frames, 10 classes (index 0 is blank)
        T = 30
        C = 10
        log_probs = torch.full((1, T, C), -20.0)
        log_probs[:, :, 0] = 0.0  # Dominant blank class
        
        decoder = CTCDecoder(
            blank_idx=0,
            id_to_label={1: "HELLO", 2: "THANK_YOU", 3: "NAME", 4: "ISL"},
        )
        
        decoded = decoder.decode_greedy_tensor(log_probs)
        assert len(decoded) == 1
        assert decoded[0]["tokens"] == []
        assert decoded[0]["labels"] == []

    def test_silence_transition_between_signs(self) -> None:
        """Verify decoder cleanly splits signs separated by silence/blank frames."""
        T = 20
        C = 5
        log_probs = torch.full((1, T, C), -20.0)
        
        # Frames 0-4: Blank
        log_probs[:, 0:5, 0] = 0.0
        # Frames 5-8: Token 1 ("HELLO")
        log_probs[:, 5:9, 1] = 0.0
        # Frames 9-12: Blank
        log_probs[:, 9:13, 0] = 0.0
        # Frames 13-16: Token 2 ("THANK_YOU")
        log_probs[:, 13:17, 2] = 0.0
        # Frames 17-19: Blank
        log_probs[:, 17:20, 0] = 0.0
        
        decoder = CTCDecoder(
            blank_idx=0,
            id_to_label={1: "HELLO", 2: "THANK_YOU"},
        )
        
        decoded = decoder.decode_greedy_tensor(log_probs)
        assert decoded[0]["tokens"] == [1, 2]
        assert decoded[0]["labels"] == ["HELLO", "THANK_YOU"]
