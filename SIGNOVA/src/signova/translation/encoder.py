"""
Encoder module for Gloss-to-English sequence-to-sequence translation.

Guiding Principles:
1. Operates strictly on discrete gloss token sequences (integer token IDs).
2. Bidirectional recurrent network (BiGRU) for contextual representation.
3. Proper padding masking and packed sequence handling.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiGRUEncoder(nn.Module):
    """
    Bidirectional GRU Encoder mapping discrete source gloss IDs to sequence representations.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Project concatenated bidirectional final hidden states to decoder hidden dimension
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        source_ids: torch.Tensor,
        source_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            source_ids: Tensor of shape (batch_size, seq_len)
            source_lengths: Optional Tensor of shape (batch_size,) with true lengths

        Returns:
            encoder_outputs: Tensor of shape (batch_size, seq_len, hidden_dim * 2)
            decoder_init_hidden: Tensor of shape (batch_size, hidden_dim)
        """
        # (batch_size, seq_len, embed_dim)
        embedded = self.dropout(self.embedding(source_ids))

        if source_lengths is not None:
            # Move lengths to CPU for pack_padded_sequence if needed
            lengths_cpu = source_lengths.to(device="cpu", dtype=torch.int64)
            # Enforce minimum length of 1 to prevent packing crashes
            lengths_cpu = torch.clamp(lengths_cpu, min=1)
            packed = pack_padded_sequence(
                embedded,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_outputs, hidden = self.gru(packed)
            # encoder_outputs: (batch_size, seq_len, hidden_dim * 2)
            encoder_outputs, _ = pad_packed_sequence(
                packed_outputs,
                batch_first=True,
                total_length=source_ids.size(1),
            )
        else:
            encoder_outputs, hidden = self.gru(embedded)

        # hidden is of shape (num_layers * 2, batch_size, hidden_dim)
        # Take the top-layer forward and backward hidden states
        # forward: hidden[-2, :, :], backward: hidden[-1, :, :]
        last_forward = hidden[-2, :, :]
        last_backward = hidden[-1, :, :]
        combined_hidden = torch.cat([last_forward, last_backward], dim=1)  # (batch, hidden_dim * 2)
        decoder_init_hidden = torch.tanh(self.fc_hidden(combined_hidden))  # (batch, hidden_dim)

        return encoder_outputs, decoder_init_hidden
