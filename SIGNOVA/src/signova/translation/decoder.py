"""
Attentive GRU Decoder module for sequence-to-sequence translation.

Guiding Principles:
1. Autoregressively generates target tokens conditioned on previous tokens and attention context.
2. Integrates Bahdanau attention over source representations.
3. Clean single-step decoding interface suitable for both training and inference loops.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from signova.translation.attention import BahdanauAttention


class AttentiveGRUDecoder(nn.Module):
    """
    Attentive GRU Decoder predicting target vocabulary distributions.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        enc_dim: int = 512,  # BiGRU enc_hidden * 2
        dec_dim: int = 256,
        attn_dim: int = 128,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.attention = BahdanauAttention(
            enc_dim=enc_dim,
            dec_dim=dec_dim,
            attn_dim=attn_dim,
        )

        # GRU Cell taking embedded token concatenated with context vector
        self.gru_cell = nn.GRUCell(
            input_size=embed_dim + enc_dim,
            hidden_size=dec_dim,
        )

        # Output projection layer: [embedded_y, dec_hidden, context] -> target_vocab
        self.fc_out = nn.Linear(embed_dim + dec_dim + enc_dim, vocab_size)

    def forward(
        self,
        input_token_id: torch.Tensor,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single decoding step.

        Args:
            input_token_id: Tensor of shape (batch_size,) containing target token ID for this step.
            decoder_hidden: Tensor of shape (batch_size, dec_dim).
            encoder_outputs: Tensor of shape (batch_size, src_len, enc_dim).
            source_mask: Optional boolean Tensor of shape (batch_size, src_len).

        Returns:
            logits: Tensor of shape (batch_size, vocab_size)
            new_hidden: Tensor of shape (batch_size, dec_dim)
            attention_weights: Tensor of shape (batch_size, src_len)
        """
        # (batch_size, embed_dim)
        embedded = self.dropout(self.embedding(input_token_id))

        # Calculate attention context given current hidden state
        # context: (batch_size, enc_dim), attn_weights: (batch_size, src_len)
        context, attn_weights = self.attention(
            decoder_hidden=decoder_hidden,
            encoder_outputs=encoder_outputs,
            source_mask=source_mask,
        )

        # gru_input: (batch_size, embed_dim + enc_dim)
        gru_input = torch.cat([embedded, context], dim=1)

        # Update hidden state
        new_hidden = self.gru_cell(gru_input, decoder_hidden)

        # Concatenate for output projection: (batch_size, embed_dim + dec_dim + enc_dim)
        out_features = torch.cat([embedded, new_hidden, context], dim=1)
        logits = self.fc_out(self.dropout(out_features))

        return logits, new_hidden, attn_weights
