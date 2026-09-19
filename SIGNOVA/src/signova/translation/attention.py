"""
Bahdanau (Additive) Attention module for sequence-to-sequence translation.

Guiding Principles:
1. Aligns target decoder state with source encoder representations.
2. Supports strict padding masking over variable-length source sequences.
3. Produces interpretable attention distribution weights.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Additive Attention (Bahdanau et al., 2015).
    energy = v^T * tanh(W_enc * enc_outputs + W_dec * dec_hidden)
    """

    def __init__(
        self,
        enc_dim: int,
        dec_dim: int,
        attn_dim: int = 128,
    ):
        super().__init__()
        self.enc_dim = enc_dim
        self.dec_dim = dec_dim
        self.attn_dim = attn_dim

        self.W_enc = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: Tensor of shape (batch_size, dec_dim)
            encoder_outputs: Tensor of shape (batch_size, src_len, enc_dim)
            source_mask: Optional boolean Tensor of shape (batch_size, src_len)
                         where True = valid token, False = padding.

        Returns:
            context: Tensor of shape (batch_size, enc_dim)
            attention_weights: Tensor of shape (batch_size, src_len)
        """
        src_len = encoder_outputs.size(1)

        # (batch_size, src_len, attn_dim)
        proj_enc = self.W_enc(encoder_outputs)

        # (batch_size, 1, attn_dim) -> (batch_size, src_len, attn_dim)
        proj_dec = self.W_dec(decoder_hidden).unsqueeze(1).repeat(1, src_len, 1)

        # energy: (batch_size, src_len, 1) -> (batch_size, src_len)
        energy = self.v(torch.tanh(proj_enc + proj_dec)).squeeze(-1)

        # Apply source padding mask if provided
        if source_mask is not None:
            # where source_mask is False (pad), set energy to large negative number
            energy = energy.masked_fill(~source_mask, -1e9)

        # attention_weights: (batch_size, src_len)
        attention_weights = F.softmax(energy, dim=-1)

        # In case all tokens were masked out (e.g. empty sequence), prevent NaN
        if source_mask is not None:
            attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        # context: (batch_size, 1, src_len) bmm (batch_size, src_len, enc_dim) -> (batch_size, enc_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attention_weights
