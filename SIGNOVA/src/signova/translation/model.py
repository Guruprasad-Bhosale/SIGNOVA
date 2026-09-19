"""
Sequence-to-sequence neural model for Gloss-to-English translation.

Guiding Principles:
1. Model class clearly named `SyntheticGlossToEnglishSeq2Seq` (with `Seq2SeqTranslationModel` alias).
2. Resource budget: strictly < 10M parameters for fast CPU/GPU execution.
3. Supports teacher forcing during training and autoregressive step-by-step generation during inference.
4. Preserves model card metadata explicitly stating synthetic training fixture status when in State C.
"""

from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from signova.translation.decoder import AttentiveGRUDecoder
from signova.translation.encoder import BiGRUEncoder


class SyntheticGlossToEnglishSeq2Seq(nn.Module):
    """
    BiGRU + Attention + GRU Seq2Seq model for Gloss-to-English Translation.
    
    Marked synthetic when trained without verified real ISL paired supervision.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_embed_dim: int = 128,
        tgt_embed_dim: int = 128,
        enc_hidden_dim: int = 256,
        dec_hidden_dim: int = 256,
        attn_dim: int = 128,
        enc_layers: int = 2,
        dropout: float = 0.1,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0,
        tgt_bos_idx: int = 1,
        tgt_eos_idx: int = 2,
        is_synthetic: bool = True,
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_embed_dim = src_embed_dim
        self.tgt_embed_dim = tgt_embed_dim
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.attn_dim = attn_dim
        self.enc_layers = enc_layers
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_bos_idx = tgt_bos_idx
        self.tgt_eos_idx = tgt_eos_idx
        self.is_synthetic = is_synthetic

        self.encoder = BiGRUEncoder(
            vocab_size=src_vocab_size,
            embed_dim=src_embed_dim,
            hidden_dim=enc_hidden_dim,
            num_layers=enc_layers,
            dropout=dropout,
            pad_idx=src_pad_idx,
        )

        self.decoder = AttentiveGRUDecoder(
            vocab_size=tgt_vocab_size,
            embed_dim=tgt_embed_dim,
            enc_dim=enc_hidden_dim * 2,
            dec_dim=dec_hidden_dim,
            attn_dim=attn_dim,
            dropout=dropout,
            pad_idx=tgt_pad_idx,
        )

    def count_parameters(self) -> int:
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        source_ids: torch.Tensor,
        target_ids: torch.Tensor,
        source_lengths: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with configurable teacher forcing for training.

        Args:
            source_ids: Tensor of shape (batch_size, src_len)
            target_ids: Tensor of shape (batch_size, tgt_len) including BOS token at idx 0
            source_lengths: Optional Tensor of shape (batch_size,)
            source_mask: Optional boolean Tensor of shape (batch_size, src_len)
            teacher_forcing_ratio: Probability of using ground truth token as next input

        Returns:
            outputs: Tensor of shape (batch_size, tgt_len - 1, tgt_vocab_size)
            attentions: Tensor of shape (batch_size, tgt_len - 1, src_len)
        """
        batch_size = source_ids.size(0)
        tgt_len = target_ids.size(1)

        if source_mask is None:
            source_mask = (source_ids != self.src_pad_idx)

        # Encode source sequence
        encoder_outputs, decoder_hidden = self.encoder(source_ids, source_lengths)

        # Tensor to store decoder outputs
        outputs = torch.zeros(
            batch_size,
            tgt_len - 1,
            self.tgt_vocab_size,
            device=source_ids.device,
        )
        attentions = torch.zeros(
            batch_size,
            tgt_len - 1,
            source_ids.size(1),
            device=source_ids.device,
        )

        # First input token is BOS (target_ids[:, 0])
        input_token = target_ids[:, 0]

        for t in range(1, tgt_len):
            logits, decoder_hidden, attn_weights = self.decoder(
                input_token_id=input_token,
                decoder_hidden=decoder_hidden,
                encoder_outputs=encoder_outputs,
                source_mask=source_mask,
            )
            outputs[:, t - 1] = logits
            attentions[:, t - 1] = attn_weights

            # Teacher forcing decision
            is_teacher_force = (random.random() < teacher_forcing_ratio) if self.training else False
            top1 = logits.argmax(dim=-1)
            input_token = target_ids[:, t] if is_teacher_force else top1

        return outputs, attentions

    @torch.no_grad()
    def generate_greedy(
        self,
        source_ids: torch.Tensor,
        source_lengths: Optional[torch.Tensor] = None,
        source_mask: Optional[torch.Tensor] = None,
        max_len: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy autoregressive decoding.

        Args:
            source_ids: Tensor of shape (batch_size, src_len)
            source_lengths: Optional Tensor of shape (batch_size,)
            source_mask: Optional boolean Tensor of shape (batch_size, src_len)
            max_len: Maximum target generation length

        Returns:
            predicted_ids: Tensor of shape (batch_size, gen_len)
            attentions: Tensor of shape (batch_size, gen_len, src_len)
        """
        self.eval()
        batch_size = source_ids.size(0)

        if source_mask is None:
            source_mask = (source_ids != self.src_pad_idx)

        encoder_outputs, decoder_hidden = self.encoder(source_ids, source_lengths)

        input_token = torch.full(
            (batch_size,),
            fill_value=self.tgt_bos_idx,
            dtype=torch.long,
            device=source_ids.device,
        )

        generated_tokens = []
        attentions = []

        finished = torch.zeros(batch_size, dtype=torch.bool, device=source_ids.device)

        for _ in range(max_len):
            logits, decoder_hidden, attn_weights = self.decoder(
                input_token_id=input_token,
                decoder_hidden=decoder_hidden,
                encoder_outputs=encoder_outputs,
                source_mask=source_mask,
            )
            top1 = logits.argmax(dim=-1)  # (batch_size,)

            generated_tokens.append(top1.unsqueeze(1))
            attentions.append(attn_weights.unsqueeze(1))

            # Update finished status
            finished = finished | (top1 == self.tgt_eos_idx)
            if finished.all():
                break

            input_token = top1

        if len(generated_tokens) > 0:
            pred_tensor = torch.cat(generated_tokens, dim=1)
            attn_tensor = torch.cat(attentions, dim=1)
        else:
            pred_tensor = torch.empty((batch_size, 0), dtype=torch.long, device=source_ids.device)
            attn_tensor = torch.empty((batch_size, 0, source_ids.size(1)), device=source_ids.device)

        return pred_tensor, attn_tensor

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save model checkpoint and architecture hyperparameters."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": {
                "src_vocab_size": self.src_vocab_size,
                "tgt_vocab_size": self.tgt_vocab_size,
                "src_embed_dim": self.src_embed_dim,
                "tgt_embed_dim": self.tgt_embed_dim,
                "enc_hidden_dim": self.enc_hidden_dim,
                "dec_hidden_dim": self.dec_hidden_dim,
                "attn_dim": self.attn_dim,
                "enc_layers": self.enc_layers,
                "src_pad_idx": self.src_pad_idx,
                "tgt_pad_idx": self.tgt_pad_idx,
                "tgt_bos_idx": self.tgt_bos_idx,
                "tgt_eos_idx": self.tgt_eos_idx,
                "is_synthetic": self.is_synthetic,
            },
            "parameters_count": self.count_parameters(),
            "extra_meta": extra_meta or {},
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load_checkpoint(
        cls,
        filepath: Union[str, Path],
        device: str = "cpu",
    ) -> "SyntheticGlossToEnglishSeq2Seq":
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint["config"]
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model


# Public alias
Seq2SeqTranslationModel = SyntheticGlossToEnglishSeq2Seq
