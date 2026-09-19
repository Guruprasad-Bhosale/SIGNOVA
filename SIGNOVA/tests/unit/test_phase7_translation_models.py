"""
Unit tests for Phase 7 Neural Seq2Seq Translation Models (Encoder, Attention, Decoder, Seq2Seq).
"""

from pathlib import Path
import tempfile
import torch

from signova.translation.attention import BahdanauAttention
from signova.translation.decoder import AttentiveGRUDecoder
from signova.translation.encoder import BiGRUEncoder
from signova.translation.model import SyntheticGlossToEnglishSeq2Seq


def test_bigru_encoder_forward():
    batch_size = 3
    seq_len = 5
    vocab_size = 20
    embed_dim = 32
    hidden_dim = 64

    encoder = BiGRUEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
    )

    source_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    source_lengths = torch.tensor([5, 4, 2], dtype=torch.long)

    outputs, hidden = encoder(source_ids, source_lengths)

    assert outputs.shape == (batch_size, seq_len, hidden_dim * 2)
    assert hidden.shape == (batch_size, hidden_dim)


def test_bahdanau_attention():
    batch_size = 2
    src_len = 4
    enc_dim = 64
    dec_dim = 32
    attn_dim = 16

    attention = BahdanauAttention(enc_dim=enc_dim, dec_dim=dec_dim, attn_dim=attn_dim)

    dec_hidden = torch.randn(batch_size, dec_dim)
    enc_outputs = torch.randn(batch_size, src_len, enc_dim)
    # Mask out the last token of the second sequence
    source_mask = torch.tensor([[True, True, True, True], [True, True, True, False]])

    context, attn_weights = attention(dec_hidden, enc_outputs, source_mask=source_mask)

    assert context.shape == (batch_size, enc_dim)
    assert attn_weights.shape == (batch_size, src_len)
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(batch_size), atol=1e-4)
    # Masked token in sample 2 should have near zero attention weight
    assert attn_weights[1, 3].item() < 1e-4


def test_attentive_gru_decoder_step():
    batch_size = 2
    tgt_vocab_size = 30
    embed_dim = 32
    enc_dim = 64
    dec_dim = 64

    decoder = AttentiveGRUDecoder(
        vocab_size=tgt_vocab_size,
        embed_dim=embed_dim,
        enc_dim=enc_dim,
        dec_dim=dec_dim,
    )

    input_token = torch.tensor([1, 2], dtype=torch.long)
    dec_hidden = torch.randn(batch_size, dec_dim)
    enc_outputs = torch.randn(batch_size, 5, enc_dim)

    logits, new_hidden, attn_weights = decoder(input_token, dec_hidden, enc_outputs)

    assert logits.shape == (batch_size, tgt_vocab_size)
    assert new_hidden.shape == (batch_size, dec_dim)
    assert attn_weights.shape == (batch_size, 5)


def test_synthetic_gloss_to_english_seq2seq_forward_and_generate():
    src_vocab = 25
    tgt_vocab = 35
    model = SyntheticGlossToEnglishSeq2Seq(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        src_embed_dim=32,
        tgt_embed_dim=32,
        enc_hidden_dim=64,
        dec_hidden_dim=64,
        enc_layers=1,
    )

    # Resource budget constraint test (< 10M params)
    param_count = model.count_parameters()
    assert param_count < 10_000_000

    batch_size = 2
    src_len = 6
    tgt_len = 7

    source_ids = torch.randint(0, src_vocab, (batch_size, src_len))
    target_ids = torch.randint(0, tgt_vocab, (batch_size, tgt_len))
    # Ensure first token is BOS
    target_ids[:, 0] = model.tgt_bos_idx

    # Training forward pass
    outputs, attentions = model(source_ids, target_ids, teacher_forcing_ratio=0.5)
    assert outputs.shape == (batch_size, tgt_len - 1, tgt_vocab)
    assert attentions.shape == (batch_size, tgt_len - 1, src_len)

    # Autoregressive generation
    pred_tokens, pred_attentions = model.generate_greedy(source_ids, max_len=10)
    assert pred_tokens.shape[0] == batch_size
    assert pred_tokens.shape[1] <= 10


def test_seq2seq_checkpoint_save_and_load():
    model = SyntheticGlossToEnglishSeq2Seq(
        src_vocab_size=15,
        tgt_vocab_size=20,
        src_embed_dim=16,
        tgt_embed_dim=16,
        enc_hidden_dim=32,
        dec_hidden_dim=32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ckpt.pt"
        model.save_checkpoint(ckpt_path, extra_meta={"note": "test_save"})
        assert ckpt_path.exists()

        loaded_model = SyntheticGlossToEnglishSeq2Seq.load_checkpoint(ckpt_path, device="cpu")
        assert loaded_model.src_vocab_size == 15
        assert loaded_model.tgt_vocab_size == 20
        assert loaded_model.count_parameters() == model.count_parameters()
