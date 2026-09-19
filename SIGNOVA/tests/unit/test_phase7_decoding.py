"""
Unit tests for Greedy and Beam Search Decoders.
"""

import torch

from signova.translation.decoding import BeamSearchDecoder, GreedyDecoder
from signova.translation.model import SyntheticGlossToEnglishSeq2Seq
from signova.translation.vocabulary import TranslationVocabulary


def test_greedy_and_beam_decoding():
    src_vocab = TranslationVocabulary().build_from_sequences([["I", "GO", "COLLEGE"]])
    tgt_vocab = TranslationVocabulary().build_from_sequences([["i", "am", "going", "to", "college"]])

    model = SyntheticGlossToEnglishSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_embed_dim=16,
        tgt_embed_dim=16,
        enc_hidden_dim=32,
        dec_hidden_dim=32,
    )
    model.eval()

    source_ids = torch.tensor([src_vocab.encode(["I", "GO"])], dtype=torch.long)

    # Test Greedy
    greedy_decoder = GreedyDecoder(max_len=8, repetition_penalty=1.2)
    greedy_res = greedy_decoder.decode_single(model, source_ids, tgt_vocab)

    assert isinstance(greedy_res.text, str)
    assert greedy_res.decoding_latency_ms >= 0.0
    assert isinstance(greedy_res.terminated_by_eos, bool)
    assert len(greedy_res.token_ids) <= 8

    # Test Beam Search
    beam_decoder = BeamSearchDecoder(beam_width=2, max_len=8)
    beam_res = beam_decoder.decode_single(model, source_ids, tgt_vocab)

    assert isinstance(beam_res.text, str)
    assert beam_res.decoding_latency_ms >= 0.0
    assert isinstance(beam_res.terminated_by_eos, bool)
