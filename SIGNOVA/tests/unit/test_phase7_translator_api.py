"""
Unit tests for production GlossToEnglishTranslator and translate_gloss_sequence API.
"""

from pathlib import Path
import tempfile
import torch

from signova.translation.model import SyntheticGlossToEnglishSeq2Seq
from signova.translation.normalization import TranslationNormalizer
from signova.translation.translator import (
    GlossToEnglishTranslator,
    translate_gloss_sequence,
)
from signova.translation.vocabulary import TranslationVocabulary


def test_translator_edge_cases_and_robustness():
    # 1. Unloaded / default translator fallback
    unloaded_translator = GlossToEnglishTranslator()
    assert unloaded_translator.translate_glosses([]) == ""
    assert unloaded_translator.translate_glosses(["HELLO", "WORLD"]) == "hello world"

    # 2. Fully loaded translator
    src_vocab = TranslationVocabulary().build_from_sequences([["I", "GO", "COLLEGE", "TODAY"]])
    tgt_vocab = TranslationVocabulary().build_from_sequences([["i", "am", "going", "to", "college", "today"]])

    model = SyntheticGlossToEnglishSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_embed_dim=16,
        tgt_embed_dim=16,
        enc_hidden_dim=32,
        dec_hidden_dim=32,
    )
    model.eval()

    translator = GlossToEnglishTranslator(
        model=model,
        source_vocab=src_vocab,
        target_vocab=tgt_vocab,
        max_source_len=10,
    )

    # Empty sequence
    assert translator.translate_glosses([]) == ""

    # All-unknown sequence
    assert translator.translate_glosses(["RANDOM_UNKNOWN_SIGN", "ANOTHER_UNKNOWN"]) == ""

    # Valid sequence
    res_greedy = translator.translate_glosses(["I", "GO", "COLLEGE"], beam_width=1)
    assert isinstance(res_greedy, str)

    # Beam search
    res_beam = translator.translate_glosses(["I", "GO", "COLLEGE"], beam_width=2)
    assert isinstance(res_beam, str)

    # Top level API call
    api_res = translate_gloss_sequence(["I", "GO"], translator=translator)
    assert isinstance(api_res, str)

    # Truncation for excessively long sequence
    long_seq = ["I"] * 50
    res_long = translator.translate_glosses(long_seq)
    assert isinstance(res_long, str)


def test_translator_load_from_directory():
    src_vocab = TranslationVocabulary(name="src").build_from_sequences([["I", "GO"]])
    tgt_vocab = TranslationVocabulary(name="tgt").build_from_sequences([["i", "go"]])

    model = SyntheticGlossToEnglishSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_embed_dim=16,
        tgt_embed_dim=16,
        enc_hidden_dim=32,
        dec_hidden_dim=32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src_vocab.save(tmp_path / "source_vocab.json")
        tgt_vocab.save(tmp_path / "target_vocab.json")
        model.save_checkpoint(tmp_path / "model_checkpoint.pt")

        loaded_translator = GlossToEnglishTranslator.load_from_directory(tmp_path, device="cpu")
        assert loaded_translator.is_loaded is True
        status = loaded_translator.get_status()
        assert status["is_loaded"] is True
        assert status["source_vocab_size"] == len(src_vocab)
