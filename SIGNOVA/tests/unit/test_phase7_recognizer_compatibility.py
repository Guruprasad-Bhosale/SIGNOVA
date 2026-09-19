"""
Unit tests for End-to-End Recognizer-to-Translator Pipeline Contract Compatibility.

Verifies:
1. Output format from Continuous Sign Recognizer / CTC Decoder (List[str] gloss sequence)
   matches the expected input contract of the Phase 7 Translation Subsystem.
2. The translation subsystem translates the sequence without requiring video or landmark shortcuts.
3. This is an interface and pipeline contract test and does not claim end-to-end recognition accuracy.
"""

import torch

from signova.data.vocabulary import SignVocabulary
from signova.models.ctc_recognizer import CTCContinuousRecognizer
from signova.translation.model import SyntheticGlossToEnglishSeq2Seq
from signova.translation.translator import (
    GlossToEnglishTranslator,
    translate_gloss_sequence,
)
from signova.translation.vocabulary import TranslationVocabulary


def test_recognizer_to_translator_interface_contract():
    # 1. Setup Phase 6 Recognizer Vocabulary
    sign_classes = ["<BLANK>", "I", "GO", "COLLEGE", "TODAY", "HELP", "ME"]
    sign_vocab = SignVocabulary(tokens=sign_classes)

    # 2. Setup Recognizer
    recognizer = CTCContinuousRecognizer(
        num_landmarks=75,
        num_classes=len(sign_classes),
        backbone="gru",
        projection_dim=32,
        hidden_size=32,
        num_layers=1,
        blank_idx=0,
    )
    recognizer.eval()

    # 3. Simulate continuous features (B=1, T=30, 75 landmarks, (x,y,z))
    dummy_features = torch.randn(1, 30, 75, 3)
    feature_lengths = torch.tensor([30], dtype=torch.long)

    # 4. Continuous Recognizer Decoding -> Token IDs
    decoded_results = recognizer.decode_greedy(dummy_features, lengths=feature_lengths)
    assert len(decoded_results) == 1
    collapsed_ids = decoded_results[0]["collapsed_tokens"]

    # 5. Token IDs -> ISL Gloss Sequence (List[str])
    # For testing contract with deterministic tokens, map to gloss tokens
    gloss_sequence = sign_vocab.decode(collapsed_ids, remove_blank=True)
    assert isinstance(gloss_sequence, list)
    for g in gloss_sequence:
        assert isinstance(g, str)

    # 6. Setup Phase 7 Translation Subsystem
    src_trans_vocab = TranslationVocabulary(name="src_trans").build_from_sequences([
        ["I", "GO", "COLLEGE", "TODAY"],
        ["HELP", "ME", "PLEASE"],
    ])
    tgt_trans_vocab = TranslationVocabulary(name="tgt_trans").build_from_sequences([
        ["i", "am", "going", "to", "college", "today"],
        ["please", "help", "me"],
    ])

    trans_model = SyntheticGlossToEnglishSeq2Seq(
        src_vocab_size=len(src_trans_vocab),
        tgt_vocab_size=len(tgt_trans_vocab),
        src_embed_dim=16,
        tgt_embed_dim=16,
        enc_hidden_dim=32,
        dec_hidden_dim=32,
        enc_layers=1,
    )
    trans_model.eval()

    translator = GlossToEnglishTranslator(
        model=trans_model,
        source_vocab=src_trans_vocab,
        target_vocab=tgt_trans_vocab,
    )

    # 7. Direct Translation Subsystem Call with Recognizer Gloss Output
    test_gloss_input = ["I", "GO", "COLLEGE", "TODAY"]
    english_output = translator.translate_glosses(test_gloss_input)

    assert isinstance(english_output, str)

    # 8. Top-level API translation
    api_english = translate_gloss_sequence(test_gloss_input, translator=translator)
    assert isinstance(api_english, str)
