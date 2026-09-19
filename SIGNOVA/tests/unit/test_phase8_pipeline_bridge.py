"""
Unit tests for RecognitionToTranslationBridge (Deterministic CTC decoding, blank removal, repeat collapse).
"""

from signova.data.vocabulary import SignVocabulary
from signova.inference.pipeline import RecognitionToTranslationBridge


def test_recognition_to_translation_bridge_decoding():
    sign_classes = ["<BLANK>", "I", "GO", "COLLEGE", "TODAY"]
    vocab = SignVocabulary(tokens=sign_classes)
    bridge = RecognitionToTranslationBridge(sign_vocab=vocab, blank_idx=0)

    id_i = vocab.token_to_id["I"]
    id_go = vocab.token_to_id["GO"]
    id_college = vocab.token_to_id["COLLEGE"]
    id_today = vocab.token_to_id["TODAY"]

    # 1. Normal sequence with blanks and repeats
    raw_ctc_ids = [0, 0, id_i, id_i, id_i, 0, id_go, id_go, 0, id_college, id_college, 0, id_today, 0]
    glosses = bridge.decode_tokens_to_glosses(raw_ctc_ids)
    assert glosses == ["I", "GO", "COLLEGE", "TODAY"]

    # 2. Sequence with repeated distinct signs separated by blank
    repeated_signs_ctc = [id_i, id_i, 0, id_i, id_i]
    glosses_rep = bridge.decode_tokens_to_glosses(repeated_signs_ctc)
    assert glosses_rep == ["I", "I"]

    # 3. All-blank sequence
    all_blanks = [0, 0, 0, 0]
    assert bridge.decode_tokens_to_glosses(all_blanks) == []

    # 4. Empty input
    assert bridge.decode_tokens_to_glosses([]) == []
