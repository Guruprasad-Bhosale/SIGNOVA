"""
Unit tests for Phase 11 Versioned Gloss Vocabulary Manager and Reserved Tokens.
"""

from pathlib import Path
import pytest
from signova.annotation.schema import VideoAnnotation
from signova.annotation.vocabulary import ProjectGlossVocabulary


def test_vocabulary_reserved_tokens():
    vocab = ProjectGlossVocabulary()
    assert len(vocab) == 2
    assert vocab.get_token(0) == "<BLANK>"
    assert vocab.get_token(1) == "<UNK>"
    assert vocab.get_token_id("<BLANK>") == 0
    assert vocab.get_token_id("<UNK>") == 1
    assert vocab.get_token_id("NON_EXISTENT_TOKEN") == 1


def test_vocabulary_add_annotation_tokens(tmp_path):
    vocab = ProjectGlossVocabulary(version="0.1.0")

    annot = VideoAnnotation(
        annotation_id="a_01",
        sample_id="s_01",
        annotator_id="u_01",
        is_temporally_aligned=False,
        glosses=["NAMASTE", "COLLEGE", "FS-DELHI"],
    )

    new_tokens = vocab.add_annotation_tokens(annot)
    assert new_tokens == ["NAMASTE", "COLLEGE", "FS-DELHI"]
    assert len(vocab) == 5  # <BLANK>, <UNK>, NAMASTE, COLLEGE, FS-DELHI

    assert vocab.get_token_id("NAMASTE") == 2
    assert vocab.get_token(2) == "NAMASTE"

    # Test serialization round-trip
    vocab_file = tmp_path / "vocab.json"
    vocab.save_to_file(vocab_file)

    loaded_vocab = ProjectGlossVocabulary.load_from_file(vocab_file)
    assert len(loaded_vocab) == 5
    assert loaded_vocab.get_token_id("FS-DELHI") == 4
