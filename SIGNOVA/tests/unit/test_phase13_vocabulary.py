"""
Phase 13 Vocabulary Unit Tests.
"""

from signova.annotation.schema import VideoAnnotation
from signova.qualification.vocabulary import Phase12GlossVocabulary


def test_vocabulary_preserves_blank_and_unk():
    vocab = Phase12GlossVocabulary()
    assert vocab.encode(["UNKNOWN_TOKEN"]) == [1]
    assert vocab.decode([0, 1], remove_blank=True) == ["<UNK>"]
