"""
Phase 14 Real Vocabulary & OOV Unit Tests.
"""

from signova.annotation.schema import VideoAnnotation
from signova.qualification.vocabulary import Phase12GlossVocabulary


def test_vocabulary_build_and_oov_handling():
    annots = [
        VideoAnnotation("a1", "s1", "u1", False, ["GREETING", "NAME"], dataset_split="train"),
        VideoAnnotation("a2", "s2", "u1", False, ["GREETING", "UNKNOWN_SIGN"], dataset_split="test"),
    ]
    vocab = Phase12GlossVocabulary.build_from_annotations(annots)

    assert "GREETING" in vocab.token_to_id
    assert "NAME" in vocab.token_to_id
    # Test token must NOT leak into train vocabulary
    assert "UNKNOWN_SIGN" not in vocab.token_to_id

    # OOV mapped to <UNK>
    encoded = vocab.encode(["UNKNOWN_SIGN"])
    assert encoded == [1]
    decoded = vocab.decode(encoded, remove_blank=True)
    assert decoded == ["<UNK>"]
