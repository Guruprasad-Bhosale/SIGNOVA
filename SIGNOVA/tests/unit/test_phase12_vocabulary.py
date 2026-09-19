"""
Phase 12 Real Vocabulary Unit Tests.
"""

from signova.annotation.schema import VideoAnnotation
from signova.qualification.constants import BLANK_ID, BLANK_TOKEN, UNK_ID, UNK_TOKEN
from signova.qualification.vocabulary import Phase12GlossVocabulary


def test_vocabulary_preserves_reserved_tokens():
    vocab = Phase12GlossVocabulary()
    assert vocab.token_to_id[BLANK_TOKEN] == BLANK_ID
    assert vocab.token_to_id[UNK_TOKEN] == UNK_ID
    assert vocab.id_to_token[BLANK_ID] == BLANK_TOKEN
    assert vocab.id_to_token[UNK_ID] == UNK_TOKEN


def test_vocabulary_build_from_annotations_and_encode():
    annots = [
        VideoAnnotation(
            annotation_id="a1",
            sample_id="s1",
            annotator_id="u1",
            is_temporally_aligned=False,
            glosses=["HELLO", "WORLD"],
            dataset_split="train",
            training_eligible=True,
        ),
        VideoAnnotation(
            annotation_id="a2",
            sample_id="s2",
            annotator_id="u1",
            is_temporally_aligned=False,
            glosses=["HELLO", "AGAIN"],
            dataset_split="train",
            training_eligible=True,
        ),
        VideoAnnotation(
            annotation_id="a3",
            sample_id="s3",
            annotator_id="u1",
            is_temporally_aligned=False,
            glosses=["UNSEEN_TEST"],
            dataset_split="test",
            training_eligible=True,
        ),
    ]

    vocab = Phase12GlossVocabulary.build_from_annotations(annots)
    assert "HELLO" in vocab.token_to_id
    assert "WORLD" in vocab.token_to_id
    assert "AGAIN" in vocab.token_to_id
    # Test-only gloss must NOT leak into train vocabulary
    assert "UNSEEN_TEST" not in vocab.token_to_id

    # Encoding unseen token returns UNK_ID
    encoded = vocab.encode(["HELLO", "UNSEEN_TEST"])
    assert encoded[0] == vocab.token_to_id["HELLO"]
    assert encoded[1] == UNK_ID

    # Decoding maps UNK_ID back to UNK_TOKEN
    decoded = vocab.decode(encoded)
    assert decoded == ["HELLO", UNK_TOKEN]

    # Check singleton detection
    singletons = vocab.get_singletons()
    assert "WORLD" in singletons
    assert "AGAIN" in singletons
    assert "HELLO" not in singletons
