"""
Phase 17 Vocabulary and OOV Management Unit Tests.

Validates:
- Vocabulary creation exclusively from training split annotations.
- Prevention of vocabulary leakage from validation or test splits.
- Encoding/decoding with special tokens (<blank>=0, <UNK>=1).
- Out-of-vocabulary (OOV) mapping and recovery.
"""

from signova.annotation.schema import VideoAnnotation
from signova.qualification.vocabulary import Phase12GlossVocabulary


def test_vocab_built_from_train_split_only():
    train_ann = VideoAnnotation(
        annotation_id="p17_a1",
        sample_id="p17_s1",
        annotator_id="u1",
        is_temporally_aligned=False,
        glosses=["HELLO", "INDIA"],
        dataset_split="train",
    )
    val_ann = VideoAnnotation(
        annotation_id="p17_a2",
        sample_id="p17_s2",
        annotator_id="u1",
        is_temporally_aligned=False,
        glosses=["TEST_ONLY_SIGN"],
        dataset_split="val",
    )

    vocab = Phase12GlossVocabulary.build_from_annotations([train_ann, val_ann])

    assert "HELLO" in vocab.token_to_id
    assert "INDIA" in vocab.token_to_id
    assert "TEST_ONLY_SIGN" not in vocab.token_to_id

    # OOV handling
    encoded = vocab.encode(["TEST_ONLY_SIGN"])
    assert encoded == [1]
    decoded = vocab.decode(encoded, remove_blank=True)
    assert decoded == ["<UNK>"]
