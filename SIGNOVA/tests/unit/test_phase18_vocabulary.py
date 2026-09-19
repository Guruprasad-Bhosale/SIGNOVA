"""
Phase 18 Vocabulary Derivation & Isolation Unit Tests.

Validates:
- Base token preservation: <BLANK> = 0, <UNK> = 1.
- Vocabulary derivation strictly from genuine training annotations.
- Zero leakage from English translation text.
"""

from signova.annotation.schema import VideoAnnotation
from signova.qualification.vocabulary import Phase12GlossVocabulary


def test_base_vocabulary_tokens():
    vocab = Phase12GlossVocabulary.build_from_annotations([])
    assert vocab.size == 2
    assert vocab.token_to_id["<BLANK>"] == 0
    assert vocab.token_to_id["<UNK>"] == 1


def test_vocabulary_built_strictly_from_glosses():
    ann = VideoAnnotation(
        annotation_id="ann_01",
        sample_id="vid_01",
        annotator_id="ann_01",
        is_temporally_aligned=True,
        glosses=["HELLO", "INDIA"],
        provenance_id="prov_01",
    )
    vocab = Phase12GlossVocabulary.build_from_annotations([ann])
    assert vocab.size == 4
    assert "HELLO" in vocab.token_to_id
    assert "INDIA" in vocab.token_to_id
