"""
Phase 21 Genuine Vocabulary Generation Tests.
"""

from signova.annotation.schema import VideoAnnotation
from signova.qualification.constants import BLANK_ID, BLANK_TOKEN, UNK_ID, UNK_TOKEN
from signova.qualification.vocabulary import Phase12GlossVocabulary


def test_vocabulary_token_contract():
    ann = VideoAnnotation(
        annotation_id="ann_01",
        sample_id="s1",
        annotator_id="u1",
        is_temporally_aligned=True,
        provenance_id="prov_01",
        glosses=["NAMASTE", "THANK_YOU"],
    )
    vocab = Phase12GlossVocabulary.build_from_annotations([ann])

    assert vocab.token_to_id[BLANK_TOKEN] == BLANK_ID
    assert vocab.token_to_id[UNK_TOKEN] == UNK_ID
    assert "NAMASTE" in vocab.token_to_id
    assert "THANK_YOU" in vocab.token_to_id

    encoded = vocab.encode(["NAMASTE", "UNKNOWN_SIGN"])
    assert encoded[0] == vocab.token_to_id["NAMASTE"]
    assert encoded[1] == UNK_ID
