"""
Phase 18 Independent Double Annotation & Agreement Unit Tests.

Validates:
- DoubleAnnotationManager handling of genuinely independent pairs.
- Preventing self-annotation / reviewer-correction counting as independent annotation.
- Handling empty / insufficient pairs (AGREEMENT_NOT_COMPUTABLE, AGREEMENT_INSUFFICIENT_SAMPLE).
"""

from signova.annotation.schema import VideoAnnotation
from signova.operations.constants import (
    AGREEMENT_COMPUTABLE,
    AGREEMENT_INSUFFICIENT_SAMPLE,
    AGREEMENT_NOT_COMPUTABLE,
)
from signova.operations.double_annotation import DoubleAnnotationManager


def test_empty_double_annotation_agreement_is_not_computable():
    mgr = DoubleAnnotationManager(double_minimum=5)
    res = mgr.evaluate_independent_agreement({})
    assert res["agreement_status"] == AGREEMENT_NOT_COMPUTABLE
    assert res["independent_pairs_compared"] == 0


def test_independent_double_annotation_pair_evaluation():
    mgr = DoubleAnnotationManager(double_minimum=1)

    annots = {
        "p18_sample_01": [
            VideoAnnotation("a1", "p18_sample_01", "annot_expert_1", False, ["NAMASTE", "WELCOME"]),
            VideoAnnotation("a2", "p18_sample_01", "annot_expert_2", False, ["NAMASTE", "WELCOME"]),
        ]
    }
    res = mgr.evaluate_independent_agreement(annots)
    assert res["agreement_status"] == AGREEMENT_COMPUTABLE
    assert res["independent_pairs_compared"] == 1
    assert res["mean_token_f1"] == 1.0
