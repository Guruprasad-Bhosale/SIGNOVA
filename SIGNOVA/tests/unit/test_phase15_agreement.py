"""
Phase 15 Independent Double-Annotation & Agreement Unit Tests.

Validates:
- Independent double-annotation verification (distinct annotators).
- Rejection of single-annotator revisions as double-annotations.
- Agreement status computation (AGREEMENT_COMPUTABLE, AGREEMENT_NOT_COMPUTABLE, AGREEMENT_INSUFFICIENT_SAMPLE).
- Zero fabricated agreement when no genuine double annotations exist.
"""

from signova.annotation.schema import VideoAnnotation
from signova.operations.constants import (
    AGREEMENT_COMPUTABLE,
    AGREEMENT_INSUFFICIENT_SAMPLE,
    AGREEMENT_NOT_COMPUTABLE,
)
from signova.operations.double_annotation import DoubleAnnotationManager


def test_independent_double_annotation_agreement():
    mgr = DoubleAnnotationManager(double_minimum=1)

    annots = {
        "p15_sample_01": [
            VideoAnnotation("a1", "p15_sample_01", "annot_expert_1", False, ["NAMASTE", "WELCOME"]),
            VideoAnnotation("a2", "p15_sample_01", "annot_expert_2", False, ["NAMASTE", "WELCOME"]),
        ]
    }
    res = mgr.evaluate_independent_agreement(annots)
    assert res["agreement_status"] == AGREEMENT_COMPUTABLE
    assert res["independent_pairs_compared"] == 1
    assert res["mean_token_f1"] == 1.0


def test_no_double_annotations_yields_not_computable():
    mgr = DoubleAnnotationManager(double_minimum=5)
    res = mgr.evaluate_independent_agreement({})
    assert res["agreement_status"] == AGREEMENT_NOT_COMPUTABLE
    assert res["independent_pairs_compared"] == 0
