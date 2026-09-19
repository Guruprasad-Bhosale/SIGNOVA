"""
Phase 14 Independent Double-Annotation & Agreement Unit Tests.
"""

from signova.annotation.schema import VideoAnnotation
from signova.operations.constants import (
    AGREEMENT_COMPUTABLE,
    AGREEMENT_INSUFFICIENT_SAMPLE,
    AGREEMENT_NOT_COMPUTABLE,
)
from signova.operations.double_annotation import DoubleAnnotationManager


def test_agreement_evaluates_independent_annotators_only():
    mgr = DoubleAnnotationManager(double_minimum=1)

    # 2 independent annotators for sample_01
    annots = {
        "sample_01": [
            VideoAnnotation("a1", "sample_01", "annot_user_A", False, ["NAMASTE", "HELP"]),
            VideoAnnotation("a2", "sample_01", "annot_user_B", False, ["NAMASTE", "WATER"]),
        ]
    }
    res = mgr.evaluate_independent_agreement(annots)
    assert res["agreement_status"] == AGREEMENT_COMPUTABLE
    assert res["independent_pairs_compared"] == 1
    assert res["mean_token_f1"] == 0.5


def test_reviewer_revisions_do_not_count_as_double_annotation():
    mgr = DoubleAnnotationManager()

    # Same annotator with reviewer note (not independent double annotation)
    annots = {
        "sample_02": [
            VideoAnnotation("a1", "sample_02", "annot_user_A", False, ["NAMASTE"]),
            VideoAnnotation("a2", "sample_02", "annot_user_A", False, ["NAMASTE", "EXTRA"]),
        ]
    }
    res = mgr.evaluate_independent_agreement(annots)
    assert res["agreement_status"] == AGREEMENT_NOT_COMPUTABLE
    assert res["independent_pairs_compared"] == 0
