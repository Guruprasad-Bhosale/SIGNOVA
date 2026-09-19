"""
Phase 19 CTC Feasibility & Repeated-Token Formula Unit Tests.

Validates:
- Exact repeated-token mathematical formula:
    T_required = L + sum_{i=1}^{L-1} I(y_i == y_{i+1})
- Mandatory test vectors:
    [A, B, C] -> 3
    [A, A] -> 3
    [A, A, B] -> 4
    [A, A, A] -> 5
    [A, B, A] -> 3
- Dataset-level feasibility evaluation and rate calculation.
"""

from signova.annotation.schema import VideoAnnotation
from signova.operations.phase19_orchestrator import Phase19Orchestrator
from signova.qualification.feasibility import calculate_ctc_required_input_length


def test_repeated_token_formula_mandatory_test_vectors():
    assert calculate_ctc_required_input_length(["A", "B", "C"]) == 3
    assert calculate_ctc_required_input_length(["A", "A"]) == 3
    assert calculate_ctc_required_input_length(["A", "A", "B"]) == 4
    assert calculate_ctc_required_input_length(["A", "A", "A"]) == 5
    assert calculate_ctc_required_input_length(["A", "B", "A"]) == 3


def test_dataset_level_ctc_feasibility_evaluation():
    ann1 = VideoAnnotation(
        annotation_id="ann_01",
        sample_id="vid_01",
        annotator_id="ann_user",
        is_temporally_aligned=True,
        provenance_id="prov_01",
        glosses=["HELLO", "WORLD"],
    )
    ann2 = VideoAnnotation(
        annotation_id="ann_02",
        sample_id="vid_02",
        annotator_id="ann_user",
        is_temporally_aligned=True,
        provenance_id="prov_01",
        glosses=["A", "A", "A"],
    )

    orch = Phase19Orchestrator()
    summary = orch._evaluate_dataset_ctc_feasibility([ann1, ann2], features_dir=None)

    assert summary.total_sequences == 2
    assert summary.feasible_sequences == 2
    assert summary.infeasible_sequences == 0
    assert summary.feasibility_rate == 1.0
    assert summary.max_required_frames == 5
    assert summary.min_gloss_length == 2
    assert summary.max_gloss_length == 3
