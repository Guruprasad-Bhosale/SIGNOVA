"""
Phase 21 CTC Feasibility Equation Tests.
"""

from signova.annotation.schema import VideoAnnotation
from signova.qualification.feasibility import (
    calculate_ctc_required_input_length,
    validate_ctc_feasibility,
)


def test_ctc_feasibility_repeated_tokens():
    # T_req = L + sum I(y_i == y_{i+1})
    # Glosses: ['HELLO', 'HELLO', 'WORLD'] -> L=3, repeats=1 -> T_req=4
    glosses = ["HELLO", "HELLO", "WORLD"]
    t_req = calculate_ctc_required_input_length(glosses)
    assert t_req == 4

    ann_pass = VideoAnnotation(
        annotation_id="a1",
        sample_id="s1",
        annotator_id="u1",
        is_temporally_aligned=True,
        provenance_id="p1",
        glosses=glosses,
        metadata={"frame_count": 64},
    )
    res_pass = validate_ctc_feasibility([ann_pass])
    assert res_pass.valid_samples == 1
    assert res_pass.rejected_samples == 0
    assert res_pass.feasibility_status == "PASSED"

    ann_fail = VideoAnnotation(
        annotation_id="a2",
        sample_id="s2",
        annotator_id="u1",
        is_temporally_aligned=True,
        provenance_id="p2",
        glosses=glosses,
        metadata={"frame_count": 2},
    )
    res_fail = validate_ctc_feasibility([ann_fail])
    assert res_fail.rejected_samples == 1
    assert res_fail.feasibility_status in ("WARNING", "FAILED")
