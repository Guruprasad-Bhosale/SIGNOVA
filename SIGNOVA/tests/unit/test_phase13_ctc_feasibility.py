"""
Phase 13 CTC Feasibility and Repeated Token Unit Tests.

Verifies:
- Feasibility calculates T_required = L + count(adjacent_identical_pairs)
- Repeated tokens [A, A], [A, B, A], [A, A, B] are correctly handled
"""

from signova.annotation.schema import VideoAnnotation
from signova.qualification.feasibility import (
    calculate_ctc_required_input_length,
    validate_ctc_feasibility,
)


def test_calculate_ctc_required_input_length_repeated_tokens():
    # [A, B] -> length 2, no adjacent repeats -> required 2
    assert calculate_ctc_required_input_length(["A", "B"]) == 2

    # [A, A] -> length 2, 1 adjacent repeat -> required 3 (A, blank, A)
    assert calculate_ctc_required_input_length(["A", "A"]) == 3

    # [A, B, A] -> length 3, 0 adjacent repeats -> required 3
    assert calculate_ctc_required_input_length(["A", "B", "A"]) == 3

    # [A, A, B] -> length 3, 1 adjacent repeat -> required 4 (A, blank, A, B)
    assert calculate_ctc_required_input_length(["A", "A", "B"]) == 4

    # [A, A, A] -> length 3, 2 adjacent repeats -> required 5 (A, blank, A, blank, A)
    assert calculate_ctc_required_input_length(["A", "A", "A"]) == 5


def test_ctc_feasibility_catches_repeated_token_length_violation():
    # Sequence [A, A] needs 3 frames. If input only has 2 frames, it must be rejected!
    annots = [
        VideoAnnotation(
            annotation_id="a1",
            sample_id="s1",
            annotator_id="u1",
            is_temporally_aligned=False,
            glosses=["A", "A"],
            metadata={"frame_count": 2},  # 2 frames < 3 required
        )
    ]
    rep = validate_ctc_feasibility(annots)
    assert rep.valid_samples == 0
    assert rep.rejected_samples == 1
    assert "CTC_LENGTH_VIOLATION" in rep.rejected_details[0]["reason"]
