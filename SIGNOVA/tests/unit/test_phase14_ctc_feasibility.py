"""
Phase 14 Exact CTC Feasibility & Repeated Token Unit Tests.
"""

from signova.qualification.feasibility import calculate_ctc_required_input_length


def test_exact_ctc_feasibility_with_repeated_tokens():
    # [A, B, C] -> minimum T = 3
    assert calculate_ctc_required_input_length(["A", "B", "C"]) == 3

    # [A, A] -> minimum T = 3 (A, blank, A)
    assert calculate_ctc_required_input_length(["A", "A"]) == 3

    # [A, A, B] -> minimum T = 4 (A, blank, A, B)
    assert calculate_ctc_required_input_length(["A", "A", "B"]) == 4

    # [A, B, A] -> minimum T = 3 (A, B, A - non-adjacent identical signs do not require blank)
    assert calculate_ctc_required_input_length(["A", "B", "A"]) == 3

    # [A, A, A] -> minimum T = 5 (A, blank, A, blank, A)
    assert calculate_ctc_required_input_length(["A", "A", "A"]) == 5
