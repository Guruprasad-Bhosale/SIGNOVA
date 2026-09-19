"""
Phase 17 Exact CTC Feasibility & Repeated Token Unit Tests.

Validates the mathematical invariant:
T_required = L + sum_{i=1}^{L-1} I(y_i == y_{i+1})
where consecutive identical tokens require an intervening blank frame for CTC separation.
"""

from signova.qualification.feasibility import calculate_ctc_required_input_length


def test_exact_ctc_feasibility_single_and_distinct_tokens():
    # Empty sequence
    assert calculate_ctc_required_input_length([]) == 0
    # Single token -> L = 1, repeats = 0 -> T = 1
    assert calculate_ctc_required_input_length(["NAMASTE"]) == 1
    # Distinct tokens -> L = 3, repeats = 0 -> T = 3
    assert calculate_ctc_required_input_length(["HELLO", "MY", "NAME"]) == 3


def test_exact_ctc_feasibility_with_repeated_tokens():
    # [A, A] -> L = 2, repeats = 1 -> T = 3 (A, blank, A)
    assert calculate_ctc_required_input_length(["A", "A"]) == 3

    # [A, A, B] -> L = 3, repeats = 1 -> T = 4 (A, blank, A, B)
    assert calculate_ctc_required_input_length(["A", "A", "B"]) == 4

    # [A, B, A] -> L = 3, repeats = 0 -> T = 3 (non-adjacent identical tokens require no blank)
    assert calculate_ctc_required_input_length(["A", "B", "A"]) == 3

    # [A, A, A] -> L = 3, repeats = 2 -> T = 5 (A, blank, A, blank, A)
    assert calculate_ctc_required_input_length(["A", "A", "A"]) == 5

    # [A, A, B, B] -> L = 4, repeats = 2 -> T = 6
    assert calculate_ctc_required_input_length(["A", "A", "B", "B"]) == 6
