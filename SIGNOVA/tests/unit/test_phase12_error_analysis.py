"""
Phase 12 Error Analysis Unit Tests.
"""

from signova.experiments.error_analysis import perform_phase12_error_analysis


def test_phase12_error_analysis():
    sample_ids = ["s1", "s2"]
    refs = [["HELLO", "WORLD"], ["I", "NEED", "HELP"]]
    hyps = [["HELLO", "WORLD"], ["I", "WANT", "HELP", "NOW"]]

    analysis = perform_phase12_error_analysis(sample_ids, refs, hyps)

    assert analysis["total_samples"] == 2
    assert analysis["exact_matches"] == 1
    assert analysis["exact_match_ratio"] == 0.5
    assert len(analysis["sample_breakdown"]) == 2

    # Sample 2 breakdown
    s2_breakdown = analysis["sample_breakdown"][1]
    assert s2_breakdown["sample_id"] == "s2"
    assert s2_breakdown["exact_match"] is False
    assert s2_breakdown["substitutions"] == 1
    assert s2_breakdown["insertions"] == 1
