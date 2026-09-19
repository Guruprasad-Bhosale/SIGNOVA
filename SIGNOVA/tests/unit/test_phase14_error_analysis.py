"""
Phase 14 Error Analysis Unit Tests.
"""

from signova.experiments.error_analysis import perform_phase12_error_analysis


def test_error_analysis_breakdown():
    sids = ["test_s1"]
    refs = [["HELLO", "WORLD"]]
    hyps = [["HELLO", "WORLD"]]

    res = perform_phase12_error_analysis(sids, refs, hyps)
    assert res["total_samples"] == 1
    assert res["exact_matches"] == 1
    assert res["exact_match_ratio"] == 1.0
