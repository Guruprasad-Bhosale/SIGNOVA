"""
Phase 13 Error Analysis Unit Tests.
"""

from signova.experiments.error_analysis import perform_phase12_error_analysis


def test_error_analysis():
    sids = ["sample_01"]
    refs = [["A", "B", "C"]]
    hyps = [["A", "X", "C"]]

    res = perform_phase12_error_analysis(sids, refs, hyps)
    assert res["total_samples"] == 1
    assert res["exact_matches"] == 0
    assert res["sample_breakdown"][0]["substitutions"] == 1
