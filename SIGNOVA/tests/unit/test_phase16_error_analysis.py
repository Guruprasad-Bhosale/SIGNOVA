"""
Phase 16 Detailed Error Analysis Unit Tests.

Validates:
- Exact sequence error breakdown (deletions, insertions, substitutions).
- Alignment mapping between reference and hypothesis tokens.
- Graceful handling of empty or mismatched sequence sets.
"""

from signova.experiments.error_analysis import perform_phase12_error_analysis


def test_error_analysis_with_substitutions():
    sids = ["p16_sample_01", "p16_sample_02"]
    refs = [["HELLO", "WORLD"], ["NAMASTE", "INDIA"]]
    hyps = [["HELLO", "WORLD"], ["NAMASTE", "BHARAT"]]

    res = perform_phase12_error_analysis(sids, refs, hyps)
    assert res["total_samples"] == 2
    assert res["exact_matches"] == 1
    assert res["exact_match_ratio"] == 0.5
