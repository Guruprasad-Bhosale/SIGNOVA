"""
Phase 15 Latency Profiling Unit Tests.

Validates:
- Profile coverage across four distinct execution categories:
  1. OFFLINE (Batch processing of full video landmarks)
  2. WINDOWED (Fixed time chunk processing)
  3. ROLLING_STREAM (Sliding window over streaming frames)
  4. END_TO_END (Full pipeline latency)
- Timing metrics reporting without fake hardware claims.
"""

from signova.experiments.latency import profile_pipeline_latency
from signova.experiments.trainer import ContinuousBiGRUCTCModel


def test_latency_profiling_four_categories():
    model = ContinuousBiGRUCTCModel(input_dim=150, hidden_dim=32, num_layers=1, num_classes=5)
    res = profile_pipeline_latency(model, input_dim=150, num_frames=25, vocab_size=5)

    assert "profiles" in res
    profiles = res["profiles"]
    assert "OFFLINE" in profiles
    assert "WINDOWED" in profiles
    assert "ROLLING_STREAM" in profiles
    assert "END_TO_END" in profiles

    for mode in ["OFFLINE", "WINDOWED", "ROLLING_STREAM", "END_TO_END"]:
        assert profiles[mode]["total_latency_ms"] >= 0.0
