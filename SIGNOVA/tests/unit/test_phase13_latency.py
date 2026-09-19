"""
Phase 13 Latency Profiling Unit Tests.
"""

from signova.experiments.latency import profile_pipeline_latency
from signova.experiments.trainer import ContinuousBiGRUCTCModel


def test_latency_profiler():
    model = ContinuousBiGRUCTCModel(input_dim=150, hidden_dim=32, num_layers=1, num_classes=5)
    res = profile_pipeline_latency(model, input_dim=150, num_frames=30, vocab_size=5)

    assert "profiles" in res
    assert "OFFLINE" in res["profiles"]
    assert "WINDOWED" in res["profiles"]
    assert "ROLLING_STREAM" in res["profiles"]
    assert "END_TO_END" in res["profiles"]
