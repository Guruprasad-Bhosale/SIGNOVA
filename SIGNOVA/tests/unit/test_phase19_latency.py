"""
Phase 19 Latency Profiling Unit Tests.

Validates:
- Standardized latency profiling across OFFLINE, CAUSAL_STREAMING, BATCH, and END_TO_END execution modes.
- Output schema integrity.
"""

from signova.experiments.latency import profile_pipeline_latency
from signova.experiments.trainer import ContinuousBiGRUCTCModel


def test_probe_model_latency_profiling():
    model = ContinuousBiGRUCTCModel(input_dim=150, hidden_dim=32, num_layers=1, num_classes=5)
    res = profile_pipeline_latency(model, input_dim=150, num_frames=25, vocab_size=5)

    assert "profiles" in res
    profiles = res["profiles"]
    assert "OFFLINE" in profiles
    assert "ROLLING_STREAM" in profiles
    assert "WINDOWED" in profiles
    assert "END_TO_END" in profiles

    for mode in ["OFFLINE", "WINDOWED", "ROLLING_STREAM", "END_TO_END"]:
        assert profiles[mode]["total_latency_ms"] >= 0.0
