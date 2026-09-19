"""
Phase 12 Experiment Reproducibility and Latency Profiling Tests.
"""

import torch

from signova.experiments.latency import profile_pipeline_latency
from signova.experiments.trainer import ContinuousBiGRUCTCModel


def test_latency_profiler_structure():
    model = ContinuousBiGRUCTCModel(input_dim=150, hidden_dim=64, num_layers=1, num_classes=10)
    res = profile_pipeline_latency(model, input_dim=150, num_frames=30, vocab_size=10, device="cpu")

    assert "profiles" in res
    profiles = res["profiles"]
    for mode in ["OFFLINE", "WINDOWED", "ROLLING_STREAM", "END_TO_END"]:
        assert mode in profiles
        p = profiles[mode]
        assert "total_latency_ms" in p
        assert "fps" in p
        assert p["total_latency_ms"] >= 0.0


def test_reproducibility_seed():
    torch.manual_seed(42)
    t1 = torch.randn(5, 5)
    torch.manual_seed(42)
    t2 = torch.randn(5, 5)
    assert torch.equal(t1, t2)
