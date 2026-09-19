"""
Phase 13 Reproducibility Unit Tests.
"""

import torch


def test_reproducibility():
    torch.manual_seed(1337)
    x1 = torch.randn(10, 10)
    torch.manual_seed(1337)
    x2 = torch.randn(10, 10)
    assert torch.equal(x1, x2)
