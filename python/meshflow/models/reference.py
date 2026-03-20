"""Reference implementations for numerical correctness testing."""

import torch


def reference_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Reference linear layer: y = weight @ x + bias."""
    return torch.nn.functional.linear(x, weight, bias)
