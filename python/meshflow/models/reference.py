"""Reference implementations for numerical correctness testing."""

import torch


def reference_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Reference linear layer: y = weight @ x + bias."""
    return torch.nn.functional.linear(x, weight, bias)


def reference_mlp(
    x: torch.Tensor,
    layers: list[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Reference MLP: chain of linear layers with ReLU activation between them."""
    for i, (W, b) in enumerate(layers):
        x = torch.nn.functional.linear(x, W, b)
        if i < len(layers) - 1:  # no activation on final layer
            x = torch.relu(x)
    return x
