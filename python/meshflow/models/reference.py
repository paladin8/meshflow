"""Reference implementations for numerical correctness testing."""

import torch


def reference_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Reference linear layer: y = weight @ x + bias."""
    return torch.nn.functional.linear(x, weight, bias)


def reference_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference RMSNorm: x / sqrt(mean(x^2) + eps) * gamma.

    For multi-position input (2D), normalizes each position independently.
    """
    if x.dim() == 1:
        rms = torch.sqrt(torch.mean(x * x) + eps)
        return x / rms * gamma
    # (num_positions, d_model)
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * gamma


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
