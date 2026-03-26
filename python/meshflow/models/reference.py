"""Reference implementations for numerical correctness testing."""

from typing import Any

import numpy as np
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


def _to_tensor(w: Any) -> torch.Tensor:
    """Convert numpy array or tensor to torch.Tensor."""
    if isinstance(w, np.ndarray):
        return torch.from_numpy(w)
    return w


def reference_transformer_block(
    x: torch.Tensor,
    weights: dict[str, dict[str, Any]],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference single-head transformer block.

    Args:
        x: Input tensor of shape (seq_len, d_model).
        weights: Dict mapping node IDs to weight dicts (numpy or torch).
        eps: RMSNorm epsilon.

    Returns:
        Output tensor of shape (seq_len, d_model).
    """
    # Sublayer 1: pre-norm attention
    normed1 = reference_rmsnorm(x, _to_tensor(weights["rn1"]["gamma"]), eps)

    Q = reference_linear(
        normed1, _to_tensor(weights["q_proj"]["weight"]), _to_tensor(weights["q_proj"]["bias"])
    )
    K = reference_linear(
        normed1, _to_tensor(weights["k_proj"]["weight"]), _to_tensor(weights["k_proj"]["bias"])
    )
    V = reference_linear(
        normed1, _to_tensor(weights["v_proj"]["weight"]), _to_tensor(weights["v_proj"]["bias"])
    )

    scores = Q @ K.T  # (seq_len, seq_len)
    attn_weights = torch.softmax(scores, dim=-1)
    attn_out = attn_weights @ V  # (seq_len, d_model)

    proj_out = reference_linear(
        attn_out,
        _to_tensor(weights["out_proj"]["weight"]),
        _to_tensor(weights["out_proj"]["bias"]),
    )
    x = x + proj_out  # residual add1

    # Sublayer 2: pre-norm FFN
    normed2 = reference_rmsnorm(x, _to_tensor(weights["rn2"]["gamma"]), eps)

    ffn1_out = reference_linear(
        normed2, _to_tensor(weights["ffn1"]["weight"]), _to_tensor(weights["ffn1"]["bias"])
    )
    ffn1_relu = torch.relu(ffn1_out)
    ffn2_out = reference_linear(
        ffn1_relu, _to_tensor(weights["ffn2"]["weight"]), _to_tensor(weights["ffn2"]["bias"])
    )
    x = x + ffn2_out  # residual add2

    return x
