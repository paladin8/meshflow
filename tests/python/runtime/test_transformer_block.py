"""End-to-end tests for the full transformer block."""

import numpy as np
import torch

from meshflow._mesh_runtime import run_program
from meshflow.compiler import CompilerConfig, compile
from meshflow.compiler.artifact import serialize
from meshflow.models.reference import reference_transformer_block
from meshflow.models.transformer import transformer_block, transformer_weights


class TestTransformerBlock:
    """Full transformer block: RMSNorm + attention + FFN + residual connections."""

    def test_residual_passthrough(self) -> None:
        """Zero projections/FFN → residual connections pass input through unchanged.

        With all-zero LINEAR weights and biases:
        - RMSNorm1(x) normalizes, but Q/K/V proj all produce 0
        - Attention output = 0, out_proj(0) = 0
        - Add1 = x + 0 = x (residual preserves input)
        - Same for FFN sublayer: Add2 = x + 0 = x
        """
        seq_len = 2
        d_model = 4
        d_ff = 8
        eps = 1e-6

        graph = transformer_block(seq_len, d_model, d_ff, eps)

        # All-zero weights → all LINEAR outputs are zero
        weights: dict[str, dict[str, np.ndarray]] = {
            "rn1": {"gamma": np.ones(d_model, dtype=np.float32)},
            "q_proj": {
                "weight": np.zeros((d_model, d_model), dtype=np.float32),
                "bias": np.zeros(d_model, dtype=np.float32),
            },
            "k_proj": {
                "weight": np.zeros((d_model, d_model), dtype=np.float32),
                "bias": np.zeros(d_model, dtype=np.float32),
            },
            "v_proj": {
                "weight": np.zeros((d_model, d_model), dtype=np.float32),
                "bias": np.zeros(d_model, dtype=np.float32),
            },
            "out_proj": {
                "weight": np.zeros((d_model, d_model), dtype=np.float32),
                "bias": np.zeros(d_model, dtype=np.float32),
            },
            "rn2": {"gamma": np.ones(d_model, dtype=np.float32)},
            "ffn1": {
                "weight": np.zeros((d_ff, d_model), dtype=np.float32),
                "bias": np.zeros(d_ff, dtype=np.float32),
            },
            "ffn2": {
                "weight": np.zeros((d_model, d_ff), dtype=np.float32),
                "bias": np.zeros(d_model, dtype=np.float32),
            },
        }

        config = CompilerConfig(mesh_height=6)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)

        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # 2 positions × 4 features
        result = run_program(artifact_bytes, inputs={"input": x})

        output = None
        for data in result.outputs.values():
            output = data
        assert output is not None, f"no output found, outputs={dict(result.outputs)}"

        actual = torch.tensor(output)
        expected = torch.tensor(x)
        assert torch.allclose(actual, expected, atol=1e-4), (
            f"residual passthrough failed: got {actual} vs expected {expected}"
        )

    def test_basic_with_torch_validation(self) -> None:
        """seq_len=4, d_model=8, d_ff=16 — full block compared against torch reference."""
        seq_len = 4
        d_model = 8
        d_ff = 16
        eps = 1e-6

        graph = transformer_block(seq_len, d_model, d_ff, eps)
        weights = transformer_weights(d_model, d_ff, seed=42)

        config = CompilerConfig(mesh_height=6)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)

        torch.manual_seed(99)
        x = torch.randn(seq_len, d_model)

        result = run_program(artifact_bytes, inputs={"input": x.flatten().tolist()})

        output = None
        for data in result.outputs.values():
            output = data
        assert output is not None, f"no output found, outputs={dict(result.outputs)}"

        expected = reference_transformer_block(x, weights, eps)
        actual = torch.tensor(output).reshape(seq_len, d_model)
        assert torch.allclose(actual, expected, atol=1e-3), (
            f"transformer block mismatch:\ngot {actual}\nvs expected {expected}\n"
            f"diff={actual - expected}"
        )

    def test_small_dimensions(self) -> None:
        """seq_len=2, d_model=4, d_ff=8 — smaller dimensions for fast debugging."""
        seq_len = 2
        d_model = 4
        d_ff = 8
        eps = 1e-6

        graph = transformer_block(seq_len, d_model, d_ff, eps)
        weights = transformer_weights(d_model, d_ff, seed=7)

        config = CompilerConfig(mesh_height=6)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)

        torch.manual_seed(123)
        x = torch.randn(seq_len, d_model)

        result = run_program(artifact_bytes, inputs={"input": x.flatten().tolist()})

        output = None
        for data in result.outputs.values():
            output = data
        assert output is not None, f"no output found, outputs={dict(result.outputs)}"

        expected = reference_transformer_block(x, weights, eps)
        actual = torch.tensor(output).reshape(seq_len, d_model)
        assert torch.allclose(actual, expected, atol=1e-3), (
            f"transformer block mismatch:\ngot {actual}\nvs expected {expected}\n"
            f"diff={actual - expected}"
        )

    def test_non_divisible_dimensions(self) -> None:
        """seq_len=3, d_model=7, d_ff=11, mesh_height=5 — nothing divides evenly.

        Exercises remainder logic in tile distribution:
        - LINEAR (d_model=7, 4 tiles): rows [2,2,2,1]
        - LINEAR (d_ff=11, 4 tiles): rows [3,3,3,2]
        - RMSNorm (feature_count=7, 3 tiles): slices [3,2,2]
        - Attention: 3 PEs (odd seq_len)
        - ConcatCollect: uneven fragment_rows per tile
        """
        seq_len = 3
        d_model = 7
        d_ff = 11
        eps = 1e-6

        graph = transformer_block(seq_len, d_model, d_ff, eps)
        weights = transformer_weights(d_model, d_ff, seed=13)

        config = CompilerConfig(mesh_height=5)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)

        torch.manual_seed(77)
        x = torch.randn(seq_len, d_model)

        result = run_program(artifact_bytes, inputs={"input": x.flatten().tolist()})

        output = None
        for data in result.outputs.values():
            output = data
        assert output is not None, f"no output found, outputs={dict(result.outputs)}"

        expected = reference_transformer_block(x, weights, eps)
        actual = torch.tensor(output).reshape(seq_len, d_model)
        assert torch.allclose(actual, expected, atol=1e-3), (
            f"transformer block mismatch:\ngot {actual}\nvs expected {expected}\n"
            f"diff={actual - expected}"
        )
