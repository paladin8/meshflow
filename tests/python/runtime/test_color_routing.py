"""Tests for color-based routing and fabric bandwidth.

Verifies that:
- link_contentions counter is exposed and correct
- Messages sharing a physical link queue behind each other (colors share bandwidth)
- Transformer block timing is reasonable with wormhole routing
- Color profiling fields (max_colors_per_link, total_colors_used) are populated
"""

import torch

from meshflow._mesh_runtime import (
    MeshConfig,
    SimInput,
    TaskKind,
    run_program,
    run_simulation,
)
from meshflow.compiler import CompilerConfig, compile
from meshflow.compiler.artifact import serialize
from meshflow.models.transformer import transformer_block, transformer_weights

WEIGHTS_SEED = 42
INPUT_SEED = 99


class TestColorContentionsExposure:
    """Verify link_contentions is accessible from Python."""

    def test_link_contentions_accessible(self) -> None:
        """The link_contentions field exists and defaults to 0 for empty sim."""
        cfg = MeshConfig(width=2, height=2)
        inp = SimInput()
        result = run_simulation(config=cfg, inputs=inp)
        assert hasattr(result, "link_contentions")
        assert result.link_contentions == 0

    def test_link_contentions_zero_for_simple_message(self) -> None:
        """A single message with no color conflicts has 0 contentions."""
        cfg = MeshConfig(width=4, height=4)
        inp = SimInput()
        inp.add_message(source=(0, 0), dest=(2, 0), payload=[1.0, 2.0])
        inp.add_task(coord=(2, 0), kind=TaskKind.CollectOutput, trigger_slot=0)
        result = run_simulation(config=cfg, inputs=inp)
        assert result.link_contentions == 0

    def test_contention_detected_same_color_same_link(self) -> None:
        """Two same-color (color 0) messages on the same link cause contention."""
        cfg = MeshConfig(width=2, height=1)
        inp = SimInput()
        # Both go (0,0) -> (1,0), both color 0 (default)
        inp.add_message(source=(0, 0), dest=(1, 0), payload=[1.0])
        inp.add_message(source=(0, 0), dest=(1, 0), payload=[2.0])
        inp.add_task(coord=(1, 0), kind=TaskKind.CollectOutput, trigger_slot=0)
        result = run_simulation(config=cfg, inputs=inp)
        assert result.link_contentions > 0, (
            "Two same-color messages on the same link at the same time should cause contention"
        )


class TestParallelSendBehavior:
    """Verify parallel send semantics via compiled artifacts."""

    def test_transformer_block_final_timestamp_reasonable(self) -> None:
        """Transformer block compiles and runs with wormhole routing.

        With M11 wormhole routing, timestamps are payload-proportional.
        """
        graph = transformer_block(4, 8, 16)
        weights = transformer_weights(8, 16, seed=WEIGHTS_SEED)
        config = CompilerConfig(mesh_height=6)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(4, 8)
        result = run_program(artifact_bytes, inputs={"input": x.flatten().tolist()})

        assert result.final_timestamp > 0
        assert result.final_timestamp <= 1400  # M12P2: fused RMSNorm (~1297)

    def test_small_transformer_link_contentions(self) -> None:
        """Small transformer block link contentions are bounded.

        With M11 shared-bandwidth fabric, colors share physical link bandwidth,
        so link contentions are expected when multiple messages traverse the
        same link.
        """
        graph = transformer_block(4, 8, 16)
        weights = transformer_weights(8, 16, seed=WEIGHTS_SEED)
        config = CompilerConfig(mesh_height=6)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(4, 8)
        result = run_program(artifact_bytes, inputs={"input": x.flatten().tolist()})

        assert result.link_contentions <= 60, (
            f"link_contentions={result.link_contentions} exceeds threshold"
        )

    def test_medium_transformer_link_contentions(self) -> None:
        """Medium transformer block link contentions are bounded."""
        graph = transformer_block(8, 16, 32)
        weights = transformer_weights(16, 32, seed=WEIGHTS_SEED)
        config = CompilerConfig(mesh_height=8)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(8, 16)
        result = run_program(artifact_bytes, inputs={"input": x.flatten().tolist()})

        assert result.link_contentions <= 120, (
            f"link_contentions={result.link_contentions} exceeds threshold"
        )


class TestNumericalCorrectness:
    """Verify parallel sends don't change numerical results."""

    def test_small_transformer_outputs_match_reference(self) -> None:
        """Outputs must match Python reference implementation."""
        from meshflow.models.reference import reference_transformer_block

        seq_len, d_model, d_ff = 4, 8, 16
        eps = 1e-6

        graph = transformer_block(seq_len, d_model, d_ff, eps)
        weights = transformer_weights(d_model, d_ff, seed=WEIGHTS_SEED)
        config = CompilerConfig(mesh_height=6)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(seq_len, d_model)
        result = run_program(artifact_bytes, inputs={"input": x.flatten().tolist()})

        # Get simulator output
        output = None
        for data in result.outputs.values():
            output = data
        assert output is not None, f"no output found, outputs={dict(result.outputs)}"

        # Reference computation
        expected = reference_transformer_block(x, weights, eps)
        actual = torch.tensor(output).reshape(seq_len, d_model)

        assert torch.allclose(actual, expected, atol=1e-3), (
            f"transformer block mismatch with parallel sends:\n"
            f"got {actual}\nvs expected {expected}\n"
            f"diff={actual - expected}"
        )


class TestColorProfilingFields:
    """Phase 4: Verify color profiling fields are populated after a run."""

    def test_color_profiling_fields_empty_sim(self) -> None:
        """Empty simulation has zero color profiling fields."""
        cfg = MeshConfig(width=2, height=2)
        inp = SimInput()
        result = run_simulation(config=cfg, inputs=inp)
        assert hasattr(result, "max_colors_per_link")
        assert hasattr(result, "total_colors_used")
        assert result.max_colors_per_link == 0
        assert result.total_colors_used == 0

    def test_color_profiling_fields_single_message(self) -> None:
        """A single message populates color profiling fields."""
        cfg = MeshConfig(width=4, height=4)
        inp = SimInput()
        inp.add_message(source=(0, 0), dest=(2, 0), payload=[1.0, 2.0])
        inp.add_task(coord=(2, 0), kind=TaskKind.CollectOutput, trigger_slot=0)
        result = run_simulation(config=cfg, inputs=inp)
        # Single message with color 0 traverses 2 links
        assert result.total_colors_used >= 1
        assert result.max_colors_per_link >= 1

    def test_color_profiling_fields_transformer_block(self) -> None:
        """Transformer block run produces meaningful color profiling stats."""
        graph = transformer_block(4, 8, 16)
        weights = transformer_weights(8, 16, seed=WEIGHTS_SEED)
        config = CompilerConfig(mesh_height=6)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(4, 8)
        result = run_program(artifact_bytes, inputs={"input": x.flatten().tolist()})

        # With color-based routing, multiple colors should be used
        assert result.total_colors_used > 0
        assert result.max_colors_per_link > 0
        # Colors used should not exceed the budget of 8
        assert result.total_colors_used <= 8


class TestBenchmarkColorMetrics:
    """Phase 4: Verify benchmark script reports color metrics."""

    def test_benchmark_returns_color_metrics(self) -> None:
        """The run_benchmark function returns color metric fields."""
        import sys

        sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[3] / "scripts"))
        from benchmark import run_benchmark

        metrics = run_benchmark("small")
        assert "total_colors_used" in metrics
        assert "max_colors_per_link" in metrics
        assert "link_contentions" in metrics
        assert isinstance(metrics["total_colors_used"], int)
        assert isinstance(metrics["max_colors_per_link"], int)
        assert isinstance(metrics["link_contentions"], int)
        assert metrics["total_colors_used"] > 0
        assert metrics["max_colors_per_link"] > 0
