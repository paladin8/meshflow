"""Tests for color-based routing: parallel sends, contention, and profiling.

Verifies that:
- Routes on different colors depart in parallel (same tick)
- Routes on the same color are serialized
- color_contentions counter is exposed and correct
- Transformer block final_timestamp improves with parallel sends
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
    """Verify color_contentions is accessible from Python."""

    def test_color_contentions_accessible(self) -> None:
        """The color_contentions field exists and defaults to 0 for empty sim."""
        cfg = MeshConfig(width=2, height=2)
        inp = SimInput()
        result = run_simulation(config=cfg, inputs=inp)
        assert hasattr(result, "color_contentions")
        assert result.color_contentions == 0

    def test_color_contentions_zero_for_simple_message(self) -> None:
        """A single message with no color conflicts has 0 contentions."""
        cfg = MeshConfig(width=4, height=4)
        inp = SimInput()
        inp.add_message(source=(0, 0), dest=(2, 0), payload=[1.0, 2.0])
        inp.add_task(coord=(2, 0), kind=TaskKind.CollectOutput, trigger_slot=0)
        result = run_simulation(config=cfg, inputs=inp)
        assert result.color_contentions == 0

    def test_contention_detected_same_color_same_link(self) -> None:
        """Two same-color (color 0) messages on the same link cause contention."""
        cfg = MeshConfig(width=2, height=1)
        inp = SimInput()
        # Both go (0,0) -> (1,0), both color 0 (default)
        inp.add_message(source=(0, 0), dest=(1, 0), payload=[1.0])
        inp.add_message(source=(0, 0), dest=(1, 0), payload=[2.0])
        inp.add_task(coord=(1, 0), kind=TaskKind.CollectOutput, trigger_slot=0)
        result = run_simulation(config=cfg, inputs=inp)
        assert result.color_contentions > 0, (
            "Two same-color messages on the same link at the same time should cause contention"
        )


class TestParallelSendBehavior:
    """Verify parallel send semantics via compiled artifacts."""

    def test_transformer_block_final_timestamp_reasonable(self) -> None:
        """Transformer block compiles and runs with parallel sends.

        The final_timestamp should be within expected bounds (not regressed).
        """
        graph = transformer_block(4, 8, 16)
        weights = transformer_weights(8, 16, seed=WEIGHTS_SEED)
        config = CompilerConfig(mesh_height=6)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(4, 8)
        result = run_program(artifact_bytes, inputs={"input": x.flatten().tolist()})

        # With parallel sends, final_timestamp should be reasonable
        # (before Phase 2 it was 699, now should be lower due to parallelism)
        assert result.final_timestamp > 0
        assert result.final_timestamp <= 700  # conservative upper bound

    def test_small_transformer_contentions_low(self) -> None:
        """Small transformer block has low color contentions.

        The Phase 1 coloring uses intermediate-PE overlap, which is conservative.
        A small number of contentions is acceptable at this stage.
        """
        graph = transformer_block(4, 8, 16)
        weights = transformer_weights(8, 16, seed=WEIGHTS_SEED)
        config = CompilerConfig(mesh_height=6)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(4, 8)
        result = run_program(artifact_bytes, inputs={"input": x.flatten().tolist()})

        # Contentions should be small (coloring handles most conflicts)
        assert result.color_contentions <= 20, (
            f"Expected low contentions, got {result.color_contentions}"
        )

    def test_medium_transformer_contentions_low(self) -> None:
        """Medium transformer block has low color contentions."""
        graph = transformer_block(8, 16, 32)
        weights = transformer_weights(16, 32, seed=WEIGHTS_SEED)
        config = CompilerConfig(mesh_height=8)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(8, 16)
        result = run_program(artifact_bytes, inputs={"input": x.flatten().tolist()})

        assert result.color_contentions <= 20, (
            f"Expected low contentions, got {result.color_contentions}"
        )

    def test_final_timestamp_improved_vs_serial_baseline(self) -> None:
        """Parallel sends should reduce final_timestamp compared to the
        pre-Phase 2 serial baseline (699 for small config, 3045 for medium).
        """
        # Small config
        graph = transformer_block(4, 8, 16)
        weights = transformer_weights(8, 16, seed=WEIGHTS_SEED)
        config = CompilerConfig(mesh_height=6)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(4, 8)
        result_small = run_program(artifact_bytes, inputs={"input": x.flatten().tolist()})

        # Pre-Phase 2 baseline was 699; parallel sends should reduce this
        assert result_small.final_timestamp < 699, (
            f"Small config final_timestamp {result_small.final_timestamp} "
            f"should be less than pre-Phase 2 baseline of 699"
        )

        # Medium config
        graph = transformer_block(8, 16, 32)
        weights = transformer_weights(16, 32, seed=WEIGHTS_SEED)
        config = CompilerConfig(mesh_height=8)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)

        torch.manual_seed(INPUT_SEED)
        x = torch.randn(8, 16)
        result_medium = run_program(artifact_bytes, inputs={"input": x.flatten().tolist()})

        # Pre-Phase 2 baseline was 3045; parallel sends should reduce this
        assert result_medium.final_timestamp < 3045, (
            f"Medium config final_timestamp {result_medium.final_timestamp} "
            f"should be less than pre-Phase 2 baseline of 3045"
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
        assert "color_contentions" in metrics
        assert isinstance(metrics["total_colors_used"], int)
        assert isinstance(metrics["max_colors_per_link"], int)
        assert isinstance(metrics["color_contentions"], int)
        assert metrics["total_colors_used"] > 0
        assert metrics["max_colors_per_link"] > 0
