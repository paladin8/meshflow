"""Benchmark regression tests — asserts key metrics don't exceed baseline.

Thresholds are tightened after each milestone phase to lock in improvements.
"""

import torch

from meshflow._mesh_runtime import run_program
from meshflow.compiler import CompilerConfig, compile
from meshflow.compiler.artifact import serialize
from meshflow.models.transformer import transformer_block, transformer_weights

WEIGHTS_SEED = 42
INPUT_SEED = 99


def _run_config(seq_len: int, d_model: int, d_ff: int, mesh_height: int) -> tuple:
    """Compile and run a transformer block, return (program, result)."""
    graph = transformer_block(seq_len, d_model, d_ff)
    weights = transformer_weights(d_model, d_ff, seed=WEIGHTS_SEED)
    config = CompilerConfig(mesh_height=mesh_height)
    program = compile(graph, config, weights=weights)
    artifact_bytes = serialize(program)

    torch.manual_seed(INPUT_SEED)
    x = torch.randn(seq_len, d_model)
    result = run_program(artifact_bytes, inputs={"input": x.flatten().tolist()})
    return program, result


def _route_stats(program) -> tuple[int, int, int]:
    """Return (total_routes, routing_table_entries, max_manhattan_hops).

    With Phase 3 routing tables, hops are no longer stored per-route.
    Estimate max_hops via Manhattan distance from source PE to route dest.
    """
    total_routes = 0
    routing_table_entries = 0
    max_hops = 0
    for pe in program.pe_programs:
        for task in pe.tasks:
            if hasattr(task, "routes"):
                for route in task.routes:
                    total_routes += 1
                    dx = abs(pe.coord[0] - route.dest[0])
                    dy = abs(pe.coord[1] - route.dest[1])
                    max_hops = max(max_hops, dx + dy)
        routing_table_entries += len(pe.routing_table)
    return total_routes, routing_table_entries, max_hops


def _max_sends(result) -> int:
    """Return the maximum messages_sent by any single PE."""
    return max(s.messages_sent for s in result.pe_stats.values())


def _max_queue_depth(result) -> int:
    """Return the maximum queue depth across all PEs."""
    return max(s.max_queue_depth for s in result.pe_stats.values())


class TestSmallConfig:
    """Regression tests for small config (seq_len=4, d_model=8, d_ff=16, mesh_height=6)."""

    @classmethod
    def setup_class(cls):
        cls.program, cls.result = _run_config(4, 8, 16, 6)
        cls.config = CompilerConfig(mesh_height=6)

    def test_mesh_width(self):
        assert self.program.mesh_config.width <= 11  # Phase 4: 12 -> 11 (ADD co-located)

    def test_total_messages(self):
        assert self.result.total_messages <= 114

    def test_total_hops(self):
        assert self.result.total_hops <= 218  # Phase 4: 221 -> 218

    def test_final_timestamp(self):
        assert self.result.final_timestamp <= 692  # M10P2: 699 -> 692 (parallel sends)

    def test_max_sends(self):
        assert _max_sends(self.result) <= 17

    def test_max_queue_depth(self):
        assert (
            _max_queue_depth(self.result) <= 8
        )  # M10P2: 6 -> 8 (parallel sends increase concurrency)

    def test_max_hops_per_route(self):
        _, _, max_hops = _route_stats(self.program)
        assert max_hops <= 8

    def test_total_colors_within_budget(self):
        """Total colors used must fit within the color budget."""
        assert self.result.total_colors_used <= self.config.color_budget, (
            f"total_colors_used={self.result.total_colors_used} "
            f"exceeds budget={self.config.color_budget}"
        )

    def test_color_contentions_low(self):
        """Color contentions should be zero or very small."""
        assert self.result.color_contentions <= 10, (
            f"color_contentions={self.result.color_contentions} exceeds threshold"
        )


class TestMediumConfig:
    """Regression tests for medium config (seq_len=8, d_model=16, d_ff=32, mesh_height=8)."""

    @classmethod
    def setup_class(cls):
        cls.program, cls.result = _run_config(8, 16, 32, 8)
        cls.config = CompilerConfig(mesh_height=8)

    def test_mesh_width(self):
        assert self.program.mesh_config.width <= 11  # Phase 4: 12 -> 11 (ADD co-located)

    def test_total_messages(self):
        assert self.result.total_messages <= 170

    def test_total_hops(self):
        assert self.result.total_hops <= 376  # Phase 4: 379 -> 376

    def test_final_timestamp(self):
        assert self.result.final_timestamp <= 3040  # M10P2.1: diversity reshuffle

    def test_max_sends(self):
        assert _max_sends(self.result) <= 22

    def test_max_queue_depth(self):
        assert _max_queue_depth(self.result) <= 9  # M10P2.1: diversity increases concurrency

    def test_total_colors_within_budget(self):
        """Total colors used must fit within the color budget."""
        assert self.result.total_colors_used <= self.config.color_budget, (
            f"total_colors_used={self.result.total_colors_used} "
            f"exceeds budget={self.config.color_budget}"
        )

    def test_color_contentions_low(self):
        """Color contentions should be zero or very small."""
        assert self.result.color_contentions <= 10, (
            f"color_contentions={self.result.color_contentions} exceeds threshold"
        )
