"""Tests for the color pass — conflict graph and greedy coloring."""

from __future__ import annotations

import pytest

from meshflow.compiler import CompilerConfig
from meshflow.compiler.passes.color import (
    _compute_intermediates,
    _greedy_color,
    _step_coord,
    color,
)
from meshflow.compiler.passes.expand import expand
from meshflow.compiler.passes.place import place
from meshflow.compiler.passes.route import route
from meshflow.compiler.schedule_ir import (
    BroadcastRoute,
    ConcatCollectForwardEntry,
    Direction,
    ForwardActivationEntry,
    LinearEntry,
    PESchedule,
    RmsNormPartialSumEntry,
    ScheduleIR,
)
from meshflow.models.transformer import transformer_block, transformer_weights


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_schedule(pe_schedules: list[PESchedule], width: int = 10, height: int = 10) -> ScheduleIR:
    """Build a minimal ScheduleIR for testing."""
    return ScheduleIR(
        width=width,
        height=height,
        pe_schedules=pe_schedules,
        input_slots=[],
    )


def _collect_all_colors(schedule: ScheduleIR) -> list[int]:
    """Collect all assigned colors from a colored schedule."""
    colors: list[int] = []
    for pe in schedule.pe_schedules:
        for task in pe.tasks:
            if isinstance(task, (ForwardActivationEntry, LinearEntry, RmsNormPartialSumEntry)):
                colors.append(task.route_color)
            elif hasattr(task, "routes"):
                for r in task.routes:
                    colors.append(r.color)
    return colors


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStepCoord:
    def test_north(self) -> None:
        assert _step_coord((1, 2), Direction.NORTH) == (1, 3)

    def test_south(self) -> None:
        assert _step_coord((1, 2), Direction.SOUTH) == (1, 1)

    def test_east(self) -> None:
        assert _step_coord((1, 2), Direction.EAST) == (2, 2)

    def test_west(self) -> None:
        assert _step_coord((1, 2), Direction.WEST) == (0, 2)


class TestIntermediatePEs:
    def test_zero_hop_empty(self) -> None:
        pe_set, details = _compute_intermediates((0, 0), [], [], 0)
        assert pe_set == set()
        assert details == []

    def test_one_hop_empty(self) -> None:
        pe_set, details = _compute_intermediates((0, 0), [Direction.EAST], [], 0)
        assert pe_set == set()
        assert details == []

    def test_two_hop_one_intermediate(self) -> None:
        pe_set, details = _compute_intermediates(
            (0, 0),
            [Direction.EAST, Direction.EAST],
            [],
            0,
        )
        assert pe_set == {(1, 0)}
        assert len(details) == 1
        assert details[0].coord == (1, 0)
        assert details[0].next_direction == Direction.EAST

    def test_multi_hop(self) -> None:
        pe_set, details = _compute_intermediates(
            (0, 0),
            [Direction.EAST, Direction.EAST, Direction.NORTH],
            [],
            0,
        )
        # (0,0) -> (1,0) -> (2,0) -> (2,1)
        # Intermediates: (1,0) and (2,0), destination (2,1) excluded
        assert pe_set == {(1, 0), (2, 0)}
        assert len(details) == 2


class TestNoConflictAllColorZero:
    """Routes on non-overlapping PEs should all get color 0."""

    def test_no_conflict_all_color_zero(self) -> None:
        # Three ForwardActivation routes that each have 1 hop (no intermediates)
        pe_schedules = [
            PESchedule(
                coord=(0, 0),
                tasks=[
                    ForwardActivationEntry(
                        trigger_slot=0,
                        input_slot=0,
                        route_dest=(1, 0),
                        route_hops=[Direction.EAST],
                        payload_slot=0,
                    ),
                ],
            ),
            PESchedule(coord=(1, 0), tasks=[]),
            PESchedule(
                coord=(2, 0),
                tasks=[
                    ForwardActivationEntry(
                        trigger_slot=0,
                        input_slot=0,
                        route_dest=(3, 0),
                        route_hops=[Direction.EAST],
                        payload_slot=0,
                    ),
                ],
            ),
            PESchedule(coord=(3, 0), tasks=[]),
            PESchedule(
                coord=(4, 0),
                tasks=[
                    ForwardActivationEntry(
                        trigger_slot=0,
                        input_slot=0,
                        route_dest=(5, 0),
                        route_hops=[Direction.EAST],
                        payload_slot=0,
                    ),
                ],
            ),
            PESchedule(coord=(5, 0), tasks=[]),
        ]
        schedule = _make_schedule(pe_schedules)
        colored = color(schedule)

        all_colors = _collect_all_colors(colored)
        assert all(c == 0 for c in all_colors), f"expected all 0, got {all_colors}"


class TestSharedPeDifferentColors:
    """Two routes sharing an intermediate PE with different directions get different colors."""

    def test_shared_pe_different_colors(self) -> None:
        # Route A: (0,0) -> E -> E -> E to (3,0), intermediates at (1,0) and (2,0)
        # Route B: (0,1) -> S -> E -> E to (2,0), intermediates at (0,0) and (1,0)
        #   Wait, (0,0) is the source of route A so it won't be intermediate.
        # Better: Route B goes through (1,0) but in a different direction.
        # Route A: (0,0) -> E -> E -> N to (2,1), intermediates: (1,0) fwd E, (2,0) fwd N
        # Route B: (0,0) -> E -> N -> E to (2,1), intermediates: (1,0) fwd N, (1,1) fwd E
        # Both share (1,0) but A forwards E, B forwards N → conflict
        pe_schedules = [
            PESchedule(
                coord=(0, 0),
                tasks=[
                    ForwardActivationEntry(
                        trigger_slot=0,
                        input_slot=0,
                        route_dest=(2, 1),
                        route_hops=[Direction.EAST, Direction.EAST, Direction.NORTH],
                        payload_slot=0,
                    ),
                    ForwardActivationEntry(
                        trigger_slot=0,
                        input_slot=0,
                        route_dest=(2, 1),
                        route_hops=[Direction.EAST, Direction.NORTH, Direction.EAST],
                        payload_slot=1,
                    ),
                ],
            ),
            PESchedule(coord=(1, 0), tasks=[]),
            PESchedule(coord=(2, 0), tasks=[]),
            PESchedule(coord=(1, 1), tasks=[]),
            PESchedule(coord=(2, 1), tasks=[]),
        ]
        schedule = _make_schedule(pe_schedules)
        colored = color(schedule)

        c0 = colored.pe_schedules[0].tasks[0].route_color
        c1 = colored.pe_schedules[0].tasks[1].route_color
        assert c0 != c1, f"expected different colors, got {c0} and {c1}"


class TestGreedyColoringOrder:
    """Most-constrained route (highest conflict degree) is colored first."""

    def test_greedy_coloring_order(self) -> None:
        # Create a star conflict graph: route 0 conflicts with routes 1,2,3
        # Route 0 has degree 3, routes 1-3 have degree 1
        # Most-constrained (route 0) should be colored first → gets color 0
        # The others each get color 0 if they don't conflict with each other
        graph: dict[int, set[int]] = {
            0: {1, 2, 3},
            1: {0},
            2: {0},
            3: {0},
        }
        colors = _greedy_color(4, graph)
        # Route 0 (most constrained) gets colored first → color 0
        assert colors[0] == 0
        # Routes 1,2,3 don't conflict with each other → all get lowest available
        # Since they each only conflict with route 0 (color 0), they get color 1
        assert colors[1] == 1
        assert colors[2] == 1
        assert colors[3] == 1


class TestBudgetExceededRaises:
    """If chromatic number exceeds color_budget, raise ValueError."""

    def test_budget_exceeded_raises(self) -> None:
        # Create routes that form a clique requiring more colors than budget.
        # 3 routes through the same intermediate PE, each needing a different direction.
        # But we only have 4 cardinal directions, so let's create the scenario:
        # 3 routes all sharing PE (1,1), each forwarding in a different direction.
        # With budget=2, this needs 3 colors → exceeds budget.
        pe_schedules = [
            PESchedule(
                coord=(0, 1),
                tasks=[
                    # Route A: (0,1) -> E -> E to (2,1), intermediate (1,1) fwd E
                    ForwardActivationEntry(
                        trigger_slot=0,
                        input_slot=0,
                        route_dest=(2, 1),
                        route_hops=[Direction.EAST, Direction.EAST],
                        payload_slot=0,
                    ),
                    # Route B: (0,1) -> E -> N to (1,2), intermediate (1,1) fwd N
                    ForwardActivationEntry(
                        trigger_slot=0,
                        input_slot=0,
                        route_dest=(1, 2),
                        route_hops=[Direction.EAST, Direction.NORTH],
                        payload_slot=1,
                    ),
                    # Route C: (0,1) -> E -> S to (1,0), intermediate (1,1) fwd S
                    ForwardActivationEntry(
                        trigger_slot=0,
                        input_slot=0,
                        route_dest=(1, 0),
                        route_hops=[Direction.EAST, Direction.SOUTH],
                        payload_slot=2,
                    ),
                ],
            ),
            PESchedule(coord=(1, 1), tasks=[]),
            PESchedule(coord=(2, 1), tasks=[]),
            PESchedule(coord=(1, 2), tasks=[]),
            PESchedule(coord=(1, 0), tasks=[]),
        ]
        schedule = _make_schedule(pe_schedules)
        config = CompilerConfig(color_budget=2)

        with pytest.raises(ValueError, match="color budget exceeded"):
            color(schedule, config)


class TestTransformerBlockWithinBudget:
    """Full transformer block compiles with <= 8 colors."""

    def test_small_config(self) -> None:
        graph = transformer_block(seq_len=4, d_model=8, d_ff=16)
        weights = transformer_weights(d_model=8, d_ff=16)
        config = CompilerConfig()
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config, weights)

        # Should not raise
        colored = color(schedule, config)

        all_colors = _collect_all_colors(colored)
        assert max(all_colors) + 1 <= config.color_budget

    def test_medium_config(self) -> None:
        graph = transformer_block(seq_len=8, d_model=16, d_ff=32)
        weights = transformer_weights(d_model=16, d_ff=32)
        config = CompilerConfig()
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config, weights)

        colored = color(schedule, config)

        all_colors = _collect_all_colors(colored)
        assert max(all_colors) + 1 <= config.color_budget


class TestColorAssignmentIdempotent:
    """Running the color pass twice produces the same assignment."""

    def test_color_assignment_idempotent(self) -> None:
        graph = transformer_block(seq_len=4, d_model=8, d_ff=16)
        weights = transformer_weights(d_model=8, d_ff=16)
        config = CompilerConfig()
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config, weights)

        colored_once = color(schedule, config)
        colors_after_first = _collect_all_colors(colored_once)

        # Run again on the already-colored schedule
        colored_twice = color(colored_once, config)
        colors_after_second = _collect_all_colors(colored_twice)

        assert colors_after_first == colors_after_second


class TestRoutingTableEntriesGenerated:
    """Verify routing table entries are created at intermediate PEs."""

    def test_routing_table_entries_generated(self) -> None:
        # Route: (0,0) -> E -> E -> E to (3,0)
        # Intermediates: (1,0) and (2,0) both forwarding E
        pe_schedules = [
            PESchedule(
                coord=(0, 0),
                tasks=[
                    ForwardActivationEntry(
                        trigger_slot=0,
                        input_slot=0,
                        route_dest=(3, 0),
                        route_hops=[Direction.EAST, Direction.EAST, Direction.EAST],
                        payload_slot=0,
                    ),
                ],
            ),
            PESchedule(coord=(1, 0), tasks=[]),
            PESchedule(coord=(2, 0), tasks=[]),
            PESchedule(coord=(3, 0), tasks=[]),
        ]
        schedule = _make_schedule(pe_schedules)
        colored = color(schedule)

        # Intermediate PEs should have routing table entries
        pe_1_0 = next(p for p in colored.pe_schedules if p.coord == (1, 0))
        pe_2_0 = next(p for p in colored.pe_schedules if p.coord == (2, 0))

        assert len(pe_1_0.routing_table) == 1
        assert len(pe_2_0.routing_table) == 1

        # Both should forward east, no delivery
        for pe in [pe_1_0, pe_2_0]:
            entry = list(pe.routing_table.values())[0]
            assert entry.direction == Direction.EAST
            assert entry.deliver_slot is None

        # Source and destination should have no routing table entries
        pe_0_0 = next(p for p in colored.pe_schedules if p.coord == (0, 0))
        pe_3_0 = next(p for p in colored.pe_schedules if p.coord == (3, 0))
        assert len(pe_0_0.routing_table) == 0
        assert len(pe_3_0.routing_table) == 0

    def test_routing_table_with_deliver_at(self) -> None:
        """Broadcast route with deliver_at generates DeliverAndForward entries."""
        # Route: (0,0) -> E -> E -> E to (3,0) with deliver_at=[1] (deliver at hop 1)
        # After hop 0: at (1,0), next dir E — no deliver
        # After hop 1: at (2,0), next dir E — deliver_at includes 1
        pe_schedules = [
            PESchedule(
                coord=(0, 0),
                tasks=[
                    ConcatCollectForwardEntry(
                        trigger_slot=0,
                        num_fragments=1,
                        total_rows=4,
                        fragment_offset=0,
                        fragment_rows=4,
                        routes=[
                            BroadcastRoute(
                                dest=(3, 0),
                                hops=[Direction.EAST, Direction.EAST, Direction.EAST],
                                deliver_at=[1],
                                payload_slot=5,
                            ),
                        ],
                    ),
                ],
            ),
            PESchedule(coord=(1, 0), tasks=[]),
            PESchedule(coord=(2, 0), tasks=[]),
            PESchedule(coord=(3, 0), tasks=[]),
        ]
        schedule = _make_schedule(pe_schedules)
        colored = color(schedule)

        pe_1_0 = next(p for p in colored.pe_schedules if p.coord == (1, 0))
        pe_2_0 = next(p for p in colored.pe_schedules if p.coord == (2, 0))

        # PE (1,0) at hop_idx=0: no deliver
        entry_1 = list(pe_1_0.routing_table.values())[0]
        assert entry_1.direction == Direction.EAST
        assert entry_1.deliver_slot is None

        # PE (2,0) at hop_idx=1: deliver_at includes 1 → deliver_slot = payload_slot = 5
        entry_2 = list(pe_2_0.routing_table.values())[0]
        assert entry_2.direction == Direction.EAST
        assert entry_2.deliver_slot == 5

    def test_transformer_block_has_routing_tables(self) -> None:
        """Full transformer block should have non-empty routing tables on some PEs."""
        graph = transformer_block(seq_len=4, d_model=8, d_ff=16)
        weights = transformer_weights(d_model=8, d_ff=16)
        config = CompilerConfig()
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config, weights)
        colored = color(schedule, config)

        pes_with_tables = [pe for pe in colored.pe_schedules if pe.routing_table]
        assert len(pes_with_tables) > 0, "expected some PEs to have routing tables"


class TestColorDiversity:
    """Diversity pass spreads colors across routes from the same PE."""

    def test_same_pe_routes_get_diverse_colors(self) -> None:
        """Routes from the same PE that don't conflict should get different colors."""
        # 4 routes from (0,0), each going to a different destination via 2 hops.
        # All share intermediate PE (1,0) going East — same behavior, no conflict.
        # Without diversity: all get color 0. With diversity: colors 0,1,2,3.
        pe_schedules = [
            PESchedule(
                coord=(0, 0),
                tasks=[
                    ConcatCollectForwardEntry(
                        trigger_slot=0,
                        num_fragments=1,
                        total_rows=4,
                        fragment_offset=0,
                        fragment_rows=4,
                        routes=[
                            BroadcastRoute(
                                dest=(2, i),
                                hops=[Direction.EAST, Direction.EAST] + [Direction.NORTH] * i,
                                payload_slot=i,
                            )
                            for i in range(4)
                        ],
                    ),
                ],
            ),
            PESchedule(coord=(1, 0), tasks=[]),
            PESchedule(coord=(2, 0), tasks=[]),
            PESchedule(coord=(2, 1), tasks=[]),
            PESchedule(coord=(2, 2), tasks=[]),
            PESchedule(coord=(2, 3), tasks=[]),
        ]
        schedule = _make_schedule(pe_schedules)
        colored = color(schedule)

        route_colors = [r.color for r in colored.pe_schedules[0].tasks[0].routes]
        distinct = len(set(route_colors))
        assert distinct == 4, f"expected 4 distinct colors, got {distinct}: {route_colors}"

    def test_transformer_block_diversity(self) -> None:
        """High-fanout PEs in transformer block should use multiple colors."""
        graph = transformer_block(seq_len=4, d_model=8, d_ff=16)
        weights = transformer_weights(d_model=8, d_ff=16)
        config = CompilerConfig()
        expanded = expand(graph, config)
        spatial = place(expanded, config)
        schedule = route(spatial, config, weights)
        colored = color(schedule, config)

        # Find the PE with the most routes
        from collections import defaultdict

        pe_colors: dict[tuple[int, int], list[int]] = defaultdict(list)
        for pe in colored.pe_schedules:
            for task in pe.tasks:
                if hasattr(task, "routes"):
                    for r in task.routes:
                        pe_colors[pe.coord].append(r.color)

        # The busiest PE should use more than 2 distinct colors
        if pe_colors:
            busiest = max(pe_colors.values(), key=len)
            distinct = len(set(busiest))
            assert distinct > 2, f"busiest PE has {len(busiest)} routes but only {distinct} colors"


class TestColorPassIntegration:
    """Integration tests: color pass works with the full compiler pipeline."""

    def test_full_pipeline_compiles(self) -> None:
        """The full compile() pipeline with color pass produces a valid artifact."""
        from meshflow.compiler import compile as meshflow_compile

        graph = transformer_block(seq_len=4, d_model=8, d_ff=16)
        weights = transformer_weights(d_model=8, d_ff=16)
        config = CompilerConfig()

        # Should not raise
        program = meshflow_compile(graph, config, weights)
        assert program is not None
        assert len(program.pe_programs) > 0

    def test_empty_schedule_is_noop(self) -> None:
        """Color pass on empty schedule returns immediately."""
        schedule = _make_schedule([])
        result = color(schedule)
        assert result is schedule
