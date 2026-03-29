"""Color pass — conflict-graph-based greedy coloring for route disambiguation.

Sits between route and lower in the compiler pipeline:

    expand -> place -> route -> color -> lower

Assigns integer color IDs (0..K-1) to routes so that no two routes needing
different forwarding behavior at the same intermediate PE share a color.
Also generates per-PE routing tables that map color -> (direction, optional
deliver_slot).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from meshflow.compiler.config import CompilerConfig
from meshflow.compiler.schedule_ir import (
    AddEntry,
    ConcatCollectForwardEntry,
    Direction,
    ForwardActivationEntry,
    LinearEntry,
    MatMulEntry,
    RmsNormNormalizeEntry,
    RmsNormPartialSumEntry,
    RmsNormReduceEntry,
    RouteTableEntry,
    ScheduleIR,
)


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class _IntermediatePE:
    """Info about a route's passage through an intermediate PE."""

    coord: tuple[int, int]
    next_direction: Direction
    deliver_slot: int | None


@dataclass
class _RouteInfo:
    """Metadata for a single route extracted from the schedule."""

    source_coord: tuple[int, int]
    hops: list[Direction]
    deliver_at: list[int]
    payload_slot: int
    intermediate_pes: set[tuple[int, int]]
    intermediate_pe_details: list[_IntermediatePE]
    write_back_ref: tuple[int, int, int]  # (pe_index, task_index, route_index)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _step_coord(coord: tuple[int, int], direction: Direction) -> tuple[int, int]:
    """Move one step in the given direction on the 2D mesh."""
    x, y = coord
    if direction is Direction.NORTH:
        return (x, y + 1)
    if direction is Direction.SOUTH:
        return (x, y - 1)
    if direction is Direction.EAST:
        return (x + 1, y)
    if direction is Direction.WEST:
        return (x - 1, y)
    raise ValueError(f"unknown direction: {direction!r}")  # pragma: no cover


def _compute_intermediates(
    source: tuple[int, int],
    hops: list[Direction],
    deliver_at: list[int],
    payload_slot: int,
) -> tuple[set[tuple[int, int]], list[_IntermediatePE]]:
    """Compute intermediate PEs and their forwarding details.

    Returns:
        A tuple of (pe_coord_set, detail_list) where:
        - pe_coord_set: the set of intermediate PE coordinates
        - detail_list: detailed per-PE forwarding info
    """
    if len(hops) <= 1:
        return set(), []

    deliver_at_set = set(deliver_at)
    pe_set: set[tuple[int, int]] = set()
    details: list[_IntermediatePE] = []

    current = source
    for hop_idx, direction in enumerate(hops):
        current = _step_coord(current, direction)
        # Skip the final destination (last hop)
        if hop_idx < len(hops) - 1:
            pe_set.add(current)
            deliver_slot: int | None = None
            if hop_idx in deliver_at_set:
                deliver_slot = payload_slot
            details.append(
                _IntermediatePE(
                    coord=current,
                    next_direction=hops[hop_idx + 1],
                    deliver_slot=deliver_slot,
                )
            )

    return pe_set, details


# ---------------------------------------------------------------------------
# Route enumeration
# ---------------------------------------------------------------------------


def _enumerate_routes(schedule: ScheduleIR) -> list[_RouteInfo]:
    """Walk all PESchedule entries and extract every route.

    Returns a flat list of _RouteInfo with write-back references so that
    assigned colors can be written back to the original IR structures.
    """
    routes: list[_RouteInfo] = []

    for pe_idx, pe in enumerate(schedule.pe_schedules):
        for task_idx, task in enumerate(pe.tasks):
            # Multi-route tasks: extract each BroadcastRoute
            if isinstance(
                task,
                (
                    ConcatCollectForwardEntry,
                    AddEntry,
                    MatMulEntry,
                    RmsNormNormalizeEntry,
                    RmsNormReduceEntry,
                ),
            ):
                for route_idx, br in enumerate(task.routes):
                    pe_set, details = _compute_intermediates(
                        pe.coord,
                        br.hops,
                        br.deliver_at,
                        br.payload_slot,
                    )
                    routes.append(
                        _RouteInfo(
                            source_coord=pe.coord,
                            hops=list(br.hops),
                            deliver_at=list(br.deliver_at),
                            payload_slot=br.payload_slot,
                            intermediate_pes=pe_set,
                            intermediate_pe_details=details,
                            write_back_ref=(pe_idx, task_idx, route_idx),
                        )
                    )

            # Single-route tasks
            elif isinstance(task, ForwardActivationEntry):
                pe_set, details = _compute_intermediates(
                    pe.coord,
                    task.route_hops,
                    [],
                    task.payload_slot,
                )
                routes.append(
                    _RouteInfo(
                        source_coord=pe.coord,
                        hops=list(task.route_hops),
                        deliver_at=[],
                        payload_slot=task.payload_slot,
                        intermediate_pes=pe_set,
                        intermediate_pe_details=details,
                        write_back_ref=(pe_idx, task_idx, -1),
                    )
                )

            elif isinstance(task, LinearEntry):
                pe_set, details = _compute_intermediates(
                    pe.coord,
                    task.route_hops,
                    [],
                    task.fragment_slot,
                )
                routes.append(
                    _RouteInfo(
                        source_coord=pe.coord,
                        hops=list(task.route_hops),
                        deliver_at=[],
                        payload_slot=task.fragment_slot,
                        intermediate_pes=pe_set,
                        intermediate_pe_details=details,
                        write_back_ref=(pe_idx, task_idx, -1),
                    )
                )

            elif isinstance(task, RmsNormPartialSumEntry):
                pe_set, details = _compute_intermediates(
                    pe.coord,
                    task.reduce_hops,
                    [],
                    task.partial_sum_slot,
                )
                routes.append(
                    _RouteInfo(
                        source_coord=pe.coord,
                        hops=list(task.reduce_hops),
                        deliver_at=[],
                        payload_slot=task.partial_sum_slot,
                        intermediate_pes=pe_set,
                        intermediate_pe_details=details,
                        write_back_ref=(pe_idx, task_idx, -1),
                    )
                )

    return routes


# ---------------------------------------------------------------------------
# Conflict graph
# ---------------------------------------------------------------------------


# A routing behavior key: (direction, deliver_slot) at a given PE.
_BehaviorKey = tuple[Direction, int | None]


def _build_conflict_graph(routes: list[_RouteInfo]) -> dict[int, set[int]]:
    """Build a conflict graph based on incompatible routing behaviors.

    Two routes conflict if they share an intermediate PE but need different
    forwarding behaviors (direction and/or deliver_slot) at that PE.  Routes
    that share intermediate PEs but always agree on the forwarding direction
    and deliver behavior can safely share a color.

    Returns an adjacency-set representation: graph[i] is the set of route
    indices that conflict with route i.
    """
    # Inverted index: (PE coord, behavior_key) -> list of route indices
    # Also track: PE coord -> set of behavior_keys used by each route
    pe_route_behaviors: dict[tuple[int, int], dict[int, _BehaviorKey]] = defaultdict(dict)

    for idx, route in enumerate(routes):
        for detail in route.intermediate_pe_details:
            behavior: _BehaviorKey = (detail.next_direction, detail.deliver_slot)
            pe_route_behaviors[detail.coord][idx] = behavior

    # Two routes conflict at a PE if they have different behavior keys
    graph: dict[int, set[int]] = defaultdict(set)
    for pe_coord, route_behaviors in pe_route_behaviors.items():
        route_indices = list(route_behaviors.keys())
        for i in range(len(route_indices)):
            for j in range(i + 1, len(route_indices)):
                a, b = route_indices[i], route_indices[j]
                if route_behaviors[a] != route_behaviors[b]:
                    graph[a].add(b)
                    graph[b].add(a)

    return graph


# ---------------------------------------------------------------------------
# Greedy coloring
# ---------------------------------------------------------------------------


def _greedy_color(
    num_routes: int,
    graph: dict[int, set[int]],
) -> list[int]:
    """Assign colors via greedy coloring, most-constrained first.

    Returns a list of color assignments indexed by route index.
    """
    # Sort by decreasing conflict degree (most-constrained first),
    # breaking ties by route index for determinism.
    order = sorted(
        range(num_routes),
        key=lambda i: (-len(graph.get(i, set())), i),
    )

    colors = [-1] * num_routes
    for idx in order:
        neighbor_colors = {colors[n] for n in graph.get(idx, set()) if colors[n] >= 0}
        # Find the smallest non-negative color not in neighbor_colors
        c = 0
        while c in neighbor_colors:
            c += 1
        colors[idx] = c

    return colors


# ---------------------------------------------------------------------------
# Write-back & routing table generation
# ---------------------------------------------------------------------------


def _write_back_colors(
    schedule: ScheduleIR,
    routes: list[_RouteInfo],
    colors: list[int],
) -> None:
    """Write assigned colors back onto the schedule IR structures."""
    for route_info, assigned_color in zip(routes, colors):
        pe_idx, task_idx, route_idx = route_info.write_back_ref
        pe = schedule.pe_schedules[pe_idx]
        task = pe.tasks[task_idx]

        if route_idx >= 0:
            # Multi-route task: set color on the BroadcastRoute
            assert isinstance(
                task,
                (
                    ConcatCollectForwardEntry,
                    AddEntry,
                    MatMulEntry,
                    RmsNormNormalizeEntry,
                    RmsNormReduceEntry,
                ),
            )
            task.routes[route_idx].color = assigned_color
        else:
            # Single-route task: set route_color on the entry
            assert isinstance(
                task,
                (ForwardActivationEntry, LinearEntry, RmsNormPartialSumEntry),
            )
            task.route_color = assigned_color


def _generate_routing_tables(
    schedule: ScheduleIR,
    routes: list[_RouteInfo],
    colors: list[int],
) -> None:
    """Build per-PE routing tables from route walks and color assignments.

    For each route, at each intermediate PE, record a routing table entry
    mapping color -> RouteTableEntry(direction, optional deliver_slot).
    """
    # Build coord -> pe_index map for fast lookup
    coord_to_pe: dict[tuple[int, int], int] = {}
    for idx, pe in enumerate(schedule.pe_schedules):
        coord_to_pe[pe.coord] = idx

    for route_info, assigned_color in zip(routes, colors):
        for detail in route_info.intermediate_pe_details:
            pe_idx = coord_to_pe.get(detail.coord)
            if pe_idx is None:
                continue  # PE not in schedule (shouldn't happen)

            pe = schedule.pe_schedules[pe_idx]
            pe.routing_table[assigned_color] = RouteTableEntry(
                direction=detail.next_direction,
                deliver_slot=detail.deliver_slot,
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def color(schedule: ScheduleIR, config: CompilerConfig | None = None) -> ScheduleIR:
    """Assign colors to all routes in the schedule and generate routing tables.

    Steps:
    1. Enumerate all routes from PESchedule entries.
    2. Extract intermediate PE sets and per-PE forwarding details.
    3. Build a conflict graph (incompatible behaviors at shared PEs).
    4. Greedy-color with most-constrained-first ordering.
    5. Assert color budget is not exceeded.
    6. Write colors back onto BroadcastRoute / single-route entries.
    7. Generate per-PE routing tables.

    Args:
        schedule: The ScheduleIR with routes from the route pass.
        config: Compiler configuration (uses color_budget). Defaults if None.

    Returns:
        The same ScheduleIR, mutated in place with color assignments and
        routing tables.

    Raises:
        ValueError: If the chromatic number exceeds config.color_budget.
    """
    if config is None:
        config = CompilerConfig()

    # Step 1-2: Enumerate routes and compute intermediate PE sets
    routes = _enumerate_routes(schedule)

    if not routes:
        return schedule

    # Step 3: Build conflict graph
    graph = _build_conflict_graph(routes)

    # Step 4: Greedy coloring
    colors = _greedy_color(len(routes), graph)

    # Step 5: Budget check
    chromatic_number = max(colors) + 1 if colors else 0
    if chromatic_number > config.color_budget:
        raise ValueError(
            f"color budget exceeded: need {chromatic_number} colors but "
            f"budget is {config.color_budget}"
        )

    # Step 6: Write back colors
    _write_back_colors(schedule, routes, colors)

    # Step 7: Generate routing tables
    _generate_routing_tables(schedule, routes, colors)

    return schedule
