"""Routing pass — generates concrete hop lists and flattens to per-PE schedules."""

from meshflow.compiler.config import CompilerConfig, RoutingStrategy
from meshflow.compiler.graph_ir import OpType
from meshflow.compiler.schedule_ir import (
    Direction,
    InputSlot,
    PESchedule,
    ScheduleIR,
    TaskEntry,
)
from meshflow.compiler.spatial_ir import SpatialIR


def route(spatial: SpatialIR, config: CompilerConfig) -> ScheduleIR:
    """Generate routes for all edges and flatten to per-PE task schedules.

    Dispatches on config.routing strategy.
    """
    if config.routing == RoutingStrategy.DIMENSION_ORDERED_XY:
        return _route_xy(spatial)

    raise ValueError(f"unknown routing strategy: {config.routing!r}")


def _route_xy(spatial: SpatialIR) -> ScheduleIR:
    """Dimension-ordered XY routing: X first, then Y."""
    node_map = {n.id: n for n in spatial.nodes}

    # Build per-PE task lists from nodes + edges
    pe_tasks: dict[tuple[int, int], list[TaskEntry]] = {}
    for node in spatial.nodes:
        pe_tasks.setdefault(node.coord, [])

    # Collect incoming edges per destination node for building tasks
    incoming_edges: dict[str, list[tuple[str, int, int]]] = {n.id: [] for n in spatial.nodes}
    for edge in spatial.edges:
        incoming_edges[edge.dst_node].append((edge.src_node, edge.src_slot, edge.dst_slot))

    # For each node, create a task entry
    for node in spatial.nodes:
        if node.op == OpType.FORWARD:
            # Find the outgoing edge to determine route_dest.
            # M2 limitation: only the first outgoing edge is routed.
            # Fan-out (multiple outgoing edges) will need one task per edge.
            outgoing = [e for e in spatial.edges if e.src_node == node.id]
            if not outgoing:
                # Forward node with no outgoing edges: no task to create.
                continue
            dst_node = node_map[outgoing[0].dst_node]
            hops = _generate_route_xy(node.coord, dst_node.coord)
            pe_tasks[node.coord].append(
                TaskEntry(
                    kind="forward_activation",
                    trigger_slot=0,
                    input_slot=0,
                    route_dest=dst_node.coord,
                    route_hops=hops,
                )
            )
        elif node.op == OpType.COLLECT:
            pe_tasks[node.coord].append(
                TaskEntry(
                    kind="collect_output",
                    trigger_slot=0,
                    input_slot=0,
                )
            )

    # Build PESchedule entries
    pe_schedules = [PESchedule(coord=coord, tasks=tasks) for coord, tasks in pe_tasks.items()]

    # Identify input nodes (no incoming edges)
    nodes_with_incoming = {e.dst_node for e in spatial.edges}
    input_slots = [
        InputSlot(name=n.id, coord=n.coord, payload_slot=0)
        for n in spatial.nodes
        if n.id not in nodes_with_incoming
    ]

    return ScheduleIR(
        width=spatial.width,
        height=spatial.height,
        pe_schedules=pe_schedules,
        input_slots=input_slots,
    )


def _generate_route_xy(src: tuple[int, int], dst: tuple[int, int]) -> list[Direction]:
    """Generate XY dimension-ordered hop list: X first, then Y."""
    hops: list[Direction] = []

    # X movement first
    sx, sy = src
    dx, dy = dst
    while sx != dx:
        if dx > sx:
            hops.append(Direction.EAST)
            sx += 1
        else:
            hops.append(Direction.WEST)
            sx -= 1

    # Then Y movement
    while sy != dy:
        if dy > sy:
            hops.append(Direction.NORTH)
            sy += 1
        else:
            hops.append(Direction.SOUTH)
            sy -= 1

    return hops
