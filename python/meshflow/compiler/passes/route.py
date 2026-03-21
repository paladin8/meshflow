"""Routing pass — generates concrete hop lists and flattens to per-PE schedules."""

import numpy as np

from meshflow.compiler.config import CompilerConfig, RoutingStrategy
from meshflow.compiler.graph_ir import OpType
from meshflow.compiler.schedule_ir import (
    CollectOutputEntry,
    ConcatCollectEntry,
    Direction,
    ForwardActivationEntry,
    InputSlot,
    LinearEntry,
    PESchedule,
    ScheduleIR,
    TaskEntry,
)
from meshflow.compiler.spatial_ir import SpatialIR


def route(
    spatial: SpatialIR,
    config: CompilerConfig,
    weights: dict[str, dict[str, np.ndarray]] | None = None,
) -> ScheduleIR:
    """Generate routes for all edges and flatten to per-PE task schedules.

    Dispatches on config.routing strategy.
    """
    if config.routing == RoutingStrategy.DIMENSION_ORDERED_XY:
        return _route_xy(spatial, weights)

    raise ValueError(f"unknown routing strategy: {config.routing!r}")


def _route_xy(
    spatial: SpatialIR,
    weights: dict[str, dict[str, np.ndarray]] | None = None,
) -> ScheduleIR:
    """Dimension-ordered XY routing: X first, then Y."""
    node_map = {n.id: n for n in spatial.nodes}

    # Build per-PE task lists and initial_sram from nodes + edges
    pe_tasks: dict[tuple[int, int], list[TaskEntry]] = {}
    pe_sram: dict[tuple[int, int], dict[int, list[float]]] = {}
    for node in spatial.nodes:
        pe_tasks.setdefault(node.coord, [])
        pe_sram.setdefault(node.coord, {})

    # Collect incoming edges per destination node for building tasks
    incoming_edges: dict[str, list[tuple[str, int, int]]] = {n.id: [] for n in spatial.nodes}
    for edge in spatial.edges:
        incoming_edges[edge.dst_node].append((edge.src_node, edge.src_slot, edge.dst_slot))

    # For each node, create task entries
    for node in spatial.nodes:
        if node.op == OpType.FORWARD:
            # Find the outgoing edge to determine route_dest.
            outgoing = [e for e in spatial.edges if e.src_node == node.id]
            if not outgoing:
                continue
            dst_node = node_map[outgoing[0].dst_node]
            hops = _generate_route_xy(node.coord, dst_node.coord)
            pe_tasks[node.coord].append(
                ForwardActivationEntry(
                    trigger_slot=0,
                    input_slot=0,
                    route_dest=dst_node.coord,
                    route_hops=hops,
                )
            )

        elif node.op == OpType.LINEAR:
            # Tile PE: compute y_i = W_i @ x + b_i, route to collect
            if node.attrs is None:
                raise ValueError(f"LINEAR tile node {node.id!r} missing attrs")
            tile_index = node.attrs["tile_index"]
            tile_rows = node.attrs["tile_rows"]
            fragment_offset = node.attrs["fragment_offset"]
            in_features = node.attrs["in_features"]
            origin_id = node.attrs["origin_id"]

            # Find the outgoing edge (tile -> collect)
            outgoing = [e for e in spatial.edges if e.src_node == node.id]
            if not outgoing:
                raise ValueError(f"LINEAR tile node {node.id!r} has no outgoing edge")
            collect_node = node_map[outgoing[0].dst_node]
            hops = _generate_route_xy(node.coord, collect_node.coord)

            pe_tasks[node.coord].append(
                LinearEntry(
                    trigger_slot=0,
                    input_slot=0,
                    weight_slot=1,
                    bias_slot=2,
                    tile_rows=tile_rows,
                    tile_cols=in_features,
                    route_dest=collect_node.coord,
                    route_hops=hops,
                    fragment_slot=tile_index,
                    fragment_offset=fragment_offset,
                )
            )

            # Tile weight and bias data into SRAM
            if weights is not None and origin_id in weights:
                w = weights[origin_id]["weight"]
                b = weights[origin_id]["bias"]
                weight_tile = w[fragment_offset : fragment_offset + tile_rows, :].flatten().tolist()
                bias_tile = b[fragment_offset : fragment_offset + tile_rows].flatten().tolist()
                pe_sram[node.coord][1] = weight_tile
                pe_sram[node.coord][2] = bias_tile

        elif node.op == OpType.COLLECT:
            # Check if this is a ConcatCollect (has num_fragments attr)
            if node.attrs is not None and "num_fragments" in node.attrs:
                num_fragments = node.attrs["num_fragments"]
                total_rows = node.attrs["total_rows"]
                # Compute per-fragment offsets
                base = total_rows // num_fragments
                remainder = total_rows % num_fragments
                for i in range(num_fragments):
                    frag_offset = i * base + min(i, remainder)
                    pe_tasks[node.coord].append(
                        ConcatCollectEntry(
                            trigger_slot=i,
                            num_fragments=num_fragments,
                            total_rows=total_rows,
                            fragment_offset=frag_offset,
                        )
                    )
            else:
                pe_tasks[node.coord].append(
                    CollectOutputEntry(
                        trigger_slot=0,
                        input_slot=0,
                    )
                )

    # Build PESchedule entries
    pe_schedules = [
        PESchedule(coord=coord, tasks=tasks, initial_sram=pe_sram.get(coord, {}))
        for coord, tasks in pe_tasks.items()
    ]

    # Identify input nodes (no incoming edges in the spatial graph)
    nodes_with_incoming = {e.dst_node for e in spatial.edges}

    # For LINEAR tile nodes, use the origin_id as the input slot name
    # (broadcast: multiple entries with the same name)
    input_slots: list[InputSlot] = []
    for n in spatial.nodes:
        if n.id in nodes_with_incoming:
            continue
        if n.op == OpType.LINEAR and n.attrs is not None:
            # Tile PE: broadcast input under the original node name
            input_slots.append(InputSlot(name=n.attrs["origin_id"], coord=n.coord, payload_slot=0))
        elif n.attrs is not None and "origin_id" in n.attrs:
            # Collect PE generated by LINEAR expansion — not an external input
            continue
        else:
            # Regular input node (FORWARD, standalone COLLECT, etc.)
            input_slots.append(InputSlot(name=n.id, coord=n.coord, payload_slot=0))

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
