"""Routing pass — generates concrete hop lists and flattens to per-PE schedules."""

import numpy as np

from meshflow.compiler.config import CompilerConfig, RoutingStrategy
from meshflow.compiler.graph_ir import OpType
from meshflow.compiler.schedule_ir import (
    CollectOutputEntry,
    ConcatCollectEntry,
    ConcatCollectForwardEntry,
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

    # Build index: origin_id → list of tile PE nodes (for inter-layer routing)
    tiles_by_origin: dict[str, list] = {}
    for node in spatial.nodes:
        if node.op == OpType.LINEAR and node.attrs is not None:
            origin = node.attrs["origin_id"]
            tiles_by_origin.setdefault(origin, []).append(node)

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
            if node.attrs is None or "num_fragments" not in node.attrs:
                # Simple collect (M2 style)
                pe_tasks[node.coord].append(CollectOutputEntry(trigger_slot=0, input_slot=0))
                continue

            num_fragments = node.attrs["num_fragments"]
            total_rows = node.attrs["total_rows"]
            activation = node.attrs.get("activation")
            route_to = node.attrs.get("route_to")

            # Compute per-fragment offsets
            base = total_rows // num_fragments
            remainder = total_rows % num_fragments

            if route_to is not None:
                # Intermediate layer: ConcatCollectForward with routes to next layer's tiles
                next_tiles = tiles_by_origin.get(route_to, [])
                route_dests: list[tuple[tuple[int, int], list[Direction]]] = []
                for tile_node in next_tiles:
                    tile_hops = _generate_route_xy(node.coord, tile_node.coord)
                    route_dests.append((tile_node.coord, tile_hops))

                for i in range(num_fragments):
                    frag_offset = i * base + min(i, remainder)
                    pe_tasks[node.coord].append(
                        ConcatCollectForwardEntry(
                            trigger_slot=i,
                            num_fragments=num_fragments,
                            total_rows=total_rows,
                            fragment_offset=frag_offset,
                            activation=activation,
                            route_dests=list(route_dests),  # copy to avoid shared mutation
                        )
                    )
            else:
                # Terminal layer: ConcatCollect (writes to outputs)
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

    # Build PESchedule entries
    pe_schedules = [
        PESchedule(coord=coord, tasks=tasks, initial_sram=pe_sram.get(coord, {}))
        for coord, tasks in pe_tasks.items()
    ]

    # Input slots: only first layer's tile PEs are external inputs.
    # For multi-layer graphs, intermediate tiles receive input from the
    # preceding collect PE's broadcast, not from external input.
    first_layer_origin = _find_first_layer_origin(spatial)
    input_slots: list[InputSlot] = []
    for n in spatial.nodes:
        if n.op == OpType.LINEAR and n.attrs is not None:
            origin = n.attrs["origin_id"]
            if first_layer_origin is not None and origin == first_layer_origin:
                input_slots.append(InputSlot(name=origin, coord=n.coord, payload_slot=0))
        elif n.attrs is not None and "origin_id" in n.attrs:
            # Collect PE from LINEAR expansion — not an external input
            continue
        elif n.id not in {e.dst_node for e in spatial.edges}:
            # Regular input node (FORWARD, standalone COLLECT, etc.)
            input_slots.append(InputSlot(name=n.id, coord=n.coord, payload_slot=0))

    return ScheduleIR(
        width=spatial.width,
        height=spatial.height,
        pe_schedules=pe_schedules,
        input_slots=input_slots,
    )


def _find_first_layer_origin(spatial: SpatialIR) -> str | None:
    """Find the origin_id of the first LINEAR layer (leftmost column, x=0)."""
    for node in spatial.nodes:
        if node.op == OpType.LINEAR and node.attrs is not None:
            if node.coord[0] == 0:  # column 0
                return node.attrs["origin_id"]
    return None


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
