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
from meshflow.compiler.spatial_ir import PlacedCollectData, PlacedTileData, SpatialIR


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

    pe_tasks: dict[tuple[int, int], list[TaskEntry]] = {}
    pe_sram: dict[tuple[int, int], dict[int, list[float]]] = {}
    for node in spatial.nodes:
        pe_tasks.setdefault(node.coord, [])
        pe_sram.setdefault(node.coord, {})

    for node in spatial.nodes:
        if node.op == OpType.FORWARD:
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

        elif isinstance(node.data, PlacedTileData):
            tile = node.data
            # Find outgoing edge to collect PE
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
                    tile_rows=tile.tile_rows,
                    tile_cols=tile.in_features,
                    route_dest=collect_node.coord,
                    route_hops=hops,
                    fragment_slot=tile.tile_index,
                    fragment_offset=tile.fragment_offset,
                )
            )

            # Weight/bias SRAM
            if weights is not None and tile.origin_id in weights:
                w = weights[tile.origin_id]["weight"]
                b = weights[tile.origin_id]["bias"]
                weight_tile = (
                    w[tile.fragment_offset : tile.fragment_offset + tile.tile_rows, :]
                    .flatten()
                    .tolist()
                )
                bias_tile = (
                    b[tile.fragment_offset : tile.fragment_offset + tile.tile_rows]
                    .flatten()
                    .tolist()
                )
                pe_sram[node.coord][1] = weight_tile
                pe_sram[node.coord][2] = bias_tile

        elif isinstance(node.data, PlacedCollectData):
            collect = node.data

            # Determine intermediate vs terminal from explicit outgoing edges
            outgoing = [e for e in spatial.edges if e.src_node == node.id]
            is_intermediate = len(outgoing) > 0

            base = collect.total_rows // collect.num_fragments
            remainder = collect.total_rows % collect.num_fragments

            if is_intermediate:
                # Build route_dests from explicit edges
                route_dests: list[tuple[tuple[int, int], list[Direction]]] = []
                for edge in outgoing:
                    dst = node_map[edge.dst_node]
                    tile_hops = _generate_route_xy(node.coord, dst.coord)
                    route_dests.append((dst.coord, tile_hops))

                for i in range(collect.num_fragments):
                    frag_offset = i * base + min(i, remainder)
                    pe_tasks[node.coord].append(
                        ConcatCollectForwardEntry(
                            trigger_slot=i,
                            num_fragments=collect.num_fragments,
                            total_rows=collect.total_rows,
                            fragment_offset=frag_offset,
                            activation=collect.activation,
                            route_dests=list(route_dests),
                        )
                    )
            else:
                # Terminal layer
                for i in range(collect.num_fragments):
                    frag_offset = i * base + min(i, remainder)
                    pe_tasks[node.coord].append(
                        ConcatCollectEntry(
                            trigger_slot=i,
                            num_fragments=collect.num_fragments,
                            total_rows=collect.total_rows,
                            fragment_offset=frag_offset,
                        )
                    )

        elif node.op == OpType.COLLECT:
            # Simple collect (M2 style — no typed data)
            pe_tasks[node.coord].append(CollectOutputEntry(trigger_slot=0, input_slot=0))

    # Input slots
    first_layer_origin = _find_first_layer_origin(spatial)
    input_slots: list[InputSlot] = []
    for n in spatial.nodes:
        if isinstance(n.data, PlacedTileData):
            if first_layer_origin is not None and n.data.origin_id == first_layer_origin:
                input_slots.append(InputSlot(name=n.data.origin_id, coord=n.coord, payload_slot=0))
        elif isinstance(n.data, PlacedCollectData):
            continue
        elif n.id not in {e.dst_node for e in spatial.edges}:
            input_slots.append(InputSlot(name=n.id, coord=n.coord, payload_slot=0))

    pe_schedules = [
        PESchedule(coord=coord, tasks=tasks, initial_sram=pe_sram.get(coord, {}))
        for coord, tasks in pe_tasks.items()
    ]

    return ScheduleIR(
        width=spatial.width,
        height=spatial.height,
        pe_schedules=pe_schedules,
        input_slots=input_slots,
    )


def _find_first_layer_origin(spatial: SpatialIR) -> str | None:
    """Find the origin_id of the first LINEAR layer (leftmost column, x=0)."""
    for node in spatial.nodes:
        if isinstance(node.data, PlacedTileData) and node.coord[0] == 0:
            return node.data.origin_id
    return None


def _generate_route_xy(src: tuple[int, int], dst: tuple[int, int]) -> list[Direction]:
    """Generate XY dimension-ordered hop list: X first, then Y."""
    hops: list[Direction] = []

    sx, sy = src
    dx, dy = dst
    while sx != dx:
        if dx > sx:
            hops.append(Direction.EAST)
            sx += 1
        else:
            hops.append(Direction.WEST)
            sx -= 1

    while sy != dy:
        if dy > sy:
            hops.append(Direction.NORTH)
            sy += 1
        else:
            hops.append(Direction.SOUTH)
            sy -= 1

    return hops
