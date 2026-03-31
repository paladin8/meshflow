"""Routing pass — generates concrete hop lists and flattens to per-PE schedules."""

from collections import defaultdict

import numpy as np

from meshflow.compiler.config import CompilerConfig, RoutingStrategy
from meshflow.compiler.schedule_ir import (
    AddEntry,
    BroadcastRoute,
    CollectOutputEntry,
    ConcatCollectEntry,
    ConcatCollectForwardEntry,
    Direction,
    ForwardActivationEntry,
    InputSlot,
    LinearEntry,
    MatMulEntry,
    PESchedule,
    RmsNormFusedEntry,
    ScheduleIR,
    SoftmaxEntry,
    TaskEntry,
)
from meshflow.compiler.spatial_ir import (
    PlacedAttentionPeData,
    PlacedCollectData,
    PlacedEdge,
    PlacedNode,
    PlacedNodeKind,
    PlacedRmsNormFusedData,
    PlacedTileData,
    SpatialIR,
)


def _outgoing_edges(spatial: SpatialIR, node_id: str) -> list[PlacedEdge]:
    """Return all edges originating from a given node."""
    return [e for e in spatial.edges if e.src_node == node_id]


def _load_linear_weights(
    pe_sram: dict[tuple[int, int], dict[int, list[float]]],
    coord: tuple[int, int],
    weights: dict[str, dict[str, np.ndarray]] | None,
    origin_id: str,
    fragment_offset: int,
    tile_rows: int,
    weight_slot: int = 1,
    bias_slot: int = 2,
) -> None:
    """Load weight and bias tiles into a PE's SRAM slots."""
    if weights is None or origin_id not in weights:
        return
    w = weights[origin_id]["weight"]
    b = weights[origin_id]["bias"]
    pe_sram[coord][weight_slot] = (
        w[fragment_offset : fragment_offset + tile_rows, :].flatten().tolist()
    )
    pe_sram[coord][bias_slot] = b[fragment_offset : fragment_offset + tile_rows].flatten().tolist()


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
    # Initialize ALL mesh PEs so that routing tables can be generated for
    # intermediate PEs that messages pass through (even if they have no tasks).
    for x in range(spatial.width):
        for y in range(spatial.height):
            pe_tasks.setdefault((x, y), [])
            pe_sram.setdefault((x, y), {})

    for node in spatial.nodes:
        if node.kind == PlacedNodeKind.FORWARD:
            outgoing = _outgoing_edges(spatial, node.id)
            if not outgoing:
                continue
            # Generate one ForwardActivationEntry per outgoing edge (broadcast)
            for edge in outgoing:
                dst_node = node_map[edge.dst_node]
                hops = _generate_route_xy(node.coord, dst_node.coord)
                pe_tasks[node.coord].append(
                    ForwardActivationEntry(
                        trigger_slot=0,
                        input_slot=0,
                        routes=[
                            BroadcastRoute(
                                dest=dst_node.coord,
                                hops=hops,
                                payload_slot=edge.dst_slot,
                            )
                        ],
                    )
                )

        elif node.kind == PlacedNodeKind.LINEAR_TILE:
            assert isinstance(node.data, PlacedTileData)
            tile = node.data
            outgoing = _outgoing_edges(spatial, node.id)
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
                    routes=[
                        BroadcastRoute(
                            dest=collect_node.coord,
                            hops=hops,
                            payload_slot=tile.tile_index,
                        )
                    ],
                    fragment_offset=tile.fragment_offset,
                )
            )

            # Weight/bias SRAM
            _load_linear_weights(
                pe_sram,
                node.coord,
                weights,
                tile.origin_id,
                tile.fragment_offset,
                tile.tile_rows,
            )

        elif node.kind == PlacedNodeKind.LINEAR_COLLECT:
            assert isinstance(node.data, PlacedCollectData)
            collect = node.data

            # Determine intermediate vs terminal from explicit outgoing edges
            outgoing = _outgoing_edges(spatial, node.id)
            is_intermediate = len(outgoing) > 0

            base = collect.total_rows // collect.num_fragments
            remainder = collect.total_rows % collect.num_fragments

            if is_intermediate:
                collect_routes: list[BroadcastRoute] = []
                for edge in outgoing:
                    dst = node_map[edge.dst_node]
                    tile_hops = _generate_route_xy(node.coord, dst.coord)
                    collect_routes.append(
                        BroadcastRoute(
                            dest=dst.coord,
                            hops=tile_hops,
                            deliver_at=[],
                            payload_slot=edge.dst_slot,
                        )
                    )

                # Detect scatter: when routing to multiple attention PEs
                # and delivering to the Q slot (0), scatter Q rows instead
                # of broadcasting the full matrix.
                use_scatter = False
                if len(outgoing) > 1:
                    all_attn = all(
                        node_map[e.dst_node].kind == PlacedNodeKind.ATTENTION_PE for e in outgoing
                    )
                    # Q slot = 0 on attention PEs
                    all_q_slot = all(e.dst_slot == 0 for e in outgoing)
                    if all_attn and all_q_slot:
                        use_scatter = True

                # Apply broadcast detection for non-scatter routes
                if not use_scatter:
                    collect_routes = _try_linear_broadcast(node.coord, collect_routes)

                for i in range(collect.num_fragments):
                    frag_offset = i * base + min(i, remainder)
                    tile_rows = base + 1 if i < remainder else base
                    pe_tasks[node.coord].append(
                        ConcatCollectForwardEntry(
                            trigger_slot=i,
                            num_fragments=collect.num_fragments,
                            total_rows=collect.total_rows,
                            fragment_offset=frag_offset,
                            fragment_rows=tile_rows,
                            activation=collect.activation,
                            routes=list(collect_routes),
                            scatter=use_scatter,
                        )
                    )
            else:
                for i in range(collect.num_fragments):
                    frag_offset = i * base + min(i, remainder)
                    tile_rows = base + 1 if i < remainder else base
                    pe_tasks[node.coord].append(
                        ConcatCollectEntry(
                            trigger_slot=i,
                            num_fragments=collect.num_fragments,
                            total_rows=collect.total_rows,
                            fragment_offset=frag_offset,
                            fragment_rows=tile_rows,
                        )
                    )

        elif node.kind == PlacedNodeKind.RMSNORM_FUSED:
            _route_rmsnorm_fused(node, spatial, node_map, pe_tasks, pe_sram, weights)

        elif node.kind == PlacedNodeKind.ATTENTION_PE:
            _route_attention_pe(node, spatial, node_map, pe_tasks)

        elif node.kind == PlacedNodeKind.ADD:
            _route_add(node, spatial, node_map, pe_tasks)

        elif node.kind == PlacedNodeKind.SOFTMAX:
            pe_tasks[node.coord].append(SoftmaxEntry(trigger_slot=0, input_slot=0, output_slot=1))

        elif node.kind == PlacedNodeKind.COLLECT_SIMPLE:
            pe_tasks[node.coord].append(CollectOutputEntry(trigger_slot=0, input_slot=0))

    # Input slots
    input_slots = _generate_input_slots(spatial)

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


def _generate_input_slots(spatial: SpatialIR) -> list[InputSlot]:
    """Generate input slots for the schedule.

    Input PEs are those with no incoming edges that aren't internal
    infrastructure (collect, reduce, attention PEs receiving from projections).
    First-layer LINEAR tile PEs are treated as broadcast input targets.
    """
    input_slots: list[InputSlot] = []
    nodes_with_incoming = {e.dst_node for e in spatial.edges}
    first_layer_origin = _find_first_layer_origin(spatial)

    # Internal PE kinds that should never be external inputs
    _INTERNAL_KINDS = {
        PlacedNodeKind.LINEAR_COLLECT,
        PlacedNodeKind.RMSNORM_FUSED,
        PlacedNodeKind.ATTENTION_PE,
    }

    for n in spatial.nodes:
        if n.kind == PlacedNodeKind.LINEAR_TILE:
            assert isinstance(n.data, PlacedTileData)
            if first_layer_origin is not None and n.data.origin_id == first_layer_origin:
                input_slots.append(InputSlot(name=n.data.origin_id, coord=n.coord, payload_slot=0))
        elif n.kind in _INTERNAL_KINDS:
            continue
        elif n.id not in nodes_with_incoming:
            input_slots.append(InputSlot(name=n.id, coord=n.coord, payload_slot=0))

    return input_slots


def _find_first_layer_origin(spatial: SpatialIR) -> str | None:
    """Find the origin_id of the first LINEAR layer (leftmost column, x=0)."""
    for node in spatial.nodes:
        if node.kind == PlacedNodeKind.LINEAR_TILE and node.coord[0] == 0:
            assert isinstance(node.data, PlacedTileData)
            return node.data.origin_id
    return None


def _route_rmsnorm_fused(
    node: PlacedNode,
    spatial: SpatialIR,
    node_map: dict[str, PlacedNode],
    pe_tasks: dict[tuple[int, int], list[TaskEntry]],
    pe_sram: dict[tuple[int, int], dict[int, list[float]]],
    weights: dict[str, dict[str, np.ndarray]] | None,
) -> None:
    """Route a fused single-PE RMSNorm: receive input, normalize, broadcast."""
    data = node.data
    assert isinstance(data, PlacedRmsNormFusedData)

    # Build routes to downstream nodes
    outgoing = _outgoing_edges(spatial, node.id)
    fused_routes: list[BroadcastRoute] = []
    for edge in outgoing:
        dst = node_map[edge.dst_node]
        hops = _generate_route_xy(node.coord, dst.coord)
        fused_routes.append(
            BroadcastRoute(
                dest=dst.coord,
                hops=hops,
                deliver_at=[],
                payload_slot=edge.dst_slot,
            )
        )

    fused_routes = _try_linear_broadcast(node.coord, fused_routes)

    pe_tasks[node.coord].append(
        RmsNormFusedEntry(
            trigger_slot=0,
            input_slot=0,
            gamma_slot=1,
            feature_count=data.feature_count,
            eps=data.eps,
            routes=fused_routes,
        )
    )

    # Load gamma weights into SRAM slot 1
    if weights is not None and data.origin_id in weights:
        gamma = weights[data.origin_id]["gamma"]
        pe_sram[node.coord][1] = gamma.flatten().tolist()


def _route_attention_pe(
    node: PlacedNode,
    spatial: SpatialIR,
    node_map: dict[str, PlacedNode],
    pe_tasks: dict[tuple[int, int], list[TaskEntry]],
) -> None:
    """Route an attention PE with co-located QK^T MatMul, Softmax, and AV MatMul.

    SRAM slot layout:
      Slot 0: Q row (d_model) — scattered from Q collect
      Slot 1: K matrix (seq_len * d_model) — broadcast from K collect
      Slot 2: V matrix (seq_len * d_model) — broadcast from V collect
      Slot 3: QK^T result (seq_len)
      Slot 4: Softmax output (seq_len)
      Slot 5: AV result (d_model)
    """
    attn = node.data
    assert isinstance(attn, PlacedAttentionPeData)

    seq_len = attn.seq_len
    q_slot = 0
    k_slot = 1
    v_slot = 2
    qkt_output_slot = 3
    softmax_output_slot = 4
    av_output_slot = 5

    # --- QK^T MatMul: K @ Q → scores(seq_len) ---
    # matrix=K(seq_len, d_model), vector=Q(d_model), transpose=false → (seq_len,)
    # Two TaskConfig entries: trigger on Q (slot 0) and K (slot 1)
    for trigger in [q_slot, k_slot]:
        pe_tasks[node.coord].append(
            MatMulEntry(
                trigger_slot=trigger,
                matrix_slot=k_slot,
                vector_slot=q_slot,
                rows=seq_len,
                cols=attn.d_model,
                transpose=False,
                output_slot=qkt_output_slot,
                routes=[],  # local write only
            )
        )

    # --- Softmax ---
    if attn.softmax_id is not None:
        pe_tasks[node.coord].append(
            SoftmaxEntry(
                trigger_slot=qkt_output_slot,
                input_slot=qkt_output_slot,
                output_slot=softmax_output_slot,
            )
        )

    # --- AV MatMul: V^T @ softmax → output(d_model) ---
    # matrix=V(seq_len, d_model), vector=softmax(seq_len), transpose=true → (d_model,)
    if attn.av_matmul_id is not None:
        # Find outgoing edges from this attention PE (inter-group, to downstream)
        av_routes: list[BroadcastRoute] = []
        for e in spatial.edges:
            if e.src_node == node.id:
                dst = node_map[e.dst_node]
                if dst.kind != PlacedNodeKind.ATTENTION_PE:
                    hops = _generate_route_xy(node.coord, dst.coord)
                    av_routes.append(
                        BroadcastRoute(
                            dest=dst.coord,
                            hops=hops,
                            deliver_at=[],
                            payload_slot=e.dst_slot,
                        )
                    )

        av_routes = _try_linear_broadcast(node.coord, av_routes)

        # Two TaskConfig entries: trigger on V (slot 2) and softmax output (slot 4)
        for trigger in [v_slot, softmax_output_slot]:
            pe_tasks[node.coord].append(
                MatMulEntry(
                    trigger_slot=trigger,
                    matrix_slot=v_slot,
                    vector_slot=softmax_output_slot,
                    rows=seq_len,
                    cols=attn.d_model,
                    transpose=True,
                    output_slot=av_output_slot,
                    routes=av_routes,
                )
            )


def _route_add(
    node: PlacedNode,
    spatial: SpatialIR,
    node_map: dict[str, PlacedNode],
    pe_tasks: dict[tuple[int, int], list[TaskEntry]],
) -> None:
    """Route an ADD PE."""
    # Find incoming edges to determine input slots
    incoming = [e for e in spatial.edges if e.dst_node == node.id]

    # input_slot_a = dst_slot of first incoming, input_slot_b = dst_slot of second
    input_slot_a = incoming[0].dst_slot if len(incoming) > 0 else 0
    input_slot_b = incoming[1].dst_slot if len(incoming) > 1 else 1

    # Output slot is after the input slots
    output_slot = max(input_slot_a, input_slot_b) + 1

    # Build outgoing routes from edge dst_slot
    add_routes: list[BroadcastRoute] = []
    for e in spatial.edges:
        if e.src_node == node.id:
            dst = node_map[e.dst_node]
            hops = _generate_route_xy(node.coord, dst.coord)
            add_routes.append(
                BroadcastRoute(
                    dest=dst.coord,
                    hops=hops,
                    deliver_at=[],
                    payload_slot=e.dst_slot,
                )
            )

    add_routes = _try_linear_broadcast(node.coord, add_routes)

    # Generate one AddEntry per input slot. The runtime has_slot guard
    # ensures only the second trigger (when both inputs are present) computes.
    for trigger_slot in [input_slot_a, input_slot_b]:
        pe_tasks[node.coord].append(
            AddEntry(
                trigger_slot=trigger_slot,
                input_slot_a=input_slot_a,
                input_slot_b=input_slot_b,
                output_slot=output_slot,
                routes=add_routes,
            )
        )


def _try_linear_broadcast(
    source_coord: tuple[int, int],
    dests: list[BroadcastRoute],
) -> list[BroadcastRoute]:
    """Detect column-aligned broadcasts and collapse into broadcast routes.

    Groups destinations by ``(dest_x, payload_slot)`` and applies column
    broadcast within each group.  Each group that has 2+ destinations in the
    same column with the same payload_slot is replaced by 1-2 broadcast routes
    (one per Y-direction from the column entry point).  Groups with only 1
    destination are left as point-to-point.

    Returns the (possibly optimised) route list.
    """
    if len(dests) <= 1:
        return dests

    # Group by (dest_x, payload_slot)
    groups: dict[tuple[int, int], list[BroadcastRoute]] = defaultdict(list)
    for d in dests:
        key = (d.dest[0], d.payload_slot)
        groups[key].append(d)

    result: list[BroadcastRoute] = []
    for (dest_x, payload_slot), group_dests in groups.items():
        if len(group_dests) == 1:
            result.extend(group_dests)
            continue
        result.extend(_broadcast_single_column(source_coord, dest_x, payload_slot, group_dests))

    return result


def _broadcast_single_column(
    source_coord: tuple[int, int],
    dest_x: int,
    payload_slot: int,
    dests: list[BroadcastRoute],
) -> list[BroadcastRoute]:
    """Build 1-2 broadcast routes for destinations in a single column.

    All destinations must share the same X coordinate (dest_x) and
    payload_slot.  Partitions into north/south groups relative to the
    column entry point and builds a broadcast route per direction.
    """
    src_x, src_y = source_coord
    entry_y = src_y  # Y coordinate when we arrive at dest_x column

    # Horizontal hops to reach the target column
    x_hops: list[Direction] = []
    cx = src_x
    while cx != dest_x:
        if dest_x > cx:
            x_hops.append(Direction.EAST)
            cx += 1
        else:
            x_hops.append(Direction.WEST)
            cx -= 1

    # Partition into north / south / same-Y
    north: list[tuple[int, int]] = []
    south: list[tuple[int, int]] = []
    same: list[tuple[int, int]] = []

    for d in dests:
        dy = d.dest[1]
        if dy > entry_y:
            north.append(d.dest)
        elif dy < entry_y:
            south.append(d.dest)
        else:
            same.append(d.dest)

    if same and not north and not south:
        return dests

    # Merge same-Y destinations into whichever group exists, preferring north
    if same:
        if north:
            north = same + north
        elif south:
            south = same + south

    def _build_broadcast(
        group: list[tuple[int, int]],
        direction: Direction,
    ) -> BroadcastRoute:
        """Build one BroadcastRoute for a uni-directional column group."""
        if direction == Direction.NORTH:
            group.sort(key=lambda c: c[1])
        else:
            group.sort(key=lambda c: -c[1])

        farthest = group[-1]
        y_distance = abs(farthest[1] - entry_y)
        y_hops: list[Direction] = [direction] * y_distance
        hops = list(x_hops) + y_hops

        x_offset = len(x_hops)
        deliver_at: list[int] = []
        for coord in group[:-1]:
            y_dist = abs(coord[1] - entry_y)
            # After x_offset horizontal hops we enter the column at entry_y.
            # The first Y hop (index x_offset) arrives at entry_y ± 1.
            # A PE at distance y_dist is reached at hop x_offset + y_dist - 1.
            hop_idx = x_offset + y_dist - 1
            deliver_at.append(hop_idx)

        return BroadcastRoute(
            dest=farthest,
            hops=hops,
            deliver_at=deliver_at,
            payload_slot=payload_slot,
        )

    result: list[BroadcastRoute] = []
    if north:
        result.append(_build_broadcast(north, Direction.NORTH))
    if south:
        result.append(_build_broadcast(south, Direction.SOUTH))

    return result


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
