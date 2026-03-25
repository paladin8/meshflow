"""Routing pass — generates concrete hop lists and flattens to per-PE schedules."""

import numpy as np

from meshflow.compiler.config import CompilerConfig, RoutingStrategy
from meshflow.compiler.schedule_ir import (
    AddEntry,
    CollectOutputEntry,
    ConcatCollectEntry,
    ConcatCollectForwardEntry,
    Direction,
    ForwardActivationEntry,
    InputSlot,
    LinearEntry,
    MatMulEntry,
    PESchedule,
    RmsNormNormalizeEntry,
    RmsNormPartialSumEntry,
    RmsNormReduceEntry,
    ScheduleIR,
    SoftmaxEntry,
    TaskEntry,
)
from meshflow.compiler.spatial_ir import (
    PlacedAttentionPeData,
    PlacedCollectData,
    PlacedNode,
    PlacedNodeKind,
    PlacedRmsNormReduceData,
    PlacedRmsNormTileData,
    PlacedTileData,
    SpatialIR,
)


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
        if node.kind == PlacedNodeKind.FORWARD:
            outgoing = [e for e in spatial.edges if e.src_node == node.id]
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
                        route_dest=dst_node.coord,
                        route_hops=hops,
                    )
                )

        elif node.kind == PlacedNodeKind.LINEAR_TILE:
            assert isinstance(node.data, PlacedTileData)
            tile = node.data
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

        elif node.kind == PlacedNodeKind.LINEAR_COLLECT:
            assert isinstance(node.data, PlacedCollectData)
            collect = node.data

            # Determine intermediate vs terminal from explicit outgoing edges
            outgoing = [e for e in spatial.edges if e.src_node == node.id]
            is_intermediate = len(outgoing) > 0

            base = collect.total_rows // collect.num_fragments
            remainder = collect.total_rows % collect.num_fragments

            if is_intermediate:
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

        elif node.kind == PlacedNodeKind.RMSNORM_TILE:
            _route_rmsnorm_tile(node, spatial, node_map, pe_tasks, pe_sram, weights)

        elif node.kind == PlacedNodeKind.RMSNORM_REDUCE:
            _route_rmsnorm_reduce(node, spatial, node_map, pe_tasks)

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
        PlacedNodeKind.RMSNORM_TILE,
        PlacedNodeKind.RMSNORM_REDUCE,
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


def _route_rmsnorm_tile(
    node: PlacedNode,
    spatial: SpatialIR,
    node_map: dict[str, PlacedNode],
    pe_tasks: dict[tuple[int, int], list[TaskEntry]],
    pe_sram: dict[tuple[int, int], dict[int, list[float]]],
    weights: dict[str, dict[str, np.ndarray]] | None,
) -> None:
    """Route an RmsNorm tile PE: phase 1 (partial sum) and phase 2 (normalize)."""
    tile = node.data
    assert isinstance(tile, PlacedRmsNormTileData)

    # Find the reduce PE via internal edges (tile → reduce)
    reduce_edge = None
    for e in spatial.edges:
        if e.src_node == node.id:
            dst = node_map[e.dst_node]
            if dst.kind == PlacedNodeKind.RMSNORM_REDUCE:
                reduce_edge = e
                break

    if reduce_edge is None:
        raise ValueError(f"RmsNorm tile node {node.id!r} has no edge to reduce PE")

    reduce_node = node_map[reduce_edge.dst_node]
    reduce_hops = _generate_route_xy(node.coord, reduce_node.coord)

    # Phase 1: RmsNormPartialSum
    pe_tasks[node.coord].append(
        RmsNormPartialSumEntry(
            trigger_slot=0,
            input_slot=0,
            reduce_dest=reduce_node.coord,
            reduce_hops=reduce_hops,
            partial_sum_slot=tile.tile_index,
            slice_offset=tile.feature_slice_offset,
            slice_size=tile.feature_slice_size,
        )
    )

    # Phase 2: RmsNormNormalize — find outgoing inter-group edges (not internal)
    # Inter-group edges go to nodes that are NOT the reduce PE
    outgoing_dests: list[tuple[tuple[int, int], list[Direction]]] = []
    outgoing_payload_slots: list[int] = []
    for e in spatial.edges:
        if e.src_node == node.id:
            dst = node_map[e.dst_node]
            if dst.kind != PlacedNodeKind.RMSNORM_REDUCE:
                hops = _generate_route_xy(node.coord, dst.coord)
                outgoing_dests.append((dst.coord, hops))
                outgoing_payload_slots.append(e.dst_slot)

    pe_tasks[node.coord].append(
        RmsNormNormalizeEntry(
            trigger_slot=1,
            input_slot=0,
            scale_slot=1,
            gamma_slot=2,
            output_dests=outgoing_dests,
            payload_slots=outgoing_payload_slots,
            slice_offset=tile.feature_slice_offset,
            slice_size=tile.feature_slice_size,
        )
    )

    # Load gamma weights into SRAM slot 2
    if weights is not None and tile.origin_id in weights:
        gamma = weights[tile.origin_id]["gamma"]
        gamma_slice = (
            gamma[tile.feature_slice_offset : tile.feature_slice_offset + tile.feature_slice_size]
            .flatten()
            .tolist()
        )
        pe_sram[node.coord][2] = gamma_slice


def _route_rmsnorm_reduce(
    node: PlacedNode,
    spatial: SpatialIR,
    node_map: dict[str, PlacedNode],
    pe_tasks: dict[tuple[int, int], list[TaskEntry]],
) -> None:
    """Route an RmsNorm reduce PE."""
    reduce = node.data
    assert isinstance(reduce, PlacedRmsNormReduceData)

    # Find tile_dests: outgoing edges from reduce → tile PEs (scale broadcast)
    tile_dests: list[tuple[tuple[int, int], list[Direction]]] = []
    for e in spatial.edges:
        if e.src_node == node.id:
            dst = node_map[e.dst_node]
            if dst.kind == PlacedNodeKind.RMSNORM_TILE:
                hops = _generate_route_xy(node.coord, dst.coord)
                tile_dests.append((dst.coord, hops))

    # Generate one RmsNormReduceEntry per partial sum slot
    for i in range(reduce.num_tiles):
        pe_tasks[node.coord].append(
            RmsNormReduceEntry(
                trigger_slot=i,
                num_tiles=reduce.num_tiles,
                feature_count=reduce.feature_count,
                eps=reduce.eps,
                tile_dests=list(tile_dests),
                scale_slot=1,
            )
        )


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
                output_dests=[],  # local write only
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
        outgoing_dests: list[tuple[tuple[int, int], list[Direction]]] = []
        av_payload_slots: list[int] = []
        for e in spatial.edges:
            if e.src_node == node.id:
                dst = node_map[e.dst_node]
                if dst.kind != PlacedNodeKind.ATTENTION_PE:
                    hops = _generate_route_xy(node.coord, dst.coord)
                    outgoing_dests.append((dst.coord, hops))
                    av_payload_slots.append(e.dst_slot)

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
                    output_dests=outgoing_dests,
                    payload_slots=av_payload_slots,
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

    # Build outgoing destinations with payload_slots from edge dst_slot
    outgoing_dests: list[tuple[tuple[int, int], list[Direction]]] = []
    add_payload_slots: list[int] = []
    for e in spatial.edges:
        if e.src_node == node.id:
            dst = node_map[e.dst_node]
            hops = _generate_route_xy(node.coord, dst.coord)
            outgoing_dests.append((dst.coord, hops))
            add_payload_slots.append(e.dst_slot)

    # Generate one AddEntry per input slot. The runtime has_slot guard
    # ensures only the second trigger (when both inputs are present) computes.
    for trigger_slot in [input_slot_a, input_slot_b]:
        pe_tasks[node.coord].append(
            AddEntry(
                trigger_slot=trigger_slot,
                input_slot_a=input_slot_a,
                input_slot_b=input_slot_b,
                output_slot=output_slot,
                output_dests=outgoing_dests,
                payload_slots=add_payload_slots,
            )
        )


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
