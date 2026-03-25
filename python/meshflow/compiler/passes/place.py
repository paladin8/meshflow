"""Placement pass — assigns each expanded node/group to a PE coordinate."""

from meshflow.compiler.config import CompilerConfig, PlacementStrategy
from meshflow.compiler.expanded_ir import (
    AttentionGroup,
    ExpandedIR,
    PassthroughGroup,
    RmsNormGroup,
    TiledComputeGroup,
)
from meshflow.compiler.graph_ir import OpType
from meshflow.compiler.spatial_ir import (
    PlacedAttentionPeData,
    PlacedCollectData,
    PlacedEdge,
    PlacedNode,
    PlacedNodeKind,
    PlacedRmsNormReduceData,
    PlacedRmsNormTileData,
    PlacedTileData,
    SpatialIR,
)


def place(expanded: ExpandedIR, config: CompilerConfig) -> SpatialIR:
    """Place expanded nodes/groups onto a 2D mesh grid."""
    if config.placement == PlacementStrategy.SEQUENTIAL:
        return _place_sequential(expanded, config)
    raise ValueError(f"unknown placement strategy: {config.placement!r}")


def _place_sequential(expanded: ExpandedIR, config: CompilerConfig) -> SpatialIR:
    """Sequential placement.

    If all groups are single-PE passthrough, use row-major layout.
    Otherwise use column-per-group vertical layout.
    """
    all_passthrough = all(isinstance(g, PassthroughGroup) for g in expanded.groups)
    if all_passthrough:
        return _place_row_major(expanded, config)
    return _place_columns(expanded, config)


# ---------------------------------------------------------------------------
# Column-per-group placement
# ---------------------------------------------------------------------------


def _place_columns(expanded: ExpandedIR, config: CompilerConfig) -> SpatialIR:
    """Column-per-group placement for operator graphs.

    Each group gets one column. PEs stack vertically.
    Inter-group edges derived from original GraphIR edges + node expansions.
    """
    placed_nodes: list[PlacedNode] = []
    placed_edges: list[PlacedEdge] = []
    node_pe_map: dict[str, list[str]] = {}

    col = 0
    max_column_height = 0

    for group in expanded.groups:
        if isinstance(group, TiledComputeGroup):
            nodes, edges, height = _place_tiled_compute_group(group, col)
            placed_nodes.extend(nodes)
            placed_edges.extend(edges)
            tile_ids = [f"{group.origin_id}_tile_{t.tile_index}" for t in group.tiles]
            collect_id = f"{group.origin_id}_collect"
            node_pe_map[group.origin_id] = tile_ids + [collect_id]

        elif isinstance(group, RmsNormGroup):
            nodes, edges, height = _place_rmsnorm_group(group, col)
            placed_nodes.extend(nodes)
            placed_edges.extend(edges)
            tile_ids = [f"{group.origin_id}_tile_{i}" for i in range(group.num_tiles)]
            reduce_id = f"{group.origin_id}_reduce"
            collect_id = f"{group.origin_id}_collect"
            node_pe_map[group.origin_id] = tile_ids + [reduce_id, collect_id]

        elif isinstance(group, AttentionGroup):
            nodes, edges_attn, height = _place_attention_group(group, col)
            placed_nodes.extend(nodes)
            placed_edges.extend(edges_attn)
            pe_ids = [f"{group.origin_id}_attn_{i}" for i in range(group.seq_len)]
            has_collect = group.av_matmul_id is not None and group.seq_len > 1
            if has_collect:
                collect_id = f"{group.origin_id}_collect"
                node_pe_map[group.origin_id] = pe_ids + [collect_id]
            else:
                node_pe_map[group.origin_id] = pe_ids
            if group.softmax_id:
                node_pe_map[group.softmax_id] = pe_ids
            if group.av_matmul_id:
                if has_collect:
                    node_pe_map[group.av_matmul_id] = pe_ids + [collect_id]
                else:
                    node_pe_map[group.av_matmul_id] = pe_ids

        elif isinstance(group, PassthroughGroup):
            node = PlacedNode(
                id=group.origin_id,
                kind=_passthrough_kind(group.op),
                coord=(col, 0),
            )
            placed_nodes.append(node)
            node_pe_map[group.origin_id] = [group.origin_id]
            height = 1

        else:
            raise ValueError(f"unknown group type: {type(group)}")

        if height > max_column_height:
            max_column_height = height
        col += 1

    # Generate inter-group edges from original GraphIR edges + node expansions
    placed_edges.extend(_generate_inter_group_edges(expanded, node_pe_map))

    width = col if col > 0 else 1
    height = max_column_height if max_column_height > 0 else 1
    if config.mesh_height is not None and height < config.mesh_height:
        height = config.mesh_height

    return SpatialIR(
        width=width,
        height=height,
        nodes=placed_nodes,
        edges=placed_edges,
        node_pe_map=node_pe_map,
    )


def _place_tiled_compute_group(
    group: TiledComputeGroup, col: int
) -> tuple[list[PlacedNode], list[PlacedEdge], int]:
    """Place a LINEAR tiled compute group in a column."""
    nodes: list[PlacedNode] = []
    edges: list[PlacedEdge] = []

    for spec in group.tiles:
        tile_id = f"{group.origin_id}_tile_{spec.tile_index}"
        nodes.append(
            PlacedNode(
                id=tile_id,
                kind=PlacedNodeKind.LINEAR_TILE,
                coord=(col, spec.tile_index),
                data=PlacedTileData(
                    tile_index=spec.tile_index,
                    tile_rows=spec.tile_rows,
                    fragment_offset=spec.fragment_offset,
                    in_features=spec.in_features,
                    origin_id=group.origin_id,
                ),
            )
        )

    collect_row = len(group.tiles)
    collect_id = f"{group.origin_id}_collect"
    nodes.append(
        PlacedNode(
            id=collect_id,
            kind=PlacedNodeKind.LINEAR_COLLECT,
            coord=(col, collect_row),
            data=PlacedCollectData(
                num_fragments=group.collect.num_fragments,
                total_rows=group.collect.total_rows,
                origin_id=group.origin_id,
                activation=group.collect.activation,
            ),
        )
    )

    # Internal edges: tile → collect
    for i in range(len(group.tiles)):
        edges.append(
            PlacedEdge(
                src_node=f"{group.origin_id}_tile_{i}",
                src_slot=0,
                dst_node=collect_id,
                dst_slot=i,
            )
        )

    return nodes, edges, collect_row + 1


def _place_rmsnorm_group(
    group: RmsNormGroup, col: int
) -> tuple[list[PlacedNode], list[PlacedEdge], int]:
    """Place an RMSNORM group: tile PEs + reduce PE in a column."""
    nodes: list[PlacedNode] = []
    edges: list[PlacedEdge] = []

    base = group.feature_count // group.num_tiles
    remainder = group.feature_count % group.num_tiles

    offset = 0
    for i in range(group.num_tiles):
        slice_size = base + 1 if i < remainder else base
        tile_id = f"{group.origin_id}_tile_{i}"
        nodes.append(
            PlacedNode(
                id=tile_id,
                kind=PlacedNodeKind.RMSNORM_TILE,
                coord=(col, i),
                data=PlacedRmsNormTileData(
                    tile_index=i,
                    feature_slice_size=slice_size,
                    feature_slice_offset=offset,
                    origin_id=group.origin_id,
                ),
            )
        )
        offset += slice_size

    reduce_row = group.num_tiles
    reduce_id = f"{group.origin_id}_reduce"
    nodes.append(
        PlacedNode(
            id=reduce_id,
            kind=PlacedNodeKind.RMSNORM_REDUCE,
            coord=(col, reduce_row),
            data=PlacedRmsNormReduceData(
                num_tiles=group.num_tiles,
                feature_count=group.feature_count,
                eps=group.eps,
                origin_id=group.origin_id,
            ),
        )
    )

    # Internal edges: tile → reduce (partial sums, phase 1)
    for i in range(group.num_tiles):
        edges.append(
            PlacedEdge(
                src_node=f"{group.origin_id}_tile_{i}",
                src_slot=0,
                dst_node=reduce_id,
                dst_slot=i,
            )
        )

    # Internal edges: reduce → tile (scale factor broadcast, phase 2)
    for i in range(group.num_tiles):
        edges.append(
            PlacedEdge(
                src_node=reduce_id,
                src_slot=i,
                dst_node=f"{group.origin_id}_tile_{i}",
                dst_slot=1,  # scale factor arrives in slot 1 on tile PEs
            )
        )

    # Collect PE gathers normalized fragments from all tiles
    collect_row = reduce_row + 1
    collect_id = f"{group.origin_id}_collect"
    nodes.append(
        PlacedNode(
            id=collect_id,
            kind=PlacedNodeKind.LINEAR_COLLECT,
            coord=(col, collect_row),
            data=PlacedCollectData(
                num_fragments=group.num_tiles,
                total_rows=group.feature_count,
                origin_id=group.origin_id,
                activation=None,
            ),
        )
    )

    # Internal edges: tile → collect (normalized fragments)
    for i in range(group.num_tiles):
        edges.append(
            PlacedEdge(
                src_node=f"{group.origin_id}_tile_{i}",
                src_slot=1,  # normalize output
                dst_node=collect_id,
                dst_slot=i,
            )
        )

    return nodes, edges, collect_row + 1


def _place_attention_group(
    group: AttentionGroup, col: int
) -> tuple[list[PlacedNode], list[PlacedEdge], int]:
    """Place attention PEs + collect PE in a column."""
    nodes: list[PlacedNode] = []
    edges: list[PlacedEdge] = []

    for i in range(group.seq_len):
        pe_id = f"{group.origin_id}_attn_{i}"
        nodes.append(
            PlacedNode(
                id=pe_id,
                kind=PlacedNodeKind.ATTENTION_PE,
                coord=(col, i),
                data=PlacedAttentionPeData(
                    pe_index=i,
                    seq_len=group.seq_len,
                    d_model=group.d_model,
                    origin_id=group.origin_id,
                    softmax_id=group.softmax_id,
                    av_matmul_id=group.av_matmul_id,
                ),
            )
        )

    # Collect PE gathers attention outputs (only needed for attention chains)
    if group.av_matmul_id is not None and group.seq_len > 1:
        collect_row = group.seq_len
        collect_id = f"{group.origin_id}_collect"
        nodes.append(
            PlacedNode(
                id=collect_id,
                kind=PlacedNodeKind.LINEAR_COLLECT,
                coord=(col, collect_row),
                data=PlacedCollectData(
                    num_fragments=group.seq_len,
                    total_rows=group.seq_len * group.d_model,
                    origin_id=group.origin_id,
                    activation=None,
                ),
            )
        )

        for i in range(group.seq_len):
            edges.append(
                PlacedEdge(
                    src_node=f"{group.origin_id}_attn_{i}",
                    src_slot=0,
                    dst_node=collect_id,
                    dst_slot=i,
                )
            )

        return nodes, edges, collect_row + 1

    return nodes, edges, group.seq_len


def _generate_inter_group_edges(
    expanded: ExpandedIR,
    node_pe_map: dict[str, list[str]],
) -> list[PlacedEdge]:
    """Generate inter-group edges from original GraphIR edges.

    Maps each original edge through node expansions to produce placed edges.
    Handles 1:1, broadcast (1→N), gather (N→1), and parallel (N→N) patterns.
    Filters out self-edges (same PE on both ends).
    """
    edges: list[PlacedEdge] = []

    for orig_edge in expanded.original_edges:
        src_expansion = expanded.node_expansions.get(orig_edge.src_node)
        dst_expansion = expanded.node_expansions.get(orig_edge.dst_node)

        if src_expansion is None or dst_expansion is None:
            continue

        src_pes = src_expansion.output_pe_ids
        dst_pes = dst_expansion.input_pe_ids

        if len(src_pes) == 1 and len(dst_pes) == 1:
            if src_pes[0] != dst_pes[0]:
                edges.append(
                    PlacedEdge(
                        src_node=src_pes[0],
                        src_slot=0,
                        dst_node=dst_pes[0],
                        dst_slot=orig_edge.dst_slot,
                    )
                )
        elif len(src_pes) == 1 and len(dst_pes) > 1:
            # Broadcast: one source to many destinations
            for i, dst_pe in enumerate(dst_pes):
                if src_pes[0] != dst_pe:
                    edges.append(
                        PlacedEdge(
                            src_node=src_pes[0],
                            src_slot=i,
                            dst_node=dst_pe,
                            dst_slot=orig_edge.dst_slot,
                        )
                    )
        elif len(src_pes) == len(dst_pes):
            # 1:1 parallel
            for src_pe, dst_pe in zip(src_pes, dst_pes):
                if src_pe != dst_pe:
                    edges.append(
                        PlacedEdge(
                            src_node=src_pe,
                            src_slot=0,
                            dst_node=dst_pe,
                            dst_slot=orig_edge.dst_slot,
                        )
                    )
        elif len(src_pes) > 1 and len(dst_pes) == 1:
            # Gather: many sources to one destination
            for i, src_pe in enumerate(src_pes):
                if src_pe != dst_pes[0]:
                    edges.append(
                        PlacedEdge(
                            src_node=src_pe,
                            src_slot=0,
                            dst_node=dst_pes[0],
                            dst_slot=i,
                        )
                    )
        else:
            # Mismatched counts — broadcast from each source
            for i, dst_pe in enumerate(dst_pes):
                for j, src_pe in enumerate(src_pes):
                    if src_pe != dst_pe:
                        edges.append(
                            PlacedEdge(
                                src_node=src_pe,
                                src_slot=i,
                                dst_node=dst_pe,
                                dst_slot=j,
                            )
                        )

    return edges


_PASSTHROUGH_KIND_MAP = {
    OpType.FORWARD: PlacedNodeKind.FORWARD,
    OpType.COLLECT: PlacedNodeKind.COLLECT_SIMPLE,
    OpType.ADD: PlacedNodeKind.ADD,
    OpType.SOFTMAX: PlacedNodeKind.SOFTMAX,
}


def _passthrough_kind(op: OpType) -> PlacedNodeKind:
    """Map an OpType to the corresponding PlacedNodeKind for passthrough nodes."""
    kind = _PASSTHROUGH_KIND_MAP.get(op)
    if kind is None:
        raise ValueError(f"no passthrough PlacedNodeKind for {op!r}")
    return kind


# ---------------------------------------------------------------------------
# Row-major placement for passthrough-only graphs
# ---------------------------------------------------------------------------


def _place_row_major(expanded: ExpandedIR, config: CompilerConfig) -> SpatialIR:
    """Row-major placement for passthrough (FORWARD/COLLECT) graphs."""
    passthrough_nodes = [g for g in expanded.groups if isinstance(g, PassthroughGroup)]
    total = len(passthrough_nodes)
    if total == 0:
        width = config.mesh_width or 1
        height = config.mesh_height or 1
    else:
        width = config.mesh_width or total
        height = config.mesh_height or 1

    placed_nodes: list[PlacedNode] = []
    placed_edges: list[PlacedEdge] = []
    node_pe_map: dict[str, list[str]] = {}

    for idx, group in enumerate(passthrough_nodes):
        x = idx % width
        y = idx // width
        if x >= width or y >= height:
            raise ValueError(
                f"node {group.origin_id!r} at index {idx} does not fit in {width}x{height} mesh"
            )
        placed_nodes.append(
            PlacedNode(id=group.origin_id, kind=_passthrough_kind(group.op), coord=(x, y))
        )
        node_pe_map[group.origin_id] = [group.origin_id]

    # Map original edges directly
    for e in expanded.original_edges:
        placed_edges.append(
            PlacedEdge(
                src_node=e.src_node,
                src_slot=e.src_slot,
                dst_node=e.dst_node,
                dst_slot=e.dst_slot,
            )
        )

    return SpatialIR(
        width=width,
        height=height,
        nodes=placed_nodes,
        edges=placed_edges,
        node_pe_map=node_pe_map,
    )
