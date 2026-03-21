"""Placement pass — assigns each graph node to a PE coordinate."""

from meshflow.compiler.config import CompilerConfig, PlacementStrategy
from meshflow.compiler.graph_ir import GraphIR, OpType
from meshflow.compiler.spatial_ir import PlacedEdge, PlacedNode, SpatialIR


def place(graph: GraphIR, config: CompilerConfig) -> SpatialIR:
    """Place graph nodes onto a 2D mesh grid.

    Dispatches on config.placement strategy.
    """
    graph.validate()

    if config.placement == PlacementStrategy.SEQUENTIAL:
        return _place_sequential(graph, config)

    raise ValueError(f"unknown placement strategy: {config.placement!r}")


def _place_sequential(graph: GraphIR, config: CompilerConfig) -> SpatialIR:
    """Sequential placement: assign coords in topological order.

    LINEAR nodes use vertical column layout (tiles stack on y-axis, one column per layer).
    FORWARD/COLLECT nodes use horizontal row-major layout (M2 compat).
    """
    topo_order = graph.topological_order()
    node_map = {n.id: n for n in graph.nodes}

    has_linear = any(node_map[nid].op == OpType.LINEAR for nid in topo_order)

    if has_linear:
        return _place_linear_columns(graph, config, topo_order, node_map)
    else:
        return _place_row_major(graph, config, topo_order, node_map)


def _place_row_major(
    graph: GraphIR,
    config: CompilerConfig,
    topo_order: list[str],
    node_map: dict,
) -> SpatialIR:
    """Row-major placement for FORWARD/COLLECT graphs (M2 compat)."""
    total = len(topo_order)
    if total == 0:
        width = config.mesh_width or 1
        height = config.mesh_height or 1
    else:
        width = config.mesh_width or total
        height = config.mesh_height or 1

    placed_nodes: list[PlacedNode] = []
    placed_edges: list[PlacedEdge] = []
    idx = 0

    for nid in topo_order:
        node = node_map[nid]
        x = idx % width
        y = idx // width
        if x >= width or y >= height:
            raise ValueError(f"node {nid!r} at index {idx} does not fit in {width}x{height} mesh")
        placed_nodes.append(PlacedNode(id=nid, op=node.op, coord=(x, y)))
        idx += 1

    for e in graph.edges:
        placed_edges.append(
            PlacedEdge(
                src_node=e.src_node,
                src_slot=e.src_slot,
                dst_node=e.dst_node,
                dst_slot=e.dst_slot,
            )
        )

    return SpatialIR(width=width, height=height, nodes=placed_nodes, edges=placed_edges)


def _compute_tile_rows(out_features: int, num_tiles: int, tile_index: int) -> int:
    """Compute rows for a given tile with uneven distribution."""
    base = out_features // num_tiles
    remainder = out_features % num_tiles
    return base + 1 if tile_index < remainder else base


def _compute_fragment_offset(out_features: int, num_tiles: int, tile_index: int) -> int:
    """Compute fragment offset for a given tile with uneven distribution."""
    base = out_features // num_tiles
    remainder = out_features % num_tiles
    return tile_index * base + min(tile_index, remainder)


def _place_linear_columns(
    graph: GraphIR,
    config: CompilerConfig,
    topo_order: list[str],
    node_map: dict,
) -> SpatialIR:
    """Column-per-layer placement for LINEAR graphs.

    Each LINEAR node gets one column. Tiles stack vertically (y-axis).
    Collect PE sits at the top of each column.
    """
    placed_nodes: list[PlacedNode] = []
    placed_edges: list[PlacedEdge] = []
    col = 0
    max_column_height = 0

    for nid in topo_order:
        node = node_map[nid]

        if node.op != OpType.LINEAR:
            continue

        if node.attrs is None:
            raise ValueError(f"LINEAR node {nid!r} requires attrs")
        out_f = node.attrs["out_features"]
        in_f = node.attrs["in_features"]

        # Determine tile count from mesh_height
        if config.mesh_height is not None:
            num_tiles = min(config.mesh_height - 1, out_f)
        else:
            num_tiles = out_f

        if num_tiles < 1:
            raise ValueError(
                f"LINEAR node {nid!r}: need at least 1 tile, "
                f"but mesh_height={config.mesh_height} is too small"
            )

        # Place tile PEs vertically in this column
        for i in range(num_tiles):
            tile_rows = _compute_tile_rows(out_f, num_tiles, i)
            frag_offset = _compute_fragment_offset(out_f, num_tiles, i)
            tile_id = f"{nid}_tile_{i}"
            placed_nodes.append(
                PlacedNode(
                    id=tile_id,
                    op=OpType.LINEAR,
                    coord=(col, i),
                    attrs={
                        "tile_index": i,
                        "tile_rows": tile_rows,
                        "fragment_offset": frag_offset,
                        "in_features": in_f,
                        "origin_id": nid,
                    },
                )
            )

        # Place collect PE at top of column
        collect_row = num_tiles
        collect_id = f"{nid}_collect"
        placed_nodes.append(
            PlacedNode(
                id=collect_id,
                op=OpType.COLLECT,
                coord=(col, collect_row),
                attrs={
                    "num_fragments": num_tiles,
                    "total_rows": out_f,
                    "origin_id": nid,
                },
            )
        )

        # Internal edges: each tile -> collect
        for i in range(num_tiles):
            placed_edges.append(
                PlacedEdge(
                    src_node=f"{nid}_tile_{i}",
                    src_slot=0,
                    dst_node=collect_id,
                    dst_slot=i,
                )
            )

        column_height = num_tiles + 1
        if column_height > max_column_height:
            max_column_height = column_height
        col += 1

    # Carry over original graph edges (for inter-layer connections).
    # NOTE: These edges reference original GraphIR node IDs (e.g., "linear1")
    # which no longer exist after LINEAR expansion. For single-layer graphs
    # this loop is a no-op. Multi-layer support (Checkpoint 2) must remap
    # these edges: outgoing from LINEAR → collect PE, incoming to LINEAR →
    # tile PEs, RELU edges absorbed into collect PE attrs.
    for e in graph.edges:
        placed_edges.append(
            PlacedEdge(
                src_node=e.src_node,
                src_slot=e.src_slot,
                dst_node=e.dst_node,
                dst_slot=e.dst_slot,
            )
        )

    width = col if col > 0 else 1
    height = max_column_height if max_column_height > 0 else 1
    # Pad to mesh_height if specified
    if config.mesh_height is not None and height < config.mesh_height:
        height = config.mesh_height

    return SpatialIR(width=width, height=height, nodes=placed_nodes, edges=placed_edges)
