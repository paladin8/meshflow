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
    """Sequential placement: assign coords in topological order, row-major.

    LINEAR nodes are expanded into N tile PEs + 1 collect PE.
    """
    topo_order = graph.topological_order()
    node_map = {n.id: n for n in graph.nodes}

    # First pass: count total placed nodes (LINEAR expands)
    total_placed = 0
    for nid in topo_order:
        node = node_map[nid]
        if node.op == OpType.LINEAR:
            if node.attrs is None:
                raise ValueError(f"LINEAR node {nid!r} requires attrs")
            out_f = node.attrs["out_features"]
            if config.mesh_width is not None:
                num_tiles = config.mesh_width - 1
            else:
                num_tiles = out_f
            total_placed += num_tiles + 1  # tiles + collect
        else:
            total_placed += 1

    # Determine mesh dimensions
    if total_placed == 0:
        width = config.mesh_width or 1
        height = config.mesh_height or 1
    else:
        width = config.mesh_width or total_placed
        height = config.mesh_height or 1

    # Second pass: assign coordinates
    placed_nodes: list[PlacedNode] = []
    placed_edges: list[PlacedEdge] = []
    col_idx = 0

    for nid in topo_order:
        node = node_map[nid]

        if node.op == OpType.LINEAR:
            if node.attrs is None:
                raise ValueError(f"LINEAR node {nid!r} requires attrs")
            out_f = node.attrs["out_features"]
            in_f = node.attrs["in_features"]
            num_tiles = width - 1 if config.mesh_width is not None else out_f
            rows_per_pe = out_f // num_tiles
            if out_f % num_tiles != 0:
                raise ValueError(
                    f"LINEAR node {nid!r}: out_features={out_f} not evenly "
                    f"divisible by num_tiles={num_tiles}"
                )

            # Place tile PEs
            for i in range(num_tiles):
                x = col_idx % width
                y = col_idx // width
                if x >= width or y >= height:
                    raise ValueError(
                        f"tile {nid}_tile_{i} at index {col_idx} "
                        f"does not fit in {width}x{height} mesh"
                    )
                tile_id = f"{nid}_tile_{i}"
                placed_nodes.append(
                    PlacedNode(
                        id=tile_id,
                        op=OpType.LINEAR,
                        coord=(x, y),
                        attrs={
                            "tile_index": i,
                            "rows_per_pe": rows_per_pe,
                            "in_features": in_f,
                            "origin_id": nid,
                        },
                    )
                )
                col_idx += 1

            # Place collect PE
            x = col_idx % width
            y = col_idx // width
            if x >= width or y >= height:
                raise ValueError(
                    f"collect {nid}_collect at index {col_idx} "
                    f"does not fit in {width}x{height} mesh"
                )
            collect_id = f"{nid}_collect"
            placed_nodes.append(
                PlacedNode(
                    id=collect_id,
                    op=OpType.COLLECT,
                    coord=(x, y),
                    attrs={
                        "num_fragments": num_tiles,
                        "rows_per_fragment": rows_per_pe,
                        "origin_id": nid,
                    },
                )
            )
            col_idx += 1

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
        else:
            # FORWARD / COLLECT — same as M2
            x = col_idx % width
            y = col_idx // width
            if x >= width or y >= height:
                raise ValueError(
                    f"node {nid!r} at index {col_idx} does not fit in {width}x{height} mesh"
                )
            placed_nodes.append(PlacedNode(id=nid, op=node.op, coord=(x, y)))
            col_idx += 1

    # Carry over original graph edges (remap for non-LINEAR nodes)
    for e in graph.edges:
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
    )
