"""Placement pass — assigns each graph node to a PE coordinate."""

from meshflow.compiler.config import CompilerConfig, PlacementStrategy
from meshflow.compiler.graph_ir import GraphIR
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
    """Sequential placement: assign coords in topological order, row-major."""
    topo_order = graph.topological_order()
    node_map = {n.id: n for n in graph.nodes}

    # Determine mesh dimensions
    n_nodes = len(topo_order)
    if n_nodes == 0:
        width = config.mesh_width or 1
        height = config.mesh_height or 1
    else:
        width = config.mesh_width or n_nodes
        height = config.mesh_height or 1

    # Assign coords in row-major order
    placed_nodes: list[PlacedNode] = []
    for i, nid in enumerate(topo_order):
        x = i % width
        y = i // width
        if x >= width or y >= height:
            raise ValueError(f"node {nid!r} at index {i} does not fit in {width}x{height} mesh")
        node = node_map[nid]
        placed_nodes.append(PlacedNode(id=nid, op=node.op, coord=(x, y)))

    # Edges carry over unchanged (they reference nodes by ID)
    placed_edges = [
        PlacedEdge(
            src_node=e.src_node,
            src_slot=e.src_slot,
            dst_node=e.dst_node,
            dst_slot=e.dst_slot,
        )
        for e in graph.edges
    ]

    return SpatialIR(
        width=width,
        height=height,
        nodes=placed_nodes,
        edges=placed_edges,
    )
