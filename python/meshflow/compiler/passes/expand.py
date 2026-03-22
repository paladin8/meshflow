"""Expansion pass — expands operator nodes into physical PE group structures."""

from meshflow.compiler.config import CompilerConfig
from meshflow.compiler.expanded_ir import (
    CollectSpec,
    ExpandedIR,
    TiledComputeGroup,
    TileSpec,
)
from meshflow.compiler.graph_ir import GraphIR, OpType


def expand(graph: GraphIR, config: CompilerConfig) -> ExpandedIR:
    """Expand graph operator nodes into physical PE groups.

    LINEAR nodes become TiledComputeGroups (tiles + collect).
    Activation nodes following LINEAR are fused onto the collect spec.
    FORWARD/COLLECT nodes pass through unchanged.
    """
    graph.validate()
    topo_order = graph.topological_order()
    node_map = {n.id: n for n in graph.nodes}

    has_linear = any(node_map[nid].op == OpType.LINEAR for nid in topo_order)

    if has_linear:
        return _expand_linear(graph, config, topo_order, node_map)
    else:
        # Preserve topological order for deterministic placement
        return ExpandedIR(
            passthrough_nodes=[node_map[nid] for nid in topo_order],
            passthrough_edges=list(graph.edges),
        )


def _expand_linear(
    graph: GraphIR,
    config: CompilerConfig,
    topo_order: list[str],
    node_map: dict,
) -> ExpandedIR:
    """Expand LINEAR graph into tiled compute groups."""
    # Detect LINEAR → activation fusion pairs
    fused_activations: set[str] = set()
    activation_for_linear: dict[str, str] = {}
    for nid in topo_order:
        node = node_map[nid]
        if node.op != OpType.LINEAR:
            continue
        outgoing = [e for e in graph.edges if e.src_node == nid]
        if len(outgoing) == 1:
            successor = node_map[outgoing[0].dst_node]
            if successor.op.is_activation:
                fused_activations.add(successor.id)
                activation_for_linear[nid] = successor.id

    groups: list[TiledComputeGroup] = []

    for nid in topo_order:
        node = node_map[nid]

        if nid in fused_activations:
            continue
        if node.op != OpType.LINEAR:
            continue

        if node.attrs is None:
            raise ValueError(f"LINEAR node {nid!r} requires attrs")
        out_f = node.attrs["out_features"]
        in_f = node.attrs["in_features"]

        # Determine tile count
        if config.mesh_height is not None:
            num_tiles = min(config.mesh_height - 1, out_f)
        else:
            num_tiles = out_f

        if num_tiles < 1:
            raise ValueError(
                f"LINEAR node {nid!r}: need at least 1 tile, "
                f"but mesh_height={config.mesh_height} is too small"
            )

        # Build tile specs
        tiles: list[TileSpec] = []
        base = out_f // num_tiles
        remainder = out_f % num_tiles
        for i in range(num_tiles):
            tile_rows = base + 1 if i < remainder else base
            frag_offset = i * base + min(i, remainder)
            tiles.append(
                TileSpec(
                    tile_index=i,
                    tile_rows=tile_rows,
                    fragment_offset=frag_offset,
                    in_features=in_f,
                )
            )

        # Activation from fused node
        activation = None
        if nid in activation_for_linear:
            act_node = node_map[activation_for_linear[nid]]
            activation = act_node.op.value

        collect = CollectSpec(
            num_fragments=num_tiles,
            total_rows=out_f,
            activation=activation,
        )

        # Determine next group: follow outgoing edges (possibly through fused activation)
        effective_src = activation_for_linear.get(nid, nid)
        next_outgoing = [e for e in graph.edges if e.src_node == effective_src]
        next_group = None
        if next_outgoing:
            next_node = node_map[next_outgoing[0].dst_node]
            if next_node.op == OpType.LINEAR:
                next_group = next_node.id

        groups.append(
            TiledComputeGroup(
                origin_id=nid,
                tiles=tiles,
                collect=collect,
                next_group=next_group,
            )
        )

    return ExpandedIR(groups=groups)
