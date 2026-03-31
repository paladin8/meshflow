"""Expansion pass — expands operator nodes into physical PE group structures."""

from typing import Any

from meshflow.compiler.config import CompilerConfig
from meshflow.compiler.expanded_ir import (
    AttentionGroup,
    CollectSpec,
    ExpandedIR,
    NodeExpansion,
    PassthroughGroup,
    RmsNormGroup,
    TiledComputeGroup,
    TileSpec,
)
from meshflow.compiler.graph_ir import GraphIR, OpType


def expand(graph: GraphIR, config: CompilerConfig) -> ExpandedIR:
    """Expand graph operator nodes into physical PE groups.

    LINEAR nodes become TiledComputeGroups (tiles + collect).
    Activation nodes following LINEAR are fused onto the collect spec.
    FORWARD/COLLECT nodes become PassthroughGroups.
    Transformer ops (MATMUL, SOFTMAX, RMSNORM, ADD) expand into
    their respective group types.
    """
    graph.validate()
    topo_order = graph.topological_order()
    node_map = {n.id: n for n in graph.nodes}

    # Detect fusion pairs
    fused_activations = _detect_relu_fusions(graph, topo_order, node_map)
    attention_chains = _detect_attention_chains(graph, topo_order, node_map)
    fused_nodes: set[str] = set()
    fused_nodes.update(fused_activations.values())
    for chain_info in attention_chains.values():
        if chain_info["softmax_id"]:
            fused_nodes.add(chain_info["softmax_id"])
        if chain_info["av_matmul_id"]:
            fused_nodes.add(chain_info["av_matmul_id"])

    groups: list = []
    node_expansions: dict[str, NodeExpansion] = {}

    for nid in topo_order:
        if nid in fused_nodes:
            continue
        node = node_map[nid]

        if node.op in (OpType.FORWARD, OpType.COLLECT):
            groups.append(PassthroughGroup(origin_id=nid, op=node.op))
            node_expansions[nid] = NodeExpansion(input_pe_ids=[nid], output_pe_ids=[nid])

        elif node.op == OpType.LINEAR:
            group = _make_linear_group(node, config, graph, fused_activations, node_map)
            tile_ids = [f"{nid}_tile_{t.tile_index}" for t in group.tiles]
            collect_id = f"{nid}_collect"
            node_expansions[nid] = NodeExpansion(input_pe_ids=tile_ids, output_pe_ids=[collect_id])
            # If activation was fused, map the activation node too
            if nid in fused_activations:
                act_nid = fused_activations[nid]
                node_expansions[act_nid] = NodeExpansion(
                    input_pe_ids=[collect_id], output_pe_ids=[collect_id]
                )
            groups.append(group)

        elif node.op == OpType.RELU:
            # Standalone RELU not preceded by LINEAR — error
            raise ValueError(f"RELU node {nid!r} is not preceded by a LINEAR node")

        elif node.op == OpType.RMSNORM:
            assert node.attrs is not None
            feature_count = node.attrs["feature_count"]
            eps = node.attrs["eps"]
            # Fused single-PE RMSNorm: no tiles or reduce PE needed.
            groups.append(
                RmsNormGroup(
                    origin_id=nid,
                    num_tiles=0,
                    feature_count=feature_count,
                    eps=eps,
                )
            )
            norm_id = f"{nid}_norm"
            node_expansions[nid] = NodeExpansion(input_pe_ids=[norm_id], output_pe_ids=[norm_id])

        elif node.op == OpType.MATMUL:
            if nid in attention_chains:
                chain = attention_chains[nid]
                assert node.attrs is not None
                seq_len = node.attrs["seq_len"]
                d_model = node.attrs.get("d_model", 0)
                attn_group = AttentionGroup(
                    origin_id=nid,
                    seq_len=seq_len,
                    d_model=d_model,
                    softmax_id=chain["softmax_id"],
                    av_matmul_id=chain["av_matmul_id"],
                )
                pe_ids = [f"{nid}_attn_{i}" for i in range(seq_len)]
                node_expansions[nid] = NodeExpansion(input_pe_ids=pe_ids, output_pe_ids=pe_ids)
                if chain["softmax_id"]:
                    node_expansions[chain["softmax_id"]] = NodeExpansion(
                        input_pe_ids=pe_ids, output_pe_ids=pe_ids
                    )
                if chain["av_matmul_id"]:
                    if seq_len > 1:
                        # Multi-PE: AV output goes through attention collect PE
                        collect_id = f"{nid}_collect"
                        node_expansions[chain["av_matmul_id"]] = NodeExpansion(
                            input_pe_ids=pe_ids, output_pe_ids=[collect_id]
                        )
                    else:
                        # Single PE: AV output goes directly downstream
                        node_expansions[chain["av_matmul_id"]] = NodeExpansion(
                            input_pe_ids=pe_ids, output_pe_ids=pe_ids
                        )
                groups.append(attn_group)
            else:
                assert node.attrs is not None
                seq_len = node.attrs["seq_len"]
                d_model = node.attrs.get("d_model", 0)
                attn_group = AttentionGroup(origin_id=nid, seq_len=seq_len, d_model=d_model)
                pe_ids = [f"{nid}_attn_{i}" for i in range(seq_len)]
                node_expansions[nid] = NodeExpansion(input_pe_ids=pe_ids, output_pe_ids=pe_ids)
                groups.append(attn_group)

        elif node.op == OpType.ADD:
            # ADD is a single-PE operation: receives two inputs, adds them,
            # broadcasts to downstream tiles. No tiling needed — the upstream
            # collect PE already gathers into a single vector.
            groups.append(PassthroughGroup(origin_id=nid, op=node.op))
            node_expansions[nid] = NodeExpansion(input_pe_ids=[nid], output_pe_ids=[nid])

        elif node.op == OpType.SOFTMAX:
            # Standalone SOFTMAX (not co-located on attention PE)
            groups.append(PassthroughGroup(origin_id=nid, op=node.op))
            node_expansions[nid] = NodeExpansion(input_pe_ids=[nid], output_pe_ids=[nid])

    return ExpandedIR(
        groups=groups,
        node_expansions=node_expansions,
        original_edges=list(graph.edges),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_relu_fusions(graph: GraphIR, topo_order: list[str], node_map: dict) -> dict[str, str]:
    """Detect LINEAR → RELU fusion pairs.

    Returns a dict mapping LINEAR node ID → fused RELU node ID.
    """
    activation_for_linear: dict[str, str] = {}
    for nid in topo_order:
        node = node_map[nid]
        if node.op != OpType.LINEAR:
            continue
        outgoing = [e for e in graph.edges if e.src_node == nid]
        if len(outgoing) == 1:
            successor = node_map[outgoing[0].dst_node]
            if successor.op.is_activation:
                activation_for_linear[nid] = successor.id
    return activation_for_linear


def _detect_attention_chains(
    graph: GraphIR, topo_order: list[str], node_map: dict
) -> dict[str, dict[str, str | None]]:
    """Detect MATMUL → SOFTMAX → MATMUL attention chains.

    Returns a dict mapping the first MATMUL ID → chain info:
    {"softmax_id": str | None, "av_matmul_id": str | None}
    """
    chains: dict[str, dict[str, str | None]] = {}
    for nid in topo_order:
        node = node_map[nid]
        if node.op != OpType.MATMUL:
            continue
        if nid in chains:
            continue
        already_claimed = any(c.get("av_matmul_id") == nid for c in chains.values())
        if already_claimed:
            continue

        outgoing = [e for e in graph.edges if e.src_node == nid]
        softmax_id: str | None = None
        av_matmul_id: str | None = None
        if len(outgoing) == 1:
            succ = node_map[outgoing[0].dst_node]
            if succ.op == OpType.SOFTMAX:
                softmax_id = succ.id
                softmax_out = [e for e in graph.edges if e.src_node == succ.id]
                if len(softmax_out) == 1:
                    av = node_map[softmax_out[0].dst_node]
                    if av.op == OpType.MATMUL:
                        av_matmul_id = av.id
        chains[nid] = {"softmax_id": softmax_id, "av_matmul_id": av_matmul_id}

    return chains


def _compute_tile_count(feature_count: int, config: CompilerConfig, reserved_rows: int = 1) -> int:
    """Compute the number of tiles for a given feature count.

    ``reserved_rows`` is the number of rows in the column reserved for
    non-tile PEs (e.g. 1 for a collect or reduce PE).
    """
    if config.mesh_height is not None:
        return min(config.mesh_height - reserved_rows, feature_count)
    return feature_count


def _make_linear_group(
    node: Any,
    config: CompilerConfig,
    graph: GraphIR,
    fused_activations: dict[str, str],
    node_map: dict,
) -> TiledComputeGroup:
    """Create a TiledComputeGroup for a LINEAR node."""
    nid = node.id
    attrs = node.attrs
    if attrs is None:
        raise ValueError(f"LINEAR node {nid!r} requires attrs")
    out_f = attrs["out_features"]
    in_f = attrs["in_features"]

    num_tiles = _compute_tile_count(out_f, config)
    if num_tiles < 1:
        raise ValueError(
            f"LINEAR node {nid!r}: need at least 1 tile, "
            f"but mesh_height={config.mesh_height} is too small"
        )

    tiles = _make_tiles(out_f, num_tiles, in_f)

    activation = None
    if nid in fused_activations:
        act_node = node_map[fused_activations[nid]]
        activation = act_node.op.value
    collect = CollectSpec(
        num_fragments=num_tiles,
        total_rows=out_f,
        activation=activation,
    )

    # Determine next group (LINEAR → LINEAR chain)
    effective_src = fused_activations.get(nid, nid)
    next_outgoing = [e for e in graph.edges if e.src_node == effective_src]
    next_group = None
    if next_outgoing:
        next_node = node_map[next_outgoing[0].dst_node]
        if next_node.op == OpType.LINEAR:
            next_group = next_node.id

    return TiledComputeGroup(
        origin_id=nid,
        tiles=tiles,
        collect=collect,
        next_group=next_group,
    )


def _make_tiles(out_features: int, num_tiles: int, in_features: int) -> list[TileSpec]:
    """Build tile specs with even distribution of output rows."""
    tiles: list[TileSpec] = []
    base = out_features // num_tiles
    remainder = out_features % num_tiles
    for i in range(num_tiles):
        tile_rows = base + 1 if i < remainder else base
        frag_offset = i * base + min(i, remainder)
        tiles.append(
            TileSpec(
                tile_index=i,
                tile_rows=tile_rows,
                fragment_offset=frag_offset,
                in_features=in_features,
            )
        )
    return tiles
