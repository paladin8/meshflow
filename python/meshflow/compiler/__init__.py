"""Meshflow compiler — Graph IR to RuntimeProgram artifact."""

import numpy as np

from meshflow.compiler.artifact import RuntimeProgram
from meshflow.compiler.config import CompilerConfig, PlacementStrategy, RoutingStrategy
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType
from meshflow.compiler.passes import color, expand, lower, place, route

__all__ = [
    "CompilerConfig",
    "Edge",
    "GraphIR",
    "Node",
    "OpType",
    "PlacementStrategy",
    "RoutingStrategy",
    "RuntimeProgram",
    "compile",
]


def compile(
    graph: GraphIR,
    config: CompilerConfig | None = None,
    weights: dict[str, dict[str, np.ndarray]] | None = None,
) -> RuntimeProgram:
    """Compile a GraphIR through all passes into a RuntimeProgram artifact.

    For LINEAR nodes, `weights` maps node IDs to {"weight": W, "bias": b}
    where W is shape (out_features, in_features) and b is shape (out_features,).
    """
    if config is None:
        config = CompilerConfig()

    graph.validate()
    _validate_weights(graph, weights)
    _validate_shape_chaining(graph)

    expanded = expand(graph, config)
    spatial = place(expanded, config)
    schedule = route(spatial, config, weights)
    schedule = color(schedule, config)
    return lower(schedule, config)


def _validate_weights(
    graph: GraphIR,
    weights: dict[str, dict[str, np.ndarray]] | None,
) -> None:
    """Validate that weights are provided for LINEAR and RMSNORM nodes."""
    for node in graph.nodes:
        if node.op == OpType.LINEAR:
            if node.attrs is None:
                raise ValueError(f"LINEAR node {node.id!r} requires attrs")
            in_f = node.attrs["in_features"]
            out_f = node.attrs["out_features"]

            if weights is None or node.id not in weights:
                raise ValueError(f"LINEAR node {node.id!r} requires weights")

            w_dict = weights[node.id]
            if "weight" not in w_dict:
                raise ValueError(f"LINEAR node {node.id!r} missing 'weight' in weights")
            if "bias" not in w_dict:
                raise ValueError(f"LINEAR node {node.id!r} missing 'bias' in weights")

            w = w_dict["weight"]
            if w.shape != (out_f, in_f):
                raise ValueError(
                    f"LINEAR node {node.id!r}: weight shape {w.shape} doesn't match ({out_f}, {in_f})"
                )

            b = w_dict["bias"]
            if b.shape != (out_f,):
                raise ValueError(
                    f"LINEAR node {node.id!r}: bias shape {b.shape} doesn't match ({out_f},)"
                )

        elif node.op == OpType.RMSNORM:
            if node.attrs is None:
                raise ValueError(f"RMSNORM node {node.id!r} requires attrs")
            fc = node.attrs["feature_count"]

            if weights is None or node.id not in weights:
                raise ValueError(f"RMSNORM node {node.id!r} requires weights")

            w_dict = weights[node.id]
            if "gamma" not in w_dict:
                raise ValueError(f"RMSNORM node {node.id!r} missing 'gamma' in weights")

            g = w_dict["gamma"]
            if g.shape != (fc,):
                raise ValueError(
                    f"RMSNORM node {node.id!r}: gamma shape {g.shape} doesn't match ({fc},)"
                )


def _validate_shape_chaining(graph: GraphIR) -> None:
    """Validate that connected LINEAR layers have compatible dimensions."""
    node_map = {n.id: n for n in graph.nodes}

    for node in graph.nodes:
        if node.op != OpType.LINEAR:
            continue
        if node.attrs is None:
            continue
        out_f = node.attrs["out_features"]

        # Follow outgoing edges through activation nodes to the next LINEAR
        outgoing = [e for e in graph.edges if e.src_node == node.id]
        for edge in outgoing:
            successor = node_map[edge.dst_node]
            # Skip through activation nodes to find the next LINEAR
            if successor.op.is_activation:
                act_outgoing = [e for e in graph.edges if e.src_node == successor.id]
                if not act_outgoing:
                    continue
                successor = node_map[act_outgoing[0].dst_node]

            if successor.op == OpType.LINEAR and successor.attrs is not None:
                in_f = successor.attrs["in_features"]
                if out_f != in_f:
                    raise ValueError(
                        f"shape mismatch: {node.id!r} out_features={out_f} != "
                        f"{successor.id!r} in_features={in_f}"
                    )
