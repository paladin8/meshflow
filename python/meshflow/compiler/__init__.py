"""Meshflow compiler — Graph IR to RuntimeProgram artifact."""

import numpy as np

from meshflow.compiler.artifact import RuntimeProgram
from meshflow.compiler.config import CompilerConfig, PlacementStrategy, RoutingStrategy
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType
from meshflow.compiler.passes import lower, place, route

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

    _validate_weights(graph, weights)

    spatial = place(graph, config)
    schedule = route(spatial, config, weights)
    return lower(schedule)


def _validate_weights(
    graph: GraphIR,
    weights: dict[str, dict[str, np.ndarray]] | None,
) -> None:
    """Validate that weights are provided for all LINEAR nodes with correct shapes."""
    for node in graph.nodes:
        if node.op != OpType.LINEAR:
            continue
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
