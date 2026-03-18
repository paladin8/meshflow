"""Meshflow compiler — Graph IR to RuntimeProgram artifact."""

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


def compile(graph: GraphIR, config: CompilerConfig | None = None) -> RuntimeProgram:
    """Compile a GraphIR through all passes into a RuntimeProgram artifact."""
    if config is None:
        config = CompilerConfig()

    spatial = place(graph, config)
    schedule = route(spatial, config)
    return lower(schedule)
