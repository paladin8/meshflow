"""Spatial IR — graph with placement (each node assigned a PE coordinate)."""

from dataclasses import dataclass
from typing import Any

from meshflow.compiler.graph_ir import OpType


@dataclass
class PlacedNode:
    id: str
    op: OpType
    coord: tuple[int, int]
    attrs: dict[str, Any] | None = None


@dataclass
class PlacedEdge:
    src_node: str
    src_slot: int
    dst_node: str
    dst_slot: int


@dataclass
class SpatialIR:
    width: int
    height: int
    nodes: list[PlacedNode]
    edges: list[PlacedEdge]
