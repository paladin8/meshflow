"""Spatial IR — graph with placement (each node assigned a PE coordinate)."""

from dataclasses import dataclass

from meshflow.compiler.graph_ir import OpType


@dataclass
class PlacedTileData:
    """Typed data for a LINEAR tile PE."""

    tile_index: int
    tile_rows: int
    fragment_offset: int
    in_features: int
    origin_id: str


@dataclass
class PlacedCollectData:
    """Typed data for a COLLECT PE from a tiled operator group."""

    num_fragments: int
    total_rows: int
    origin_id: str
    activation: str | None = None


PlacedNodeData = PlacedTileData | PlacedCollectData | None


@dataclass
class PlacedNode:
    id: str
    op: OpType
    coord: tuple[int, int]
    data: PlacedNodeData = None


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
