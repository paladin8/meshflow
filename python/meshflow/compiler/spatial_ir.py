"""Spatial IR — graph with placement (each node assigned a PE coordinate)."""

from dataclasses import dataclass, field

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


@dataclass
class PlacedRmsNormTileData:
    """Typed data for an RMSNORM tile PE."""

    tile_index: int
    feature_slice_size: int
    feature_slice_offset: int
    origin_id: str


@dataclass
class PlacedRmsNormReduceData:
    """Typed data for an RMSNORM reduce PE."""

    num_tiles: int
    feature_count: int
    eps: float
    origin_id: str


@dataclass
class PlacedAttentionPeData:
    """Typed data for an attention PE (MATMUL + co-located SOFTMAX)."""

    pe_index: int
    seq_len: int
    origin_id: str
    softmax_id: str | None = None
    av_matmul_id: str | None = None


PlacedNodeData = (
    PlacedTileData
    | PlacedCollectData
    | PlacedRmsNormTileData
    | PlacedRmsNormReduceData
    | PlacedAttentionPeData
    | None
)


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
    # Mapping from original GraphIR node IDs to expanded PE IDs.
    # Populated for transformer/mixed graphs.
    node_pe_map: dict[str, list[str]] = field(default_factory=dict)
