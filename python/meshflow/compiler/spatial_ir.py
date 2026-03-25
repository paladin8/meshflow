"""Spatial IR — graph with placement (each node assigned a PE coordinate)."""

from dataclasses import dataclass, field
from enum import Enum


class PlacedNodeKind(Enum):
    """Granular kind for placed PEs — disambiguates subtypes within an OpType."""

    FORWARD = "forward"
    COLLECT_SIMPLE = "collect_simple"
    LINEAR_TILE = "linear_tile"
    LINEAR_COLLECT = "linear_collect"
    RMSNORM_TILE = "rmsnorm_tile"
    RMSNORM_REDUCE = "rmsnorm_reduce"
    ATTENTION_PE = "attention_pe"
    ADD = "add"
    SOFTMAX = "softmax"


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
    d_model: int
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
    kind: PlacedNodeKind
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
    node_pe_map: dict[str, list[str]] = field(default_factory=dict)
