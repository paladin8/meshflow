"""Expanded IR — operator nodes expanded into physical PE groups, before placement."""

from dataclasses import dataclass, field

from meshflow.compiler.graph_ir import Edge, OpType


@dataclass
class TileSpec:
    """Specification for a single compute tile within a tiled operator group."""

    tile_index: int
    tile_rows: int
    fragment_offset: int
    in_features: int


@dataclass
class CollectSpec:
    """Specification for the collect PE that gathers tile fragments."""

    num_fragments: int
    total_rows: int
    activation: str | None = None


@dataclass
class TiledComputeGroup:
    """A LINEAR operator expanded into parallel compute tiles + a collect PE.

    Each tile computes a fragment of the output (y_i = W_i @ x + b_i).
    The collect PE gathers fragments into the full output vector.
    """

    origin_id: str
    tiles: list[TileSpec]
    collect: CollectSpec
    next_group: str | None = None


@dataclass
class RmsNormGroup:
    """RMSNORM expanded into tile PEs + a reduce PE."""

    origin_id: str
    num_tiles: int
    feature_count: int
    eps: float


@dataclass
class AttentionGroup:
    """MATMUL attention PEs for row-parallel attention.

    SOFTMAX and AV MATMUL are co-located on the same PEs.
    """

    origin_id: str
    seq_len: int
    d_model: int = 0
    softmax_id: str | None = None
    av_matmul_id: str | None = None


@dataclass
class PassthroughGroup:
    """A single-PE group for FORWARD, COLLECT, or standalone SOFTMAX nodes."""

    origin_id: str
    op: OpType


@dataclass
class NodeExpansion:
    """Tracks how an original GraphIR node was expanded into PE IDs."""

    input_pe_ids: list[str]
    output_pe_ids: list[str]


OperatorGroup = TiledComputeGroup | RmsNormGroup | AttentionGroup | PassthroughGroup


@dataclass
class ExpandedIR:
    """IR after operator expansion, before coordinate assignment.

    ``groups`` is an ordered list of operator groups in topological order.
    ``node_expansions`` maps original GraphIR node IDs to their expanded PE IDs.
    ``original_edges`` preserves the GraphIR edges for inter-group connectivity.
    """

    groups: list[OperatorGroup] = field(default_factory=list)
    node_expansions: dict[str, NodeExpansion] = field(default_factory=dict)
    original_edges: list[Edge] = field(default_factory=list)
