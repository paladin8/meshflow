"""Expanded IR — operator nodes expanded into physical PE groups, before placement."""

from dataclasses import dataclass, field

from meshflow.compiler.graph_ir import Edge, Node


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
class ExpandedIR:
    """IR after operator expansion, before coordinate assignment.

    For LINEAR graphs: groups contain the tiled structure.
    For FORWARD/COLLECT graphs: passthrough_nodes/edges carry the original graph.
    """

    groups: list[TiledComputeGroup] = field(default_factory=list)
    passthrough_nodes: list[Node] = field(default_factory=list)
    passthrough_edges: list[Edge] = field(default_factory=list)
