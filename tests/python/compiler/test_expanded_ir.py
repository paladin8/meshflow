"""Tests for ExpandedIR data types."""

from meshflow.compiler.expanded_ir import (
    CollectSpec,
    ExpandedIR,
    PassthroughGroup,
    TiledComputeGroup,
    TileSpec,
)
from meshflow.compiler.graph_ir import Edge, OpType


class TestTileSpec:
    def test_construction(self):
        spec = TileSpec(tile_index=0, tile_rows=4, fragment_offset=0, in_features=8)
        assert spec.tile_index == 0
        assert spec.tile_rows == 4
        assert spec.fragment_offset == 0
        assert spec.in_features == 8

    def test_uneven_tile(self):
        spec = TileSpec(tile_index=2, tile_rows=3, fragment_offset=9, in_features=8)
        assert spec.tile_rows == 3
        assert spec.fragment_offset == 9


class TestCollectSpec:
    def test_construction(self):
        spec = CollectSpec(num_fragments=3, total_rows=10)
        assert spec.num_fragments == 3
        assert spec.total_rows == 10
        assert spec.activation is None

    def test_with_activation(self):
        spec = CollectSpec(num_fragments=2, total_rows=8, activation="relu")
        assert spec.activation == "relu"


class TestTiledComputeGroup:
    def test_single_tile(self):
        group = TiledComputeGroup(
            origin_id="linear1",
            tiles=[TileSpec(tile_index=0, tile_rows=4, fragment_offset=0, in_features=8)],
            collect=CollectSpec(num_fragments=1, total_rows=4),
        )
        assert group.origin_id == "linear1"
        assert len(group.tiles) == 1
        assert group.next_group is None

    def test_with_next_group(self):
        group = TiledComputeGroup(
            origin_id="linear1",
            tiles=[TileSpec(tile_index=0, tile_rows=4, fragment_offset=0, in_features=8)],
            collect=CollectSpec(num_fragments=1, total_rows=4, activation="relu"),
            next_group="linear2",
        )
        assert group.next_group == "linear2"
        assert group.collect.activation == "relu"


class TestExpandedIR:
    def test_empty(self):
        ir = ExpandedIR()
        assert len(ir.groups) == 0

    def test_passthrough(self):
        edges = [Edge(src_node="fwd", src_slot=0, dst_node="col", dst_slot=0)]
        ir = ExpandedIR(
            groups=[PassthroughGroup(origin_id="fwd", op=OpType.FORWARD)],
            original_edges=edges,
        )
        assert len(ir.groups) == 1
        assert len(ir.original_edges) == 1

    def test_with_groups(self):
        group = TiledComputeGroup(
            origin_id="linear1",
            tiles=[
                TileSpec(tile_index=0, tile_rows=2, fragment_offset=0, in_features=4),
                TileSpec(tile_index=1, tile_rows=2, fragment_offset=2, in_features=4),
            ],
            collect=CollectSpec(num_fragments=2, total_rows=4),
        )
        ir = ExpandedIR(groups=[group])
        assert len(ir.groups) == 1
        assert ir.groups[0].collect.num_fragments == 2  # type: ignore[union-attr]
