"""Tests for the expansion pass."""

import pytest

from meshflow.compiler.config import CompilerConfig
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType
from meshflow.compiler.passes.expand import expand


class TestExpandPassthrough:
    """FORWARD/COLLECT graphs pass through unchanged."""

    def test_forward_collect_passthrough(self):
        graph = GraphIR(
            nodes=[
                Node(id="fwd", op=OpType.FORWARD),
                Node(id="col", op=OpType.COLLECT),
            ],
            edges=[Edge(src_node="fwd", src_slot=0, dst_node="col", dst_slot=0)],
        )
        expanded = expand(graph, CompilerConfig())
        assert len(expanded.groups) == 0
        assert len(expanded.passthrough_nodes) == 2
        assert len(expanded.passthrough_edges) == 1

    def test_empty_graph(self):
        graph = GraphIR(nodes=[], edges=[])
        expanded = expand(graph, CompilerConfig())
        assert len(expanded.groups) == 0
        assert len(expanded.passthrough_nodes) == 0

    def test_passthrough_preserves_topological_order(self):
        """Passthrough nodes should be in topological order."""
        graph = GraphIR(
            nodes=[
                Node(id="c", op=OpType.COLLECT),
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.FORWARD),
            ],
            edges=[
                Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0),
                Edge(src_node="b", src_slot=0, dst_node="c", dst_slot=0),
            ],
        )
        expanded = expand(graph, CompilerConfig())
        ids = [n.id for n in expanded.passthrough_nodes]
        assert ids == ["a", "b", "c"]


class TestExpandSingleLinear:
    """Single LINEAR node expansion."""

    def test_single_linear_default_tiles(self):
        """Without mesh_height, each output row gets its own tile."""
        graph = GraphIR(
            nodes=[Node(id="lin", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 3})],
            edges=[],
        )
        expanded = expand(graph, CompilerConfig())
        assert len(expanded.groups) == 1
        group = expanded.groups[0]
        assert group.origin_id == "lin"
        assert len(group.tiles) == 3
        assert group.collect.num_fragments == 3
        assert group.collect.total_rows == 3
        assert group.next_group is None

    def test_single_linear_with_mesh_height(self):
        """mesh_height limits tile count."""
        graph = GraphIR(
            nodes=[Node(id="lin", op=OpType.LINEAR, attrs={"in_features": 8, "out_features": 6})],
            edges=[],
        )
        config = CompilerConfig(mesh_height=4)  # 3 tiles + 1 collect
        expanded = expand(graph, config)
        group = expanded.groups[0]
        assert len(group.tiles) == 3
        assert group.collect.num_fragments == 3
        assert group.collect.total_rows == 6

    def test_tile_rows_even_split(self):
        """Even split: 6 rows / 3 tiles = 2 rows each."""
        graph = GraphIR(
            nodes=[Node(id="lin", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 6})],
            edges=[],
        )
        config = CompilerConfig(mesh_height=4)
        expanded = expand(graph, config)
        tiles = expanded.groups[0].tiles
        assert [t.tile_rows for t in tiles] == [2, 2, 2]
        assert [t.fragment_offset for t in tiles] == [0, 2, 4]

    def test_tile_rows_uneven_split(self):
        """Uneven split: 7 rows / 3 tiles = [3, 2, 2]."""
        graph = GraphIR(
            nodes=[Node(id="lin", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 7})],
            edges=[],
        )
        config = CompilerConfig(mesh_height=4)
        expanded = expand(graph, config)
        tiles = expanded.groups[0].tiles
        assert [t.tile_rows for t in tiles] == [3, 2, 2]
        assert [t.fragment_offset for t in tiles] == [0, 3, 5]

    def test_in_features_propagated(self):
        graph = GraphIR(
            nodes=[Node(id="lin", op=OpType.LINEAR, attrs={"in_features": 16, "out_features": 4})],
            edges=[],
        )
        expanded = expand(graph, CompilerConfig())
        for tile in expanded.groups[0].tiles:
            assert tile.in_features == 16

    def test_mesh_height_too_small(self):
        """mesh_height=1 leaves 0 tiles — should error."""
        graph = GraphIR(
            nodes=[Node(id="lin", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 4})],
            edges=[],
        )
        config = CompilerConfig(mesh_height=1)
        with pytest.raises(ValueError, match="need at least 1 tile"):
            expand(graph, config)


class TestExpandActivationFusion:
    """LINEAR→activation fusion."""

    def test_linear_relu_fused(self):
        graph = GraphIR(
            nodes=[
                Node(id="lin", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 4}),
                Node(id="relu", op=OpType.RELU),
            ],
            edges=[Edge(src_node="lin", src_slot=0, dst_node="relu", dst_slot=0)],
        )
        expanded = expand(graph, CompilerConfig())
        assert len(expanded.groups) == 1
        assert expanded.groups[0].collect.activation == "relu"

    def test_linear_no_activation(self):
        graph = GraphIR(
            nodes=[Node(id="lin", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 4})],
            edges=[],
        )
        expanded = expand(graph, CompilerConfig())
        assert expanded.groups[0].collect.activation is None


class TestExpandMultiLayer:
    """Multi-layer LINEAR chains."""

    def test_two_layer_chain(self):
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 6}),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 6, "out_features": 3}),
            ],
            edges=[Edge(src_node="l1", src_slot=0, dst_node="l2", dst_slot=0)],
        )
        expanded = expand(graph, CompilerConfig())
        assert len(expanded.groups) == 2
        assert expanded.groups[0].origin_id == "l1"
        assert expanded.groups[0].next_group == "l2"
        assert expanded.groups[1].origin_id == "l2"
        assert expanded.groups[1].next_group is None

    def test_three_layer_with_relu(self):
        """L1 → RELU → L2 → RELU → L3"""
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 6}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 6, "out_features": 6}),
                Node(id="r2", op=OpType.RELU),
                Node(id="l3", op=OpType.LINEAR, attrs={"in_features": 6, "out_features": 3}),
            ],
            edges=[
                Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0),
                Edge(src_node="r1", src_slot=0, dst_node="l2", dst_slot=0),
                Edge(src_node="l2", src_slot=0, dst_node="r2", dst_slot=0),
                Edge(src_node="r2", src_slot=0, dst_node="l3", dst_slot=0),
            ],
        )
        expanded = expand(graph, CompilerConfig())
        assert len(expanded.groups) == 3
        assert expanded.groups[0].collect.activation == "relu"
        assert expanded.groups[0].next_group == "l2"
        assert expanded.groups[1].collect.activation == "relu"
        assert expanded.groups[1].next_group == "l3"
        assert expanded.groups[2].collect.activation is None
        assert expanded.groups[2].next_group is None

    def test_groups_preserve_topological_order(self):
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 4}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 4}),
            ],
            edges=[
                Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0),
                Edge(src_node="r1", src_slot=0, dst_node="l2", dst_slot=0),
            ],
        )
        expanded = expand(graph, CompilerConfig())
        assert [g.origin_id for g in expanded.groups] == ["l1", "l2"]
