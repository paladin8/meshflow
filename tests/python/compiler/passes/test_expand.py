"""Tests for the expansion pass."""

import pytest

from meshflow.compiler.config import CompilerConfig
from meshflow.compiler.expanded_ir import (
    AttentionGroup,
    PassthroughGroup,
    RmsNormGroup,
    TiledComputeGroup,
)
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType
from meshflow.compiler.passes.expand import expand


def _linear_groups(expanded):
    """Extract TiledComputeGroups from an expanded IR."""
    return [g for g in expanded.groups if isinstance(g, TiledComputeGroup)]


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
        assert len(expanded.groups) == 2
        assert all(isinstance(g, PassthroughGroup) for g in expanded.groups)
        assert len(expanded.original_edges) == 1

    def test_empty_graph(self):
        graph = GraphIR(nodes=[], edges=[])
        expanded = expand(graph, CompilerConfig())
        assert len(expanded.groups) == 0

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
        ids = [g.origin_id for g in expanded.groups]
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
        groups = _linear_groups(expanded)
        assert len(groups) == 1
        group = groups[0]
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
        group = _linear_groups(expanded)[0]
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
        tiles = _linear_groups(expanded)[0].tiles
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
        tiles = _linear_groups(expanded)[0].tiles
        assert [t.tile_rows for t in tiles] == [3, 2, 2]
        assert [t.fragment_offset for t in tiles] == [0, 3, 5]

    def test_in_features_propagated(self):
        graph = GraphIR(
            nodes=[Node(id="lin", op=OpType.LINEAR, attrs={"in_features": 16, "out_features": 4})],
            edges=[],
        )
        expanded = expand(graph, CompilerConfig())
        for tile in _linear_groups(expanded)[0].tiles:
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
        groups = _linear_groups(expanded)
        assert len(groups) == 1
        assert groups[0].collect.activation == "relu"

    def test_linear_no_activation(self):
        graph = GraphIR(
            nodes=[Node(id="lin", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 4})],
            edges=[],
        )
        expanded = expand(graph, CompilerConfig())
        assert _linear_groups(expanded)[0].collect.activation is None


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
        groups = _linear_groups(expanded)
        assert len(groups) == 2
        assert groups[0].origin_id == "l1"
        assert groups[0].next_group == "l2"
        assert groups[1].origin_id == "l2"
        assert groups[1].next_group is None

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
        groups = _linear_groups(expanded)
        assert len(groups) == 3
        assert groups[0].collect.activation == "relu"
        assert groups[0].next_group == "l2"
        assert groups[1].collect.activation == "relu"
        assert groups[1].next_group == "l3"
        assert groups[2].collect.activation is None
        assert groups[2].next_group is None

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
        groups = _linear_groups(expanded)
        assert [g.origin_id for g in groups] == ["l1", "l2"]


class TestExpandRmsNorm:
    """RMSNORM expansion into tile PEs + reduce PE."""

    def test_rmsnorm_creates_group(self):
        graph = GraphIR(
            nodes=[
                Node(id="fwd", op=OpType.FORWARD),
                Node(id="rn", op=OpType.RMSNORM, attrs={"eps": 1e-6, "feature_count": 8}),
                Node(id="col", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="fwd", src_slot=0, dst_node="rn", dst_slot=0),
                Edge(src_node="rn", src_slot=0, dst_node="col", dst_slot=0),
            ],
        )
        expanded = expand(graph, CompilerConfig())
        assert len(expanded.groups) == 3

        rn_group = next(g for g in expanded.groups if isinstance(g, RmsNormGroup))
        assert rn_group.origin_id == "rn"
        assert rn_group.num_tiles == 8
        assert rn_group.feature_count == 8
        assert rn_group.eps == 1e-6

    def test_rmsnorm_tile_count_limited_by_mesh(self):
        graph = GraphIR(
            nodes=[Node(id="rn", op=OpType.RMSNORM, attrs={"eps": 1e-6, "feature_count": 16})],
            edges=[],
        )
        config = CompilerConfig(mesh_height=5)  # 4 tiles + 1 reduce
        expanded = expand(graph, config)

        rn_group = next(g for g in expanded.groups if isinstance(g, RmsNormGroup))
        assert rn_group.num_tiles == 4

    def test_rmsnorm_node_expansions(self):
        graph = GraphIR(
            nodes=[Node(id="rn", op=OpType.RMSNORM, attrs={"eps": 1e-6, "feature_count": 4})],
            edges=[],
        )
        expanded = expand(graph, CompilerConfig())
        exp = expanded.node_expansions["rn"]
        assert exp.input_pe_ids == ["rn_tile_0", "rn_tile_1", "rn_tile_2", "rn_tile_3"]
        assert exp.output_pe_ids == ["rn_tile_0", "rn_tile_1", "rn_tile_2", "rn_tile_3"]


class TestExpandMatMul:
    """MATMUL expansion into attention PEs."""

    def test_matmul_creates_attention_group(self):
        graph = GraphIR(
            nodes=[Node(id="mm", op=OpType.MATMUL, attrs={"seq_len": 4})],
            edges=[],
        )
        expanded = expand(graph, CompilerConfig())

        assert len(expanded.groups) == 1
        group = expanded.groups[0]
        assert isinstance(group, AttentionGroup)
        assert group.seq_len == 4
        assert group.origin_id == "mm"

    def test_attention_chain_detected(self):
        """MATMUL → SOFTMAX → MATMUL fuses into one AttentionGroup."""
        graph = GraphIR(
            nodes=[
                Node(id="qkt", op=OpType.MATMUL, attrs={"seq_len": 4}),
                Node(id="sm", op=OpType.SOFTMAX),
                Node(id="av", op=OpType.MATMUL, attrs={"seq_len": 4}),
            ],
            edges=[
                Edge(src_node="qkt", src_slot=0, dst_node="sm", dst_slot=0),
                Edge(src_node="sm", src_slot=0, dst_node="av", dst_slot=0),
            ],
        )
        expanded = expand(graph, CompilerConfig())

        # Should produce 1 group (SOFTMAX and AV are fused)
        assert len(expanded.groups) == 1
        group = expanded.groups[0]
        assert isinstance(group, AttentionGroup)
        assert group.origin_id == "qkt"
        assert group.softmax_id == "sm"
        assert group.av_matmul_id == "av"

    def test_attention_chain_node_expansions(self):
        """Co-located nodes share the same PE IDs."""
        graph = GraphIR(
            nodes=[
                Node(id="qkt", op=OpType.MATMUL, attrs={"seq_len": 2}),
                Node(id="sm", op=OpType.SOFTMAX),
                Node(id="av", op=OpType.MATMUL, attrs={"seq_len": 2}),
            ],
            edges=[
                Edge(src_node="qkt", src_slot=0, dst_node="sm", dst_slot=0),
                Edge(src_node="sm", src_slot=0, dst_node="av", dst_slot=0),
            ],
        )
        expanded = expand(graph, CompilerConfig())
        pe_ids = ["qkt_attn_0", "qkt_attn_1"]
        assert expanded.node_expansions["qkt"].input_pe_ids == pe_ids
        assert expanded.node_expansions["sm"].input_pe_ids == pe_ids
        assert expanded.node_expansions["av"].input_pe_ids == pe_ids


class TestExpandAdd:
    """ADD expansion as single-PE passthrough."""

    def test_add_creates_passthrough(self):
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.FORWARD),
                Node(id="add", op=OpType.ADD),
            ],
            edges=[
                Edge(src_node="a", src_slot=0, dst_node="add", dst_slot=0),
                Edge(src_node="b", src_slot=0, dst_node="add", dst_slot=1),
            ],
        )
        expanded = expand(graph, CompilerConfig())

        add_group = next(g for g in expanded.groups if g.origin_id == "add")
        assert isinstance(add_group, PassthroughGroup)
        assert add_group.op == OpType.ADD

    def test_add_node_expansions(self):
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.FORWARD),
                Node(id="add", op=OpType.ADD),
            ],
            edges=[
                Edge(src_node="a", src_slot=0, dst_node="add", dst_slot=0),
                Edge(src_node="b", src_slot=0, dst_node="add", dst_slot=1),
            ],
        )
        expanded = expand(graph, CompilerConfig())
        exp = expanded.node_expansions["add"]
        assert exp.input_pe_ids == ["add"]
        assert exp.output_pe_ids == ["add"]
