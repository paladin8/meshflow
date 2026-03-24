"""Tests for GraphIR construction and validation."""

import pytest
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType


class TestConstruction:
    def test_simple_chain(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.FORWARD),
                Node(id="c", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0),
                Edge(src_node="b", src_slot=0, dst_node="c", dst_slot=0),
            ],
        )
        graph.validate()
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2

    def test_single_node_no_edges(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="x", op=OpType.COLLECT)],
            edges=[],
        )
        graph.validate()

    def test_empty_graph(self) -> None:
        graph = GraphIR(nodes=[], edges=[])
        graph.validate()


class TestValidation:
    def test_duplicate_node_ids(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="a", op=OpType.COLLECT),
            ],
            edges=[],
        )
        with pytest.raises(ValueError, match="duplicate node id"):
            graph.validate()

    def test_edge_references_unknown_source(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="b", op=OpType.COLLECT)],
            edges=[Edge(src_node="missing", src_slot=0, dst_node="b", dst_slot=0)],
        )
        with pytest.raises(ValueError, match="unknown source node"):
            graph.validate()

    def test_edge_references_unknown_dest(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="a", op=OpType.FORWARD)],
            edges=[Edge(src_node="a", src_slot=0, dst_node="missing", dst_slot=0)],
        )
        with pytest.raises(ValueError, match="unknown destination node"):
            graph.validate()

    def test_cycle_detected(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.FORWARD),
            ],
            edges=[
                Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0),
                Edge(src_node="b", src_slot=0, dst_node="a", dst_slot=0),
            ],
        )
        with pytest.raises(ValueError, match="cycle"):
            graph.validate()

    def test_self_loop_detected(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="a", op=OpType.FORWARD)],
            edges=[Edge(src_node="a", src_slot=0, dst_node="a", dst_slot=0)],
        )
        with pytest.raises(ValueError, match="cycle"):
            graph.validate()

    def test_linear_valid(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="l", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 6})],
            edges=[],
        )
        graph.validate()  # should not raise

    def test_linear_missing_attrs(self) -> None:
        graph = GraphIR(nodes=[Node(id="l", op=OpType.LINEAR)], edges=[])
        with pytest.raises(ValueError, match="requires attrs"):
            graph.validate()

    def test_linear_missing_in_features(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="l", op=OpType.LINEAR, attrs={"out_features": 6})],
            edges=[],
        )
        with pytest.raises(ValueError, match="missing required attr.*in_features"):
            graph.validate()

    def test_linear_non_positive_attr(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="l", op=OpType.LINEAR, attrs={"in_features": 0, "out_features": 6})],
            edges=[],
        )
        with pytest.raises(ValueError, match="must be a positive integer"):
            graph.validate()


class TestInputNodes:
    def test_single_input(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.COLLECT),
            ],
            edges=[Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0)],
        )
        assert graph.input_node_ids() == ["a"]

    def test_multiple_inputs(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.FORWARD),
                Node(id="c", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="a", src_slot=0, dst_node="c", dst_slot=0),
                Edge(src_node="b", src_slot=0, dst_node="c", dst_slot=1),
            ],
        )
        assert graph.input_node_ids() == ["a", "b"]

    def test_all_nodes_are_inputs(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.COLLECT),
            ],
            edges=[],
        )
        assert graph.input_node_ids() == ["a", "b"]


class TestReluValidation:
    def test_relu_after_linear_valid(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 8}),
                Node(id="r1", op=OpType.RELU),
            ],
            edges=[Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0)],
        )
        graph.validate()  # should not raise

    def test_relu_not_after_linear(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="f", op=OpType.FORWARD),
                Node(id="r", op=OpType.RELU),
            ],
            edges=[Edge(src_node="f", src_slot=0, dst_node="r", dst_slot=0)],
        )
        with pytest.raises(ValueError, match="must follow a LINEAR"):
            graph.validate()

    def test_relu_no_incoming(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="r", op=OpType.RELU)],
            edges=[],
        )
        with pytest.raises(ValueError, match="exactly one incoming edge"):
            graph.validate()

    def test_relu_multiple_outgoing(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 8}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 8, "out_features": 4}),
                Node(id="l3", op=OpType.LINEAR, attrs={"in_features": 8, "out_features": 4}),
            ],
            edges=[
                Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0),
                Edge(src_node="r1", src_slot=0, dst_node="l2", dst_slot=0),
                Edge(src_node="r1", src_slot=1, dst_node="l3", dst_slot=0),
            ],
        )
        with pytest.raises(ValueError, match="at most one outgoing edge"):
            graph.validate()

    def test_mlp_graph_valid(self) -> None:
        """Full 2-layer MLP: Linear → ReLU → Linear."""
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 8}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 8, "out_features": 3}),
            ],
            edges=[
                Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0),
                Edge(src_node="r1", src_slot=0, dst_node="l2", dst_slot=0),
            ],
        )
        graph.validate()  # should not raise


class TestRmsNormValidation:
    def test_rmsnorm_valid(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="rn", op=OpType.RMSNORM, attrs={"eps": 1e-6, "feature_count": 8})],
            edges=[],
        )
        graph.validate()

    def test_rmsnorm_missing_eps(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="rn", op=OpType.RMSNORM, attrs={"feature_count": 8})],
            edges=[],
        )
        with pytest.raises(ValueError, match="requires 'eps'"):
            graph.validate()

    def test_rmsnorm_missing_feature_count(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="rn", op=OpType.RMSNORM, attrs={"eps": 1e-6})],
            edges=[],
        )
        with pytest.raises(ValueError, match="requires 'feature_count'"):
            graph.validate()

    def test_rmsnorm_no_attrs(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="rn", op=OpType.RMSNORM)],
            edges=[],
        )
        with pytest.raises(ValueError, match="requires 'eps'"):
            graph.validate()


class TestMatMulValidation:
    def test_matmul_valid(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="mm", op=OpType.MATMUL, attrs={"seq_len": 4})],
            edges=[],
        )
        graph.validate()

    def test_matmul_missing_seq_len(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="mm", op=OpType.MATMUL)],
            edges=[],
        )
        with pytest.raises(ValueError, match="requires 'seq_len'"):
            graph.validate()


class TestAddValidation:
    def test_add_valid(self) -> None:
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
        graph.validate()

    def test_add_wrong_edge_count(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="add", op=OpType.ADD),
            ],
            edges=[Edge(src_node="a", src_slot=0, dst_node="add", dst_slot=0)],
        )
        with pytest.raises(ValueError, match="exactly 2 incoming edges"):
            graph.validate()


class TestSoftmaxValidation:
    def test_softmax_valid(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="mm", op=OpType.MATMUL, attrs={"seq_len": 4}),
                Node(id="sm", op=OpType.SOFTMAX),
            ],
            edges=[Edge(src_node="mm", src_slot=0, dst_node="sm", dst_slot=0)],
        )
        graph.validate()

    def test_softmax_no_incoming(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="sm", op=OpType.SOFTMAX)],
            edges=[],
        )
        with pytest.raises(ValueError, match="exactly 1 incoming edge"):
            graph.validate()


class TestTopologicalOrder:
    def test_simple_chain(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.FORWARD),
                Node(id="c", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0),
                Edge(src_node="b", src_slot=0, dst_node="c", dst_slot=0),
            ],
        )
        assert graph.topological_order() == ["a", "b", "c"]

    def test_diamond_shape(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.FORWARD),
                Node(id="c", op=OpType.FORWARD),
                Node(id="d", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0),
                Edge(src_node="a", src_slot=0, dst_node="c", dst_slot=0),
                Edge(src_node="b", src_slot=0, dst_node="d", dst_slot=0),
                Edge(src_node="c", src_slot=0, dst_node="d", dst_slot=1),
            ],
        )
        order = graph.topological_order()
        assert order[0] == "a"
        assert order[-1] == "d"
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_cycle_raises(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.FORWARD),
                Node(id="c", op=OpType.FORWARD),
            ],
            edges=[
                Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0),
                Edge(src_node="b", src_slot=0, dst_node="c", dst_slot=0),
                Edge(src_node="c", src_slot=0, dst_node="a", dst_slot=0),
            ],
        )
        with pytest.raises(ValueError, match="cycle"):
            graph.topological_order()
