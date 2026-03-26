"""Operator e2e tests: compile graph -> serialize -> run -> validate against torch reference."""

import numpy as np
import torch

from meshflow._mesh_runtime import MeshConfig, SimInput, TaskKind, run_program, run_simulation
from meshflow.compiler import CompilerConfig, compile
from meshflow.compiler.artifact import serialize
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType
from meshflow.models.mlp import mlp_block, mlp_weights
from meshflow.models.reference import reference_mlp, reference_rmsnorm, reference_transformer_block
from meshflow.models.transformer import transformer_block, transformer_weights


def _run_graph(
    graph: GraphIR,
    weights: dict[str, dict[str, np.ndarray]] | None,
    mesh_height: int = 6,
    inputs: dict[str, list[float]] | None = None,
) -> list[float]:
    """Compile, serialize, run, and extract output from a graph.

    Returns the output tensor as a flat list of floats from the first output found.
    """
    config = CompilerConfig(mesh_height=mesh_height)
    program = compile(graph, config, weights=weights)
    artifact_bytes = serialize(program)

    run_inputs: dict[str, list[float]] = inputs if inputs is not None else {}
    result = run_program(artifact_bytes, inputs=run_inputs)

    output = None
    for data in result.outputs.values():
        output = data
    assert output is not None, f"no output found, outputs={dict(result.outputs)}"
    return output


def _run_graph_result(
    graph: GraphIR,
    weights: dict[str, dict[str, np.ndarray]] | None,
    mesh_height: int = 6,
    inputs: dict[str, list[float]] | None = None,
) -> object:
    """Compile, serialize, run, and return the full SimResult (with final_timestamp)."""
    config = CompilerConfig(mesh_height=mesh_height)
    program = compile(graph, config, weights=weights)
    artifact_bytes = serialize(program)

    run_inputs: dict[str, list[float]] = inputs if inputs is not None else {}
    return run_program(artifact_bytes, inputs=run_inputs)


class TestMlpHelper:
    """End-to-end tests using the mlp_block() and mlp_weights() helpers."""

    def test_two_layer(self) -> None:
        """mlp_block([4, 8, 4]) with random weights matches reference_mlp."""
        layer_dims = [4, 8, 4]
        graph = mlp_block(layer_dims)
        weights = mlp_weights(layer_dims, seed=42)

        torch.manual_seed(7)
        x = torch.randn(layer_dims[0])

        output = _run_graph(
            graph,
            weights,
            mesh_height=6,
            inputs={"input": x.tolist()},
        )

        # Build reference layers from the same weights
        ref_layers = [
            (
                torch.from_numpy(weights[f"linear{i}"]["weight"]),
                torch.from_numpy(weights[f"linear{i}"]["bias"]),
            )
            for i in range(len(layer_dims) - 1)
        ]
        expected = reference_mlp(x, ref_layers)
        actual = torch.tensor(output)

        assert torch.allclose(actual, expected, atol=1e-5), (
            f"two-layer MLP mismatch:\ngot     {actual}\nexpected {expected}\n"
            f"diff={actual - expected}"
        )

    def test_three_layer(self) -> None:
        """mlp_block([4, 8, 16, 4]) with random weights matches reference_mlp."""
        layer_dims = [4, 8, 16, 4]
        graph = mlp_block(layer_dims)
        weights = mlp_weights(layer_dims, seed=13)

        torch.manual_seed(99)
        x = torch.randn(layer_dims[0])

        output = _run_graph(
            graph,
            weights,
            mesh_height=6,
            inputs={"input": x.tolist()},
        )

        # Build reference layers from the same weights
        ref_layers = [
            (
                torch.from_numpy(weights[f"linear{i}"]["weight"]),
                torch.from_numpy(weights[f"linear{i}"]["bias"]),
            )
            for i in range(len(layer_dims) - 1)
        ]
        expected = reference_mlp(x, ref_layers)
        actual = torch.tensor(output)

        assert torch.allclose(actual, expected, atol=1e-5), (
            f"three-layer MLP mismatch:\ngot     {actual}\nexpected {expected}\n"
            f"diff={actual - expected}"
        )


class TestRmsNorm:
    """End-to-end tests for a minimal FORWARD → RMSNORM → COLLECT graph."""

    @staticmethod
    def _make_rmsnorm_graph(d_model: int, eps: float = 1e-6) -> GraphIR:
        """Build a minimal FORWARD → RMSNORM → COLLECT GraphIR."""
        return GraphIR(
            nodes=[
                Node(id="input", op=OpType.FORWARD),
                Node(
                    id="rn",
                    op=OpType.RMSNORM,
                    attrs={"eps": eps, "feature_count": d_model},
                ),
                Node(id="output", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="input", src_slot=0, dst_node="rn", dst_slot=0),
                Edge(src_node="rn", src_slot=0, dst_node="output", dst_slot=0),
            ],
        )

    def test_single_position(self) -> None:
        """d_model=8, single input vector matches reference_rmsnorm with atol=1e-4."""
        d_model = 8
        eps = 1e-6

        torch.manual_seed(11)
        x = torch.randn(d_model)
        gamma = torch.ones(d_model)

        graph = self._make_rmsnorm_graph(d_model, eps)
        weights = {"rn": {"gamma": gamma.numpy().astype(np.float32)}}

        output = _run_graph(
            graph,
            weights,
            mesh_height=6,
            inputs={"input": x.tolist()},
        )

        expected = reference_rmsnorm(x, gamma, eps)
        actual = torch.tensor(output)

        assert torch.allclose(actual, expected, atol=1e-4), (
            f"single-position RmsNorm mismatch:\ngot      {actual}\nexpected {expected}\n"
            f"diff={actual - expected}"
        )

    def test_multi_position(self) -> None:
        """seq_len=3, d_model=8 — three positions match per-position reference_rmsnorm."""
        seq_len = 3
        d_model = 8
        eps = 1e-6

        torch.manual_seed(22)
        x = torch.randn(seq_len, d_model)
        gamma = torch.ones(d_model)

        graph = self._make_rmsnorm_graph(d_model, eps)
        weights = {"rn": {"gamma": gamma.numpy().astype(np.float32)}}

        output = _run_graph(
            graph,
            weights,
            mesh_height=6,
            inputs={"input": x.flatten().tolist()},
        )

        expected = reference_rmsnorm(x, gamma, eps)  # (seq_len, d_model)
        actual = torch.tensor(output).reshape(seq_len, d_model)

        assert torch.allclose(actual, expected, atol=1e-4), (
            f"multi-position RmsNorm mismatch:\ngot\n{actual}\nexpected\n{expected}\n"
            f"diff=\n{actual - expected}"
        )

    def test_non_divisible_features(self) -> None:
        """d_model=7, mesh_height=5 — uneven tile distribution in RmsNorm slicing."""
        d_model = 7
        eps = 1e-6
        mesh_height = 5

        torch.manual_seed(33)
        x = torch.randn(d_model)
        gamma = torch.ones(d_model)

        graph = self._make_rmsnorm_graph(d_model, eps)
        weights = {"rn": {"gamma": gamma.numpy().astype(np.float32)}}

        output = _run_graph(
            graph,
            weights,
            mesh_height=mesh_height,
            inputs={"input": x.tolist()},
        )

        expected = reference_rmsnorm(x, gamma, eps)
        actual = torch.tensor(output)

        assert torch.allclose(actual, expected, atol=1e-4), (
            f"non-divisible RmsNorm mismatch:\ngot      {actual}\nexpected {expected}\n"
            f"diff={actual - expected}"
        )


class TestSoftmax:
    """End-to-end tests for SOFTMAX via a transformer block attention chain.

    Softmax is tested through the attention chain because the route pass only generates
    co-located Softmax tasks within attention chains.
    """

    def test_softmax_in_attention_chain(self) -> None:
        """seq_len=2, d_model=4, d_ff=8 — full transformer block matches reference."""
        seq_len = 2
        d_model = 4
        d_ff = 8
        eps = 1e-6

        graph = transformer_block(seq_len, d_model, d_ff, eps)
        weights = transformer_weights(d_model, d_ff, seed=5)

        torch.manual_seed(17)
        x = torch.randn(seq_len, d_model)

        output = _run_graph(
            graph,
            weights,
            mesh_height=6,
            inputs={"input": x.flatten().tolist()},
        )

        expected = reference_transformer_block(x, weights, eps)
        actual = torch.tensor(output).reshape(seq_len, d_model)

        assert torch.allclose(actual, expected, atol=1e-3), (
            f"softmax attention chain mismatch:\ngot\n{actual}\nexpected\n{expected}\n"
            f"diff=\n{actual - expected}"
        )

    def test_softmax_numerical_stability(self) -> None:
        """Scale Q/K weights by 100x — verify no NaN/Inf and match reference within atol=1e-2."""
        seq_len = 2
        d_model = 4
        d_ff = 8
        eps = 1e-6

        graph = transformer_block(seq_len, d_model, d_ff, eps)
        weights = transformer_weights(d_model, d_ff, seed=5)

        # Scale up Q and K projection weights to produce large attention scores
        weights["q_proj"]["weight"] = (weights["q_proj"]["weight"] * 100.0).astype(np.float32)
        weights["k_proj"]["weight"] = (weights["k_proj"]["weight"] * 100.0).astype(np.float32)

        torch.manual_seed(17)
        x = torch.randn(seq_len, d_model)

        output = _run_graph(
            graph,
            weights,
            mesh_height=6,
            inputs={"input": x.flatten().tolist()},
        )

        actual = torch.tensor(output).reshape(seq_len, d_model)

        assert not torch.any(torch.isnan(actual)), f"NaN detected in output: {actual}"
        assert not torch.any(torch.isinf(actual)), f"Inf detected in output: {actual}"

        expected = reference_transformer_block(x, weights, eps)
        assert torch.allclose(actual, expected, atol=1e-2), (
            f"softmax numerical stability mismatch:\ngot\n{actual}\nexpected\n{expected}\n"
            f"diff=\n{actual - expected}"
        )


class TestAdd:
    """End-to-end tests for a standalone ADD graph: two FORWARD nodes → ADD → COLLECT."""

    @staticmethod
    def _make_add_graph() -> GraphIR:
        """Build a minimal two-input ADD GraphIR."""
        return GraphIR(
            nodes=[
                Node(id="input_a", op=OpType.FORWARD),
                Node(id="input_b", op=OpType.FORWARD),
                Node(id="add", op=OpType.ADD),
                Node(id="output", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="input_a", src_slot=0, dst_node="add", dst_slot=0),
                Edge(src_node="input_b", src_slot=0, dst_node="add", dst_slot=1),
                Edge(src_node="add", src_slot=0, dst_node="output", dst_slot=0),
            ],
        )

    def test_basic(self) -> None:
        """[1,2,3,4] + [10,20,30,40] = [11,22,33,44]."""
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = torch.tensor([10.0, 20.0, 30.0, 40.0])

        graph = self._make_add_graph()
        output = _run_graph(
            graph, weights={}, inputs={"input_a": a.tolist(), "input_b": b.tolist()}
        )

        expected = a + b
        actual = torch.tensor(output)

        assert torch.allclose(actual, expected), (
            f"basic ADD mismatch:\ngot      {actual}\nexpected {expected}\ndiff={actual - expected}"
        )

    def test_multi_position(self) -> None:
        """8-element vectors added element-wise."""
        torch.manual_seed(55)
        a = torch.randn(8)
        b = torch.randn(8)

        graph = self._make_add_graph()
        output = _run_graph(
            graph, weights={}, inputs={"input_a": a.tolist(), "input_b": b.tolist()}
        )

        expected = a + b
        actual = torch.tensor(output)

        assert torch.allclose(actual, expected), (
            f"multi-position ADD mismatch:\ngot      {actual}\nexpected {expected}\n"
            f"diff={actual - expected}"
        )


class TestMatMul:
    """End-to-end tests for MATMUL via a transformer block attention chain.

    MatMul is tested through the attention chain because standalone MATMUL routing
    is not supported; the route pass only generates MatMul tasks within attention chains.
    """

    def test_attention_matmul_small(self) -> None:
        """seq_len=2, d_model=4, d_ff=8 — transformer block with small dimensions."""
        seq_len = 2
        d_model = 4
        d_ff = 8
        eps = 1e-6

        graph = transformer_block(seq_len, d_model, d_ff, eps)
        weights = transformer_weights(d_model, d_ff, seed=3)

        torch.manual_seed(41)
        x = torch.randn(seq_len, d_model)

        output = _run_graph(
            graph,
            weights,
            mesh_height=6,
            inputs={"input": x.flatten().tolist()},
        )

        expected = reference_transformer_block(x, weights, eps)
        actual = torch.tensor(output).reshape(seq_len, d_model)

        assert torch.allclose(actual, expected, atol=1e-3), (
            f"attention matmul small mismatch:\ngot\n{actual}\nexpected\n{expected}\n"
            f"diff=\n{actual - expected}"
        )

    def test_attention_matmul_larger(self) -> None:
        """seq_len=4, d_model=8, d_ff=16 — transformer block with larger dimensions."""
        seq_len = 4
        d_model = 8
        d_ff = 16
        eps = 1e-6

        graph = transformer_block(seq_len, d_model, d_ff, eps)
        weights = transformer_weights(d_model, d_ff, seed=9)

        torch.manual_seed(61)
        x = torch.randn(seq_len, d_model)

        output = _run_graph(
            graph,
            weights,
            mesh_height=6,
            inputs={"input": x.flatten().tolist()},
        )

        expected = reference_transformer_block(x, weights, eps)
        actual = torch.tensor(output).reshape(seq_len, d_model)

        assert torch.allclose(actual, expected, atol=1e-3), (
            f"attention matmul larger mismatch:\ngot\n{actual}\nexpected\n{expected}\n"
            f"diff=\n{actual - expected}"
        )


class TestCostModel:
    """Tests for data-proportional cost model behavior."""

    def test_linear_cost_scales_with_dimensions(self) -> None:
        """Larger MLP dimensions produce higher final_timestamp."""
        torch.manual_seed(100)
        x_small = torch.randn(4)
        x_large = torch.randn(4)

        # Small MLP: [4, 4]
        small_graph = mlp_block([4, 4])
        small_weights = mlp_weights([4, 4], seed=1)
        small_result = _run_graph_result(
            small_graph, small_weights, inputs={"input": x_small.tolist()}
        )

        # Large MLP: [4, 16]
        large_graph = mlp_block([4, 16])
        large_weights = mlp_weights([4, 16], seed=1)
        large_result = _run_graph_result(
            large_graph, large_weights, inputs={"input": x_large.tolist()}
        )

        assert large_result.final_timestamp > small_result.final_timestamp, (
            f"Expected larger MLP to take longer: "
            f"small={small_result.final_timestamp}, large={large_result.final_timestamp}"
        )

    def test_matmul_cost_scales_with_matrix_size(self) -> None:
        """Larger transformer dimensions produce higher final_timestamp."""
        torch.manual_seed(200)

        # Small transformer: seq_len=2, d_model=4
        small_graph = transformer_block(seq_len=2, d_model=4, d_ff=8, eps=1e-6)
        small_weights = transformer_weights(d_model=4, d_ff=8, seed=2)
        x_small = torch.randn(2, 4)
        small_result = _run_graph_result(
            small_graph,
            small_weights,
            mesh_height=6,
            inputs={"input": x_small.flatten().tolist()},
        )

        # Large transformer: seq_len=4, d_model=8
        large_graph = transformer_block(seq_len=4, d_model=8, d_ff=16, eps=1e-6)
        large_weights = transformer_weights(d_model=8, d_ff=16, seed=2)
        x_large = torch.randn(4, 8)
        large_result = _run_graph_result(
            large_graph,
            large_weights,
            mesh_height=6,
            inputs={"input": x_large.flatten().tolist()},
        )

        assert large_result.final_timestamp > small_result.final_timestamp, (
            f"Expected larger transformer to take longer: "
            f"small={small_result.final_timestamp}, large={large_result.final_timestamp}"
        )

    def test_cost_per_element_configurable(self) -> None:
        """cost_per_element plumbing works: FORWARD->COLLECT with 0 elements is unaffected."""
        # ForwardActivation has 0 elements, so cost_per_element should not matter.
        config_low = MeshConfig(width=3, height=1, cost_per_element=1)
        config_high = MeshConfig(width=3, height=1, cost_per_element=10)

        def build_input() -> SimInput:
            inp = SimInput()
            inp.add_task((0, 0), TaskKind.ForwardActivation, 0, route_dest=(2, 0))
            inp.add_task((2, 0), TaskKind.CollectOutput, 0)
            inp.add_message((0, 0), (0, 0), [1.0, 2.0, 3.0], 0)
            return inp

        result_low = run_simulation(config_low, build_input())
        result_high = run_simulation(config_high, build_input())

        assert result_low.final_timestamp == result_high.final_timestamp, (
            f"ForwardActivation (0 elements) should be unaffected by cost_per_element: "
            f"low={result_low.final_timestamp}, high={result_high.final_timestamp}"
        )

    def test_forward_activation_zero_element_cost(self) -> None:
        """FORWARD->COLLECT graph completes with low final_timestamp (< 20)."""
        graph = GraphIR(
            nodes=[
                Node(id="input", op=OpType.FORWARD),
                Node(id="output", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="input", src_slot=0, dst_node="output", dst_slot=0),
            ],
        )
        result = _run_graph_result(graph, weights={}, inputs={"input": [1.0, 2.0, 3.0]})

        assert result.final_timestamp < 20, (
            f"FORWARD->COLLECT should be fast (0 elements): "
            f"final_timestamp={result.final_timestamp}"
        )
