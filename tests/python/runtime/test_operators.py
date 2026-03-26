"""Operator e2e tests: compile graph -> serialize -> run -> validate against torch reference."""

import numpy as np
import torch

from meshflow._mesh_runtime import run_program
from meshflow.compiler import CompilerConfig, compile
from meshflow.compiler.artifact import serialize
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType
from meshflow.models.mlp import mlp_block, mlp_weights
from meshflow.models.reference import reference_mlp, reference_rmsnorm


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
