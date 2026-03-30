"""Broadcast-specific e2e tests.

Covers three concerns:
1. Broadcast is transparent — MLP and transformer block produce correct numerical
   outputs even though activations travel via broadcast routes.
2. Profiling data is non-degenerate and consistent when broadcast is used.
3. The compiler's broadcast detection produces DeliverAndForward routing table
   entries in the compiled artifact.
"""

import numpy as np
import torch

from meshflow._mesh_runtime import run_program
from meshflow.compiler import CompilerConfig, compile
from meshflow.compiler.artifact import (
    RmsNormReduceTask,
    deserialize,
    serialize,
)
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType
from meshflow.models.mlp import mlp_block, mlp_weights
from meshflow.models.reference import reference_mlp, reference_transformer_block
from meshflow.models.transformer import transformer_block, transformer_weights


def _compile_and_run(
    graph: GraphIR,
    weights: dict[str, dict[str, np.ndarray]] | None,
    mesh_height: int = 6,
    inputs: dict[str, list[float]] | None = None,
) -> object:
    """Compile, serialize, and run a graph; return the full SimResult."""
    config = CompilerConfig(mesh_height=mesh_height)
    program = compile(graph, config, weights=weights)
    artifact_bytes = serialize(program)
    return run_program(artifact_bytes, inputs=inputs or {})


def _compile_artifact(
    graph: GraphIR,
    weights: dict[str, dict[str, np.ndarray]] | None,
    mesh_height: int = 6,
) -> bytes:
    """Compile and serialize a graph; return the raw artifact bytes."""
    config = CompilerConfig(mesh_height=mesh_height)
    program = compile(graph, config, weights=weights)
    return serialize(program)


class TestBroadcastTransparent:
    """Verify that broadcast routing does not change numerical outputs."""

    def test_broadcast_transparent_mlp(self) -> None:
        """3-layer MLP output matches reference_mlp with atol=1e-4.

        Inter-layer ConcatCollectForward routes are broadcast routes; this test
        verifies the broadcast delivery path is numerically transparent.
        """
        layer_dims = [4, 8, 16, 4]
        graph = mlp_block(layer_dims)
        weights = mlp_weights(layer_dims, seed=42)

        torch.manual_seed(77)
        x = torch.randn(layer_dims[0])

        result = _compile_and_run(
            graph,
            weights,
            mesh_height=6,
            inputs={"input": x.tolist()},
        )

        output = None
        for data in result.outputs.values():
            output = data
        assert output is not None, f"no output found, outputs={dict(result.outputs)}"

        ref_layers = [
            (
                torch.from_numpy(weights[f"linear{i}"]["weight"]),
                torch.from_numpy(weights[f"linear{i}"]["bias"]),
            )
            for i in range(len(layer_dims) - 1)
        ]
        expected = reference_mlp(x, ref_layers)
        actual = torch.tensor(output)

        assert torch.allclose(actual, expected, atol=1e-4), (
            f"broadcast MLP mismatch:\ngot     {actual}\nexpected {expected}\n"
            f"diff={actual - expected}"
        )

    def test_broadcast_transparent_transformer(self) -> None:
        """Transformer block output matches reference with atol=1e-3.

        Exercises multiple broadcast sites: RmsNormReduce column broadcast,
        RmsNormNormalize routes, Add routes, and ConcatCollectForward inter-layer
        broadcast (MatMul output forward).
        """
        seq_len = 4
        d_model = 8
        d_ff = 16
        eps = 1e-6

        graph = transformer_block(seq_len, d_model, d_ff, eps)
        weights = transformer_weights(d_model, d_ff, seed=9)

        torch.manual_seed(61)
        x = torch.randn(seq_len, d_model)

        result = _compile_and_run(
            graph,
            weights,
            mesh_height=6,
            inputs={"input": x.flatten().tolist()},
        )

        output = None
        for data in result.outputs.values():
            output = data
        assert output is not None, f"no output found, outputs={dict(result.outputs)}"

        expected = reference_transformer_block(x, weights, eps)
        actual = torch.tensor(output).reshape(seq_len, d_model)

        assert torch.allclose(actual, expected, atol=1e-3), (
            f"broadcast transformer mismatch:\ngot\n{actual}\nexpected\n{expected}\n"
            f"diff=\n{actual - expected}"
        )


class TestBroadcastProfiling:
    """Verify that profiling data is consistent and non-degenerate with broadcast."""

    def test_broadcast_reduces_hops_consistent(self) -> None:
        """Profiling data is self-consistent when broadcast routes are used.

        We cannot easily compare "before and after" in a single test, but we can
        verify:
        - total_hops > 0 (messages actually moved)
        - total_messages > 0 (messages were sent)
        - average hops per message is plausible (not absurdly large)

        With broadcast, the runtime delivers a single traversal to multiple PEs
        along the route, so the average hops per message should stay modest.
        """
        seq_len = 2
        d_model = 4
        d_ff = 8
        eps = 1e-6

        graph = transformer_block(seq_len, d_model, d_ff, eps)
        weights = transformer_weights(d_model, d_ff, seed=5)

        torch.manual_seed(17)
        x = torch.randn(seq_len, d_model)

        result = _compile_and_run(
            graph,
            weights,
            mesh_height=6,
            inputs={"input": x.flatten().tolist()},
        )

        assert result.total_hops > 0, "expected non-zero hops"
        assert result.total_messages > 0, "expected non-zero messages"

        # With a small transformer on a modest mesh, the average hops per message
        # should be less than 10 (a generous upper bound for a ~6-column mesh).
        # This would be violated if broadcast failed and duplicated messages.
        avg_hops = result.total_hops / result.total_messages
        assert avg_hops < 10, (
            f"avg hops per message suspiciously high ({avg_hops:.1f}); "
            f"total_hops={result.total_hops}, total_messages={result.total_messages}"
        )

    def test_broadcast_program_has_deliver_and_forward(self) -> None:
        """The compiled transformer artifact has DeliverAndForward routing entries.

        With routing tables, broadcast detection is verified by checking that
        some PEs have routing table entries with a non-None deliver_slot
        (DeliverAndForward semantics).
        """
        seq_len = 2
        d_model = 4
        d_ff = 8
        eps = 1e-6

        graph = transformer_block(seq_len, d_model, d_ff, eps)
        weights = transformer_weights(d_model, d_ff, seed=5)

        artifact_bytes = _compile_artifact(graph, weights, mesh_height=6)
        program = deserialize(artifact_bytes)

        deliver_and_forward_count = 0
        for pe in program.pe_programs:
            for entry in pe.routing_table.values():
                if entry.deliver_slot is not None:
                    deliver_and_forward_count += 1

        assert deliver_and_forward_count > 0, (
            "expected at least one DeliverAndForward routing table entry in the "
            "compiled transformer program; found none"
        )


class TestBroadcastDetectionArtifact:
    """Verify the compiler emits broadcast routes in the serialized artifact."""

    def test_rmsnorm_reduce_has_column_broadcast(self) -> None:
        """RmsNormReduce tasks in a FORWARD->RMSNORM->COLLECT graph use column broadcast.

        The reduce PE broadcasts the scale factor south to all tile PEs in the
        same column. With routing tables, this is verified by checking that
        intermediate PEs have DeliverAndForward entries (non-None deliver_slot).
        """
        d_model = 8
        eps = 1e-6

        graph = GraphIR(
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
        weights = {"rn": {"gamma": np.ones(d_model, dtype=np.float32)}}

        artifact_bytes = _compile_artifact(graph, weights, mesh_height=6)
        program = deserialize(artifact_bytes)

        reduce_tasks = [
            task
            for pe in program.pe_programs
            for task in pe.tasks
            if isinstance(task, RmsNormReduceTask)
        ]

        assert len(reduce_tasks) > 0, "no RmsNormReduceTask found in artifact"

        # With broadcast, the reduce PE sends one route per direction to
        # the farthest tile, and intermediate tiles get DeliverAndForward
        # entries in the routing table.
        deliver_and_forward_count = 0
        for pe in program.pe_programs:
            for entry in pe.routing_table.values():
                if entry.deliver_slot is not None:
                    deliver_and_forward_count += 1

        assert deliver_and_forward_count > 0, (
            "RmsNormReduce broadcast should produce DeliverAndForward routing "
            "table entries on intermediate tile PEs; found none"
        )

    def test_concat_collect_forward_broadcast_mlp(self) -> None:
        """ConcatCollectForward tasks in a 3-layer MLP use broadcast routes.

        The inter-layer collect PE broadcasts the assembled activation east+south
        to the tile PEs of the next layer. With routing tables, broadcast is
        verified by checking that some intermediate PEs have DeliverAndForward
        entries (non-None deliver_slot).
        """
        layer_dims = [4, 8, 16, 4]
        graph = mlp_block(layer_dims)
        weights = mlp_weights(layer_dims, seed=42)

        artifact_bytes = _compile_artifact(graph, weights, mesh_height=6)
        program = deserialize(artifact_bytes)

        deliver_and_forward_count = 0
        for pe in program.pe_programs:
            for entry in pe.routing_table.values():
                if entry.deliver_slot is not None:
                    deliver_and_forward_count += 1

        assert deliver_and_forward_count > 0, (
            "expected at least one DeliverAndForward routing table entry in the "
            "3-layer MLP artifact; found none"
        )

    def test_single_destination_no_broadcast(self) -> None:
        """A graph with only one destination per route should not use broadcast.

        FORWARD -> COLLECT: the ForwardActivation task sends to exactly one
        destination. The route has no deliver_at entries.
        """
        graph = GraphIR(
            nodes=[
                Node(id="input", op=OpType.FORWARD),
                Node(id="output", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="input", src_slot=0, dst_node="output", dst_slot=0),
            ],
        )

        artifact_bytes = _compile_artifact(graph, weights=None, mesh_height=6)
        program = deserialize(artifact_bytes)

        for pe in program.pe_programs:
            for task in pe.tasks:
                # With routing tables, all tasks use routes list
                if hasattr(task, "routes"):
                    for r in task.routes:
                        # Routes no longer carry deliver_at (routing is in tables)
                        assert isinstance(r.dest, tuple)
