"""End-to-end tests: compile graph -> serialize -> load in Rust -> run -> verify."""

import pytest
import torch
from meshflow.compiler import compile, CompilerConfig
from meshflow.compiler.artifact import serialize
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType
from meshflow.models.reference import reference_linear, reference_mlp
from meshflow._mesh_runtime import (
    MeshConfig,
    SimInput,
    TaskKind,
    run_program,
    run_simulation,
)


class TestEndToEnd:
    def test_three_node_chain(self) -> None:
        """Compile a 3-node forward chain and run via artifact."""
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
        program = compile(graph)
        artifact_bytes = serialize(program)

        result = run_program(artifact_bytes, inputs={"a": [1.0, 2.0, 3.0]})

        assert result.outputs[(2, 0)] == [1.0, 2.0, 3.0]
        assert result.total_messages >= 1

    def test_single_collect(self) -> None:
        """Single collect node receives input directly."""
        graph = GraphIR(
            nodes=[Node(id="x", op=OpType.COLLECT)],
            edges=[],
        )
        program = compile(graph)
        artifact_bytes = serialize(program)

        result = run_program(artifact_bytes, inputs={"x": [42.0]})

        assert result.outputs[(0, 0)] == [42.0]

    def test_two_node_forward_collect(self) -> None:
        """Simple forward -> collect."""
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.COLLECT),
            ],
            edges=[Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0)],
        )
        program = compile(graph)
        artifact_bytes = serialize(program)

        result = run_program(artifact_bytes, inputs={"a": [5.0, 6.0]})

        assert result.outputs[(1, 0)] == [5.0, 6.0]
        assert result.total_hops == 1


class TestArtifactMatchesManual:
    def test_chain_matches_m1_manual(self) -> None:
        """Artifact path should produce same results as M1 manual SimInput setup."""
        # M2 artifact path
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
        program = compile(graph)
        artifact_bytes = serialize(program)
        artifact_result = run_program(artifact_bytes, inputs={"a": [1.0, 2.0, 3.0]})

        # M1 manual path — equivalent topology
        cfg = MeshConfig(width=3, height=1)
        inp = SimInput()
        inp.add_task(
            coord=(0, 0), kind=TaskKind.ForwardActivation, trigger_slot=0, route_dest=(1, 0)
        )
        inp.add_task(
            coord=(1, 0), kind=TaskKind.ForwardActivation, trigger_slot=0, route_dest=(2, 0)
        )
        inp.add_task(coord=(2, 0), kind=TaskKind.CollectOutput, trigger_slot=0)
        inp.add_message(source=(0, 0), dest=(0, 0), payload=[1.0, 2.0, 3.0])
        manual_result = run_simulation(config=cfg, inputs=inp)

        # Compare outputs
        assert artifact_result.outputs == manual_result.outputs
        assert artifact_result.total_hops == manual_result.total_hops
        assert artifact_result.total_messages == manual_result.total_messages


class TestMultipleInputs:
    def test_two_inputs_to_collect(self) -> None:
        """Two independent input nodes feeding into a shared collect (via separate chains)."""
        # Two independent forward -> collect chains
        graph = GraphIR(
            nodes=[
                Node(id="a", op=OpType.FORWARD),
                Node(id="b", op=OpType.COLLECT),
                Node(id="c", op=OpType.FORWARD),
                Node(id="d", op=OpType.COLLECT),
            ],
            edges=[
                Edge(src_node="a", src_slot=0, dst_node="b", dst_slot=0),
                Edge(src_node="c", src_slot=0, dst_node="d", dst_slot=0),
            ],
        )
        program = compile(graph)
        artifact_bytes = serialize(program)

        result = run_program(
            artifact_bytes,
            inputs={"a": [10.0], "c": [20.0]},
        )

        # Topo order: [a, c, b, d] → placement: a@(0,0), c@(1,0), b@(2,0), d@(3,0)
        assert result.outputs[(2, 0)] == [10.0]  # b collects from a
        assert result.outputs[(3, 0)] == [20.0]  # d collects from c


class TestErrorHandling:
    def test_unknown_input_name(self) -> None:
        graph = GraphIR(
            nodes=[Node(id="x", op=OpType.COLLECT)],
            edges=[],
        )
        program = compile(graph)
        artifact_bytes = serialize(program)

        with pytest.raises(RuntimeError, match="unknown input slot"):
            run_program(artifact_bytes, inputs={"nonexistent": [1.0]})

    def test_malformed_artifact(self) -> None:
        with pytest.raises(ValueError, match="deserialize"):
            run_program(b"not valid msgpack", inputs={})


class TestLinearEndToEnd:
    def test_linear_matches_torch(self) -> None:
        """Compile a tiled linear layer and verify against torch reference."""
        torch.manual_seed(42)
        in_f, out_f = 4, 6

        W = torch.randn(out_f, in_f)
        b = torch.randn(out_f)
        x = torch.randn(in_f)

        # Compile and run on simulator
        graph = GraphIR(
            nodes=[
                Node(
                    id="linear1",
                    op=OpType.LINEAR,
                    attrs={"in_features": in_f, "out_features": out_f},
                )
            ],
            edges=[],
        )
        config = CompilerConfig(mesh_height=4)  # 3 tiles + 1 collect (vertical)
        program = compile(
            graph, config, weights={"linear1": {"weight": W.numpy(), "bias": b.numpy()}}
        )
        artifact_bytes = serialize(program)
        result = run_program(artifact_bytes, inputs={"linear1": x.tolist()})

        # Compare against torch reference — find the single output coord dynamically
        expected = reference_linear(x, W, b)
        assert len(result.outputs) == 1
        actual = torch.tensor(next(iter(result.outputs.values())))
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_linear_single_tile(self) -> None:
        """Degenerate case: one tile PE does the full computation."""
        torch.manual_seed(99)
        in_f, out_f = 3, 2

        W = torch.randn(out_f, in_f)
        b = torch.randn(out_f)
        x = torch.randn(in_f)

        graph = GraphIR(
            nodes=[
                Node(
                    id="lin",
                    op=OpType.LINEAR,
                    attrs={"in_features": in_f, "out_features": out_f},
                )
            ],
            edges=[],
        )
        config = CompilerConfig(mesh_height=2)  # 1 tile + 1 collect (vertical)
        program = compile(graph, config, weights={"lin": {"weight": W.numpy(), "bias": b.numpy()}})
        artifact_bytes = serialize(program)
        result = run_program(artifact_bytes, inputs={"lin": x.tolist()})

        expected = reference_linear(x, W, b)
        assert len(result.outputs) == 1
        actual = torch.tensor(next(iter(result.outputs.values())))
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_linear_profiling(self) -> None:
        """Verify profiling counters for a linear layer execution."""
        torch.manual_seed(42)
        in_f, out_f = 4, 6

        W = torch.randn(out_f, in_f)
        b = torch.randn(out_f)
        x = torch.randn(in_f)

        graph = GraphIR(
            nodes=[
                Node(
                    id="linear1",
                    op=OpType.LINEAR,
                    attrs={"in_features": in_f, "out_features": out_f},
                )
            ],
            edges=[],
        )
        config = CompilerConfig(mesh_height=4)
        program = compile(
            graph, config, weights={"linear1": {"weight": W.numpy(), "bias": b.numpy()}}
        )
        artifact_bytes = serialize(program)
        result = run_program(artifact_bytes, inputs={"linear1": x.tolist()})

        # 3 broadcast input messages + 3 fragment messages from tile PEs
        assert result.total_messages == 6
        # 3 linear completions + 1 concat_collect completion
        assert result.total_tasks_executed == 4
        # Fragment hops: with staggered collect placement (col=0, stagger_offset=0,
        # delta=-1), collect is at edge of column.  Single-column layouts don't
        # benefit from stagger but multi-column layouts get traffic distribution.
        assert result.total_hops == 6


def _make_mlp_weights(
    layer_sizes: list[tuple[int, int]], seed: int = 42
) -> tuple[dict[str, dict[str, "torch.Tensor"]], list[tuple["torch.Tensor", "torch.Tensor"]]]:
    """Create weights for an MLP. Returns (compiler_weights, reference_layers)."""
    torch.manual_seed(seed)
    compiler_weights: dict = {}
    ref_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i, (in_f, out_f) in enumerate(layer_sizes):
        W = torch.randn(out_f, in_f)
        b = torch.randn(out_f)
        name = f"l{i + 1}"
        compiler_weights[name] = {"weight": W.numpy(), "bias": b.numpy()}
        ref_layers.append((W, b))
    return compiler_weights, ref_layers


class TestMLPEndToEnd:
    def test_two_layer_mlp(self) -> None:
        """Linear(4,8) → ReLU → Linear(8,6): verify against reference_mlp."""
        torch.manual_seed(42)
        x = torch.randn(4)
        weights, ref_layers = _make_mlp_weights([(4, 8), (8, 6)])

        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 8}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 8, "out_features": 6}),
            ],
            edges=[
                Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0),
                Edge(src_node="r1", src_slot=0, dst_node="l2", dst_slot=0),
            ],
        )
        config = CompilerConfig(mesh_height=4)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)
        result = run_program(artifact_bytes, inputs={"l1": x.tolist()})

        expected = reference_mlp(x, ref_layers)
        assert len(result.outputs) == 1
        actual = torch.tensor(next(iter(result.outputs.values())))
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_three_layer_mlp(self) -> None:
        """Linear(4,8) -> ReLU -> Linear(8,6) -> ReLU -> Linear(6,3)."""
        torch.manual_seed(42)
        x = torch.randn(4)
        weights, ref_layers = _make_mlp_weights([(4, 8), (8, 6), (6, 3)])

        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 8}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 8, "out_features": 6}),
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
        config = CompilerConfig(mesh_height=4)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)
        result = run_program(artifact_bytes, inputs={"l1": x.tolist()})

        expected = reference_mlp(x, ref_layers)
        assert len(result.outputs) == 1
        actual = torch.tensor(next(iter(result.outputs.values())))
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_uneven_tiling_mlp(self) -> None:
        """MLP with out_features not divisible by num_tiles."""
        torch.manual_seed(42)
        x = torch.randn(4)
        # 7 out_features with mesh_height=4 -> 3 tiles -> [3, 2, 2] rows
        weights, ref_layers = _make_mlp_weights([(4, 7), (7, 3)])

        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 7}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 7, "out_features": 3}),
            ],
            edges=[
                Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0),
                Edge(src_node="r1", src_slot=0, dst_node="l2", dst_slot=0),
            ],
        )
        config = CompilerConfig(mesh_height=4)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)
        result = run_program(artifact_bytes, inputs={"l1": x.tolist()})

        expected = reference_mlp(x, ref_layers)
        assert len(result.outputs) == 1
        actual = torch.tensor(next(iter(result.outputs.values())))
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_single_tile_mlp(self) -> None:
        """Degenerate: mesh_height=2 (1 tile + 1 collect per layer)."""
        torch.manual_seed(42)
        x = torch.randn(3)
        weights, ref_layers = _make_mlp_weights([(3, 4), (4, 2)])

        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 3, "out_features": 4}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 2}),
            ],
            edges=[
                Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0),
                Edge(src_node="r1", src_slot=0, dst_node="l2", dst_slot=0),
            ],
        )
        config = CompilerConfig(mesh_height=2)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)
        result = run_program(artifact_bytes, inputs={"l1": x.tolist()})

        expected = reference_mlp(x, ref_layers)
        assert len(result.outputs) == 1
        actual = torch.tensor(next(iter(result.outputs.values())))
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_mlp_profiling(self) -> None:
        """Verify profiling counters for a 2-layer MLP."""
        torch.manual_seed(42)
        x = torch.randn(4)
        weights, _ = _make_mlp_weights([(4, 6), (6, 3)])

        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 6}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 6, "out_features": 3}),
            ],
            edges=[
                Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0),
                Edge(src_node="r1", src_slot=0, dst_node="l2", dst_slot=0),
            ],
        )
        config = CompilerConfig(mesh_height=4)
        program = compile(graph, config, weights=weights)
        artifact_bytes = serialize(program)
        result = run_program(artifact_bytes, inputs={"l1": x.tolist()})

        # Layer 1: 3 broadcast inputs + 3 fragments to collect = 6 messages
        # Inter-layer: 3 broadcast messages from l1 collect to l2 tiles = 3 messages
        # Layer 2: 3 fragments to collect = 3 messages
        # Total = 6 + 3 + 3 = 12
        assert result.total_messages == 12

        # Layer 1: 3 linear completions + 1 concat_collect_forward completion = 4
        # Layer 2: 3 linear completions + 1 concat_collect completion = 4
        # Total = 8
        assert result.total_tasks_executed == 8


class TestMLPValidationErrors:
    def test_relu_not_after_linear(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="f", op=OpType.FORWARD),
                Node(id="r", op=OpType.RELU),
            ],
            edges=[Edge(src_node="f", src_slot=0, dst_node="r", dst_slot=0)],
        )
        with pytest.raises(ValueError, match="must follow a LINEAR"):
            compile(graph)

    def test_shape_mismatch(self) -> None:
        graph = GraphIR(
            nodes=[
                Node(id="l1", op=OpType.LINEAR, attrs={"in_features": 4, "out_features": 8}),
                Node(id="r1", op=OpType.RELU),
                Node(id="l2", op=OpType.LINEAR, attrs={"in_features": 6, "out_features": 3}),
            ],
            edges=[
                Edge(src_node="l1", src_slot=0, dst_node="r1", dst_slot=0),
                Edge(src_node="r1", src_slot=0, dst_node="l2", dst_slot=0),
            ],
        )
        with pytest.raises(ValueError, match="shape mismatch"):
            compile(
                graph,
                weights={
                    "l1": {"weight": torch.zeros(8, 4).numpy(), "bias": torch.zeros(8).numpy()},
                    "l2": {"weight": torch.zeros(3, 6).numpy(), "bias": torch.zeros(3).numpy()},
                },
            )
