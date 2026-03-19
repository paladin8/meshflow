"""End-to-end tests: compile graph -> serialize -> load in Rust -> run -> verify."""

import pytest
from meshflow.compiler import compile
from meshflow.compiler.artifact import serialize
from meshflow.compiler.graph_ir import Edge, GraphIR, Node, OpType
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
